from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


DEFAULT_INPUT = Path("data/raw/jpm-20251231.htm")
DEFAULT_OUTPUT = Path("data/processed/jpm_2025_10k_chunks.jsonl")

COMPANY = "JPMorgan Chase & Co."
DOC_TYPE = "10-K"
FILING_DATE = "2026-02-13"
MIN_TEXT_WORDS = 35
TINY_TEXT_CHARS = 150
TARGET_TEXT_MIN_CHARS = 300
TARGET_TEXT_MAX_CHARS = 1200
HARD_TEXT_MAX_CHARS = 1500
MAX_TEXT_CHARS = TARGET_TEXT_MAX_CHARS
MAX_TABLE_CELLS = 35
MAX_FINAL_TEXT_CHARS = TARGET_TEXT_MAX_CHARS


@dataclass
class HtmlBlock:
    text: str
    in_table: bool


@dataclass
class SectionSpec:
    key: str
    label: str
    start_pattern: re.Pattern[str]
    end_pattern: re.Pattern[str]
    start_after: int = 0


class VisibleTextParser(HTMLParser):
    """Extract visible block-like text from the inline XBRL HTML filing."""

    block_tags = {"div", "p", "tr", "li", "br", "hr"}
    flush_tags = {"div", "p", "tr", "li", "td", "th", "br", "hr"}
    hidden_tags = {"head", "script", "style"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._hidden_stack = [False]
        self._table_depth = 0
        self._buffer: list[str] = []
        self._buffer_in_table = False
        self.blocks: list[HtmlBlock] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name: value or "" for name, value in attrs}
        style = attributes.get("style", "").lower().replace(" ", "")
        hidden = (
            self._hidden_stack[-1]
            or tag.lower() in self.hidden_tags
            or "display:none" in style
            or attributes.get("hidden", "").lower() in {"hidden", "true"}
        )
        self._hidden_stack.append(hidden)
        if tag.lower() == "table" and not hidden:
            self._table_depth += 1
        if not hidden and tag.lower() in self.block_tags:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        hidden = self._hidden_stack[-1]
        if not hidden and tag in self.flush_tags:
            self._flush()
        if tag == "table" and not hidden and self._table_depth:
            self._table_depth -= 1
        if len(self._hidden_stack) > 1:
            self._hidden_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._hidden_stack[-1]:
            return
        text = data.replace("\xa0", " ")
        if not text.strip():
            return
        self._buffer.append(text)
        self._buffer_in_table = self._buffer_in_table or self._table_depth > 0

    def close(self) -> None:
        self._flush()
        super().close()

    def _flush(self) -> None:
        if not self._buffer:
            return
        text = normalize_spaces(" ".join(self._buffer))
        if text:
            self.blocks.append(HtmlBlock(text=text, in_table=self._buffer_in_table))
        self._buffer = []
        self._buffer_in_table = False


def normalize_spaces(text: str) -> str:
    text = text.replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:%)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return text.strip()


def parse_visible_blocks(input_path: Path) -> list[HtmlBlock]:
    parser = VisibleTextParser()
    parser.feed(input_path.read_text(encoding="utf-8", errors="ignore"))
    parser.close()
    return parser.blocks


def find_section_bounds(blocks: list[HtmlBlock], spec: SectionSpec) -> tuple[int, int]:
    start = None
    for index in range(spec.start_after, len(blocks)):
        if spec.start_pattern.fullmatch(blocks[index].text):
            start = index
            break
    if start is None:
        raise ValueError(f"Could not find section start for {spec.label}")

    for index in range(start + 1, len(blocks)):
        if spec.end_pattern.fullmatch(blocks[index].text):
            return start, index
    raise ValueError(f"Could not find section end for {spec.label}")


def find_mda_bounds(blocks: list[HtmlBlock]) -> tuple[int, int]:
    start = None
    for index in range(len(blocks) - 1):
        if (
            blocks[index].text == "Management's discussion and analysis"
            and blocks[index + 1].text.startswith("The following is Management's discussion and analysis")
        ):
            start = index
            break
    if start is None:
        raise ValueError("Could not find Item 7 MD&A start")

    for index in range(start + 1, len(blocks)):
        if blocks[index].text == "Management's report on internal control over financial reporting":
            return start, index
    raise ValueError("Could not find Item 7 MD&A end")


def should_skip_block(block: HtmlBlock) -> bool:
    text = block.text
    if not text:
        return True
    if text in {"Part I", "Table of contents"}:
        return True
    if re.fullmatch(r"\d{1,3}", text):
        return True
    if re.fullmatch(r"[$%()\-.,\d]+", text):
        return True
    if "JPMorgan Chase & Co./2025 Form 10-K" in text:
        return True
    if "makes available on its website" in text:
        return True
    if "provided in the Management's discussion" in text:
        return True
    if "is included in both JPMorganChase's Annual Report" in text:
        return True
    if "does not contain all of the information" in text:
        return True
    if "should be read in its entirety" in text:
        return True
    if "information provided below" in text.lower():
        return True
    if "forward-looking statements" in text.lower():
        return True
    if "private securities litigation reform act" in text.lower():
        return True
    if "readers should not consider" in text.lower():
        return True
    if text.lower().startswith("refer to ") and re.search(r"\bpages?\b|\bnote\b", text, re.I):
        return True
    if text.lower().startswith("for additional information, refer to"):
        return True
    if text.lower().startswith("for more information") and "refer to" in text.lower():
        return True
    if text.lower().startswith("for a further discussion") and "refer to" in text.lower():
        return True
    if text.lower().startswith("the following table"):
        return True
    if text.lower().startswith("in the following table"):
        return True
    if text.lower().startswith("in the following tables"):
        return True
    if re.match(r"^\([a-z]\)\s", text):
        return True
    return False


def clean_paragraph(text: str) -> str:
    text = normalize_spaces(text)
    text = re.sub(
        r"\s*(?:For (?:more|further|additional) information[^.]*,\s*)?refer to [^.]+(?:\.|$)",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\s*For additional discussion[^.]*refer to [^.]+(?:\.|$)",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\s*The following tables? (?:summarizes?|provides?|presents?|sets forth)[^.]+(?:\.|$)",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\s*The table below [^.]+(?:\.|$)",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\s*The following graph [^.]+(?:\.|$)",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s+provided in the following table", "", text, flags=re.I)
    text = re.sub(
        r"\s*[^.]*\b(?:provided|reported|captured|presented)\s+in\s+(?:the\s+)?(?:table below|following table)[^.]*[.:]?",
        "",
        text,
        flags=re.I,
    )
    text = normalize_spaces(text)
    return text


def is_heading(text: str, section_label: str) -> bool:
    if not text or text.startswith("• "):
        return False
    if re.fullmatch(r"Item\s+\d+[A-Z]?\..*", text):
        return True
    if text.endswith(":") and len(text) <= 140:
        return True
    if len(text) <= 95 and not text.endswith(".") and count_words(text) <= 12:
        return not looks_like_metric_row(text)
    if text.isupper() and len(text) <= 120:
        return True
    if section_label == "Item 1A Risk Factors":
        if len(text) <= 220 and text.endswith(".") and count_words(text) <= 28:
            risk_starters = (
                "JPMorganChase",
                "The Firm",
                "The financial",
                "Unfavorable",
                "A reduction",
                "A failure",
                "Damage",
                "Employee",
                "An outbreak",
                "The laws",
            )
            return text.startswith(risk_starters)
    return False


def looks_like_metric_row(text: str) -> bool:
    if re.search(r"\d", text) and count_words(text) <= 5:
        return True
    table_labels = {
        "Year ended December 31,",
        "Selected income statement data",
        "Selected ratios and metrics",
        "As of or for the year ended December 31,",
        "Page",
    }
    return text in table_labels


def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


def semantic_topic(title: str, section_label: str, text: str = "") -> str:
    raw_title = re.sub(r"^Item\s+\d+[A-Z]?\.\s*", "", title).strip().rstrip(".:")
    haystack = f"{raw_title} {text}".lower()
    if "principal bank subsidiary" in haystack:
        return "Subsidiaries and operating structure"
    if "liquidity risk" in haystack:
        return "Liquidity risk management"
    if "jpmse" in haystack or "j.p. morgan se" in haystack or "j.p. morgan securities plc" in haystack:
        return "Regulatory capital for subsidiaries"
    if section_label == "Item 7 MD&A" and re.search(r"\bvar\b", haystack):
        return "Value-at-risk"

    topic_rules = [
        ("principal bank subsidiary", "Subsidiaries and operating structure"),
        ("operations worldwide", "Firm business overview"),
        ("business segments", "Business segments"),
        ("competition", "Competitive environment"),
        ("supervision and regulation", "Regulatory environment"),
        ("financial holding company", "Financial holding company regulation"),
        ("subsidiary banks", "Bank subsidiary regulation"),
        ("securities and broker-dealer", "Broker-dealer regulation"),
        ("investment management regulation", "Investment management regulation"),
        ("derivatives regulation", "Derivatives regulation"),
        ("data, privacy, cybersecurity", "Data and cybersecurity regulation"),
        ("bank secrecy act", "Financial crimes compliance"),
        ("anti-corruption", "Anti-corruption compliance"),
        ("compensation practices", "Compensation governance"),
        ("sustainability", "Sustainability regulation"),
        ("human capital", "Human capital strategy"),
        ("global workforce", "Workforce profile"),
        ("attracting and retaining", "Talent attraction and retention"),
        ("developing employees", "Employee development"),
        ("rewarding and supporting", "Employee compensation and benefits"),
        ("executive overview", "Financial performance overview"),
        ("consolidated results", "Consolidated financial performance"),
        ("balance sheets", "Balance sheet and cash flow trends"),
        ("non-gaap", "Non-GAAP financial measures"),
        ("consumer & community banking", "Consumer banking performance"),
        ("commercial & investment bank", "Commercial and investment banking performance"),
        ("asset & wealth management", "Asset and wealth management performance"),
        ("treasury and cio", "Treasury and CIO activities"),
        ("firmwide risk management", "Firmwide risk management"),
        ("three lines of defense", "Risk governance model"),
        ("risk governance", "Risk governance"),
        ("risk identification", "Risk ownership"),
        ("board oversight", "Board risk oversight"),
        ("management oversight", "Management risk oversight"),
        ("risk appetite", "Risk appetite"),
        ("strategic risk", "Strategic risk management"),
        ("capital risk", "Capital risk management"),
        ("stress capital buffer", "Capital stress testing"),
        ("comprehensive capital analysis", "Capital stress testing"),
        ("basel iii", "Regulatory capital requirements"),
        ("total loss-absorbing capacity", "Loss-absorbing capital requirements"),
        ("preferred stock", "Preferred stock capital actions"),
        ("common stock dividends", "Common stock capital actions"),
        ("j.p. morgan se", "Regulatory capital for subsidiaries"),
        ("j.p. morgan securities plc", "Regulatory capital for subsidiaries"),
        ("jpmorgan chase bank, n.a.", "Regulatory capital for bank subsidiary"),
        ("liquidity risk", "Liquidity risk management"),
        ("liquidity management", "Liquidity risk management"),
        ("contingency funding", "Contingency funding plan"),
        ("long-term funding", "Long-term funding"),
        ("credit risk", "Credit risk management"),
        ("consumer credit", "Consumer credit risk"),
        ("wholesale credit", "Wholesale credit risk"),
        ("allowance for credit losses", "Allowance for credit losses"),
        ("market risk", "Market risk management"),
        ("value-at-risk", "Value-at-risk"),
        ("operational risk", "Operational risk management"),
        ("model risk", "Model risk management"),
        ("reputation risk", "Reputation risk management"),
        ("climate risk", "Climate risk management"),
        ("critical accounting", "Critical accounting estimates"),
    ]
    for needle, topic in topic_rules:
        if needle in haystack:
            return topic

    if section_label == "Item 1A Risk Factors":
        risk_rules = [
            ("cyber", "Cybersecurity risk"),
            ("data", "Data and technology risk"),
            ("market", "Market risk"),
            ("credit", "Credit risk"),
            ("liquidity", "Liquidity risk"),
            ("capital", "Capital risk"),
            ("regulat", "Legal and regulatory risk"),
            ("government", "Government policy risk"),
            ("political", "Political and geopolitical risk"),
            ("competition", "Competitive risk"),
            ("employee", "People risk"),
            ("country", "Country risk"),
            ("reputation", "Reputation risk"),
            ("conduct", "Conduct risk"),
            ("strateg", "Strategic risk"),
            ("operational", "Operational risk"),
        ]
        for needle, topic in risk_rules:
            if needle in haystack:
                return topic
        return "Risk factor"

    generic_topics = {
        "business": "Business overview",
        "risk factors": "Risk factors",
        "management's discussion and analysis": "Management discussion and analysis",
        "item 7 md&a": "Management discussion and analysis",
        "partially offset by": "Financial performance drivers",
        "largely offset by": "Financial performance drivers",
    }
    lowered = raw_title.lower()
    if lowered in generic_topics:
        return generic_topics[lowered]

    short = re.split(r",|\breflecting\b|\bdriven by\b|\bincluding\b|\bwas\b|\bare\b", raw_title)[0]
    short = re.sub(r"\s+", " ", short).strip(" .:-")
    words = short.split()
    if len(words) > 7:
        short = " ".join(words[:7])
    return short or section_label


def is_low_value_topic(topic: str, text: str) -> bool:
    lower_topic = topic.lower()
    lower_text = text.lower()
    if lower_topic in {"management discussion and analysis", "financial performance drivers"}:
        return True
    if lower_text.startswith("comparisons noted in the sections below"):
        return True
    if lower_text.startswith("selected business metrics"):
        return True
    if "the following is a discussion" in lower_text:
        return True
    return False


def build_section_chunks(
    blocks: list[HtmlBlock],
    section_label: str,
    section_key: str,
    source: str,
    max_chars: int = MAX_TEXT_CHARS,
    min_chars: int = 450,
) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    active_title = section_label
    active_paragraphs: list[str] = []
    active_table_cells: list[str] = []

    def add_chunk(
        title: str,
        paragraphs: list[str],
        chunk_type: str = "text",
        topic_override: str | None = None,
    ) -> None:
        cleaned = [paragraph for paragraph in paragraphs if paragraph]
        if not cleaned:
            return
        body = "\n".join(cleaned)
        topic = topic_override or semantic_topic(title, section_label, body)
        text = body
        if topic and not body.lower().startswith(topic.lower()):
            text = f"{topic}: {body}"
        if chunk_type == "text" and count_words(text) < 20:
            return
        chunk_index = len(chunks) + 1
        chunks.append(
            {
                "id": f"jpm_2025_10k_{section_key}_{chunk_index:04d}",
                "company": COMPANY,
                "doc_type": DOC_TYPE,
                "filing_date": FILING_DATE,
                "section": section_label,
                "topic": topic,
                "chunk_type": chunk_type,
                "title": title.rstrip("."),
                "text": text,
                "source": source,
            }
        )

    def flush_table() -> None:
        nonlocal active_table_cells
        if len(active_table_cells) < 8:
            active_table_cells = []
            return
        cells = compact_table_cells(active_table_cells)
        active_table_cells = []
        if len(cells) < 8:
            return
        topic = semantic_topic(active_title, section_label, " ".join(cells))
        selected_cells: list[str] = []
        for cell in cells[:MAX_TABLE_CELLS]:
            candidate = "; ".join([*selected_cells, cell])
            if len(candidate) > MAX_FINAL_TEXT_CHARS - 120:
                break
            selected_cells.append(cell)
        table_text = f"Table summary for {topic}: " + "; ".join(selected_cells)
        if not table_text.endswith("."):
            table_text += "."
        add_chunk(
            active_title,
            [table_text],
            chunk_type="table_summary",
            topic_override=topic,
        )

    def flush() -> None:
        nonlocal active_paragraphs
        flush_table()
        paragraphs = [paragraph for paragraph in active_paragraphs if paragraph]
        if not paragraphs:
            return
        if active_title == "Overview" and len(paragraphs) > 1:
            add_chunk("Overview", [paragraphs[0]], topic_override="Firm business overview")
            add_chunk("Subsidiaries and structure", paragraphs[1:], topic_override="Subsidiaries and operating structure")
        else:
            add_chunk(active_title, paragraphs)
        active_paragraphs = []

    for block in blocks:
        if block.in_table:
            if not should_skip_block(HtmlBlock(text=block.text, in_table=False)):
                active_table_cells.append(clean_paragraph(block.text))
            continue
        flush_table()
        if should_skip_block(block):
            continue
        text = clean_paragraph(block.text)
        if not text or should_skip_block(HtmlBlock(text=text, in_table=False)):
            continue
        if is_heading(text, section_label):
            flush()
            active_title = text.rstrip()
            if section_label == "Item 1A Risk Factors" and text.endswith("."):
                active_paragraphs.append(text)
            continue
        if not active_paragraphs:
            active_paragraphs.append(text)
            continue
        projected = len("\n".join(active_paragraphs)) + len(text) + 1
        if projected > max_chars and len("\n".join(active_paragraphs)) >= min_chars:
            flush()
        active_paragraphs.append(text)
    flush()
    return chunks


def compact_table_cells(cells: list[str]) -> list[str]:
    compacted: list[str] = []
    previous = ""
    for raw_cell in cells:
        cell = normalize_spaces(raw_cell)
        if not cell or cell == previous:
            continue
        if should_skip_block(HtmlBlock(text=cell, in_table=False)):
            continue
        if cell in {"$", "%", "-", "-"}:
            continue
        if len(cell) > 180:
            cell = cell[:177].rstrip() + "..."
        compacted.append(cell)
        previous = cell
    return compacted


def postprocess_chunks(chunks: list[dict[str, str]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for chunk in chunks:
        current = chunk.copy()
        if current["chunk_type"] == "table_summary":
            cleaned.append(current)
            continue

        current["chunk_type"] = "text"
        current["text"] = clean_paragraph(current["text"])
        current["topic"] = semantic_topic(current["title"], current["section"], current["text"])
        if is_low_value_topic(current["topic"], current["text"]):
            continue
        if count_words(current["text"]) < MIN_TEXT_WORDS and len(current["text"]) < TARGET_TEXT_MIN_CHARS:
            continue
        cleaned.extend(split_large_text_chunk(current))

    merged = merge_tiny_same_topic_chunks(cleaned)
    filtered = [
        chunk
        for chunk in merged
        if chunk["chunk_type"] == "table_summary"
        or len(chunk["text"]) >= TINY_TEXT_CHARS
        or count_words(chunk["text"]) >= MIN_TEXT_WORDS
    ]
    return reindex_chunks(filtered)


def split_large_text_chunk(chunk: dict[str, str]) -> list[dict[str, str]]:
    if len(chunk["text"]) <= HARD_TEXT_MAX_CHARS:
        return [chunk]

    units = semantic_text_units(chunk["text"])
    split_chunks: list[dict[str, str]] = []
    active: list[str] = []

    def flush_active() -> None:
        nonlocal active
        if not active:
            return
        text = "\n".join(active).strip()
        if text:
            new_chunk = chunk.copy()
            new_chunk["text"] = ensure_topic_prefix(text, chunk["topic"])
            split_chunks.append(new_chunk)
        active = []

    for unit in units:
        if len(unit) > HARD_TEXT_MAX_CHARS:
            flush_active()
            for sentence_group in sentence_groups(unit, chunk["topic"]):
                new_chunk = chunk.copy()
                new_chunk["text"] = sentence_group
                split_chunks.append(new_chunk)
            continue

        projected = len("\n".join(active)) + len(unit) + (1 if active else 0)
        if active and projected > TARGET_TEXT_MAX_CHARS and len("\n".join(active)) >= TARGET_TEXT_MIN_CHARS:
            flush_active()
        active.append(strip_duplicate_topic_prefix(unit, chunk["topic"]))
    flush_active()
    return split_chunks


def semantic_text_units(text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n") if paragraph.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    return sentence_units(text)


def sentence_units(text: str) -> list[str]:
    normalized = normalize_spaces(text)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", normalized)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def sentence_groups(text: str, topic: str) -> list[str]:
    groups: list[str] = []
    active: list[str] = []
    for sentence in sentence_units(text):
        sentence = strip_duplicate_topic_prefix(sentence, topic)
        projected = len(" ".join(active)) + len(sentence) + (1 if active else 0)
        if active and projected > TARGET_TEXT_MAX_CHARS:
            group_text = ensure_topic_prefix(" ".join(active), topic)
            groups.append(group_text)
            active = []
        active.append(sentence)
    if active:
        groups.append(ensure_topic_prefix(" ".join(active), topic))
    return groups


def merge_tiny_same_topic_chunks(chunks: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    index = 0
    while index < len(chunks):
        current = chunks[index].copy()
        if current["chunk_type"] != "text" or len(current["text"]) >= TARGET_TEXT_MIN_CHARS:
            merged.append(current)
            index += 1
            continue

        if (
            index + 1 < len(chunks)
            and chunks[index + 1]["chunk_type"] == "text"
            and chunks[index + 1]["section"] == current["section"]
            and chunks[index + 1]["topic"] == current["topic"]
            and len(current["text"]) + len(chunks[index + 1]["text"]) <= TARGET_TEXT_MAX_CHARS
        ):
            nxt = chunks[index + 1].copy()
            nxt["text"] = combine_same_topic_text(current["text"], nxt["text"], current["topic"])
            chunks[index + 1] = nxt
            index += 1
            continue

        if (
            merged
            and merged[-1]["chunk_type"] == "text"
            and merged[-1]["section"] == current["section"]
            and merged[-1]["topic"] == current["topic"]
            and len(merged[-1]["text"]) + len(current["text"]) <= TARGET_TEXT_MAX_CHARS
        ):
            merged[-1]["text"] = combine_same_topic_text(merged[-1]["text"], current["text"], current["topic"])
            index += 1
            continue

        index += 1
    return merged


def combine_same_topic_text(left: str, right: str, topic: str) -> str:
    right = strip_duplicate_topic_prefix(right, topic)
    return ensure_topic_prefix(f"{left}\n{right}", topic)


def ensure_topic_prefix(text: str, topic: str) -> str:
    clean = normalize_spaces(text) if "\n" not in text else "\n".join(normalize_spaces(part) for part in text.split("\n"))
    if clean.lower().startswith(topic.lower()):
        return clean
    return f"{topic}: {clean}"


def strip_duplicate_topic_prefix(text: str, topic: str) -> str:
    pattern = rf"^{re.escape(topic)}\s*:\s*"
    return re.sub(pattern, "", text.strip(), flags=re.I)


def reindex_chunks(chunks: list[dict[str, str]]) -> list[dict[str, str]]:
    counters: dict[str, int] = {}
    key_by_section = {
        "Item 1 Business": "item1",
        "Item 1A Risk Factors": "item1a",
        "Item 7 MD&A": "item7",
    }
    for index, chunk in enumerate(chunks):
        section_key = key_by_section[chunk["section"]]
        counters[section_key] = counters.get(section_key, 0) + 1
        chunk_id = f"jpm_2025_10k_{section_key}_{counters[section_key]:04d}"
        chunk_type = "table_summary" if chunk["chunk_type"] == "table_summary" else "text"
        topic = chunk.pop("topic")
        primary_topic, secondary_topic = taxonomy_topics(
            semantic_topic=topic,
            title=chunk["title"],
            text=chunk["text"],
            section=chunk["section"],
        )
        output_chunk = {
            "id": chunk_id,
            "company": chunk["company"],
            "doc_type": chunk["doc_type"],
            "filing_date": chunk["filing_date"],
            "section": chunk["section"],
            "primary_topic": primary_topic,
            "secondary_topic": secondary_topic,
            "chunk_type": chunk_type,
            "quality": quality_label({**chunk, "chunk_type": chunk_type}),
            "title": concise_title(topic, chunk["title"]),
            "text": chunk["text"],
            "source": chunk["source"],
        }
        chunks[index] = output_chunk
    return chunks


def taxonomy_topics(
    semantic_topic: str,
    title: str,
    text: str,
    section: str,
) -> tuple[str, str]:
    semantic_lower = semantic_topic.lower()
    haystack = f"{semantic_topic} {title} {text}".lower()

    if section == "Item 1 Business":
        if "competitive" in semantic_lower or "competition" in semantic_lower:
            return "competition", "competitive_environment"
        if "business segment" in semantic_lower:
            return "business_segment", "segment_overview"
        if "subsidiar" in semantic_lower or "structure" in semantic_lower:
            if "international" in haystack or "outside the u.s." in haystack or "jpmse" in haystack:
                return "business_structure", "international_structure"
            if "bank subsidiary" in haystack or "national bank" in haystack:
                return "business_structure", "bank_subsidiaries"
            return "business_structure", "major_subsidiaries"
        if any(token in haystack for token in ["overview", "leading financial services", "operations worldwide"]):
            return "company_overview", "company_profile"
        if any(token in haystack for token in ["subsidiar", "operating structure", "bank, n.a.", "jpmse"]):
            if "international" in haystack or "outside the u.s." in haystack or "jpmse" in haystack:
                return "business_structure", "international_structure"
            if "bank subsidiary" in haystack or "national bank" in haystack:
                return "business_structure", "bank_subsidiaries"
            return "business_structure", "major_subsidiaries"
        if "business segment" in haystack or "consumer & community banking" in haystack:
            return "business_segment", "segment_overview"
        if "competition" in haystack or "competitive" in haystack:
            return "competition", "competitive_environment"
        if any(token in haystack for token in ["regulation", "supervision", "compliance", "capital requirement"]):
            return "regulatory_risk", regulatory_secondary_topic(haystack)
        return "company_overview", "company_profile"

    if section == "Item 1A Risk Factors":
        if any(token in haystack for token in ["cyber", "technology", "data", "operational", "model"]):
            return "operational_risk", operational_secondary_topic(haystack)
        if any(token in haystack for token in ["credit", "market", "liquidity", "funding", "capital", "collateral", "margin"]):
            return "financial_risk", financial_secondary_topic(haystack)
        if any(token in haystack for token in ["regulat", "legal", "litigation", "government", "jurisdiction", "compliance"]):
            return "regulatory_risk", regulatory_secondary_topic(haystack)
        if any(token in haystack for token in ["competition", "strategy", "strategic", "country", "geopolitical", "people", "reputation", "conduct"]):
            return "operational_risk", operational_secondary_topic(haystack)
        return "operational_risk", "operational_risk"

    if any(token in haystack for token in ["revenue", "expense", "income", "tax", "balance sheet", "cash flow", "receivable", "segment performance", "markets revenue"]):
        return "performance_analysis", performance_secondary_topic(haystack)
    if any(token in haystack for token in ["credit", "market", "liquidity", "funding", "var", "collateral", "margin", "allowance"]):
        return "financial_risk", financial_secondary_topic(haystack)
    if any(token in haystack for token in ["capital", "basel", "tlac", "stress capital", "mrel", "regulatory requirement"]):
        return "regulatory_risk", "capital_regulation"
    if any(token in haystack for token in ["cyber", "technology", "operational", "model", "governance", "control"]):
        return "operational_risk", operational_secondary_topic(haystack)
    if any(token in haystack for token in ["consumer banking", "investment banking", "asset and wealth", "business segment"]):
        return "business_segment", business_segment_secondary_topic(haystack)
    return "performance_analysis", "segment_performance"


def regulatory_secondary_topic(haystack: str) -> str:
    if "capital" in haystack or "basel" in haystack or "tlac" in haystack or "mrel" in haystack:
        return "capital_regulation"
    if "litigation" in haystack or "enforcement" in haystack or "penalt" in haystack:
        return "litigation_and_enforcement"
    if "consumer" in haystack or "cfpb" in haystack:
        return "consumer_finance_regulation"
    if "interchange" in haystack:
        return "interchange_fee_regulation"
    if "jurisdiction" in haystack or "country" in haystack or "u.k." in haystack or "eu" in haystack:
        return "jurisdictional_regulation"
    if "legal" in haystack:
        return "legal_risk"
    return "supervision_and_compliance"


def financial_secondary_topic(haystack: str) -> str:
    if "liquidity" in haystack:
        return "liquidity_risk"
    if "funding" in haystack or "deposit" in haystack:
        return "funding_risk"
    if "market" in haystack or "var" in haystack or "value-at-risk" in haystack:
        return "market_risk"
    if "commitment" in haystack:
        return "lending_commitments"
    if "collateral" in haystack or "margin" in haystack:
        return "collateral_and_margin"
    return "credit_risk"


def operational_secondary_topic(haystack: str) -> str:
    if "cyber" in haystack:
        return "cyber_risk"
    if "technology" in haystack or "data" in haystack:
        return "technology_risk"
    if "model governance" in haystack or "governance" in haystack:
        return "model_governance"
    if "model" in haystack:
        return "model_risk"
    return "technology_risk"


def performance_secondary_topic(haystack: str) -> str:
    if "expense" in haystack or "compensation" in haystack:
        return "expense_drivers"
    if "balance" in haystack or "cash flow" in haystack or "assets" in haystack or "liabilities" in haystack:
        return "balance_sheet_trends"
    if "receivable" in haystack:
        return "receivables"
    if "segment" in haystack or "consumer banking" in haystack or "investment banking" in haystack:
        return "segment_performance"
    return "revenue_drivers"


def business_segment_secondary_topic(haystack: str) -> str:
    if "consumer" in haystack or "ccb" in haystack:
        return "consumer_banking"
    if "investment" in haystack or "cib" in haystack or "markets" in haystack:
        return "investment_banking"
    if "asset" in haystack or "wealth" in haystack or "awm" in haystack:
        return "asset_wealth_management"
    return "segment_overview"


def quality_label(chunk: dict[str, str]) -> str:
    text = chunk["text"]
    length = len(text)
    if chunk["chunk_type"] == "table_summary":
        return "high" if 300 <= length <= 1200 else "medium"
    if 300 <= length <= 1200 and count_words(text) >= MIN_TEXT_WORDS:
        return "high"
    if TINY_TEXT_CHARS <= length <= HARD_TEXT_MAX_CHARS:
        return "medium"
    return "low"


def concise_title(topic: str, original_title: str) -> str:
    title = topic.strip() or original_title.strip()
    title = re.sub(r"\s+", " ", title).strip(" .:-")
    if len(title) <= 80:
        return title
    return title[:77].rstrip() + "..."


def build_chunks(input_path: Path) -> list[dict[str, str]]:
    blocks = parse_visible_blocks(input_path)

    item1_start, item1_end = find_section_bounds(
        blocks,
        SectionSpec(
            key="item1",
            label="Item 1 Business",
            start_pattern=re.compile(r"Item 1\. Business\."),
            end_pattern=re.compile(r"Item 1A\. Risk Factors\."),
        ),
    )
    item1a_start, item1a_end = find_section_bounds(
        blocks,
        SectionSpec(
            key="item1a",
            label="Item 1A Risk Factors",
            start_pattern=re.compile(r"Item 1A\. Risk Factors\."),
            end_pattern=re.compile(r"Item 1B\. Unresolved Staff Comments\."),
            start_after=item1_end,
        ),
    )
    mda_start, mda_end = find_mda_bounds(blocks)

    source = str(input_path)
    all_chunks: list[dict[str, str]] = []
    all_chunks.extend(
        build_section_chunks(blocks[item1_start:item1_end], "Item 1 Business", "item1", source)
    )
    all_chunks.extend(
        build_section_chunks(blocks[item1a_start:item1a_end], "Item 1A Risk Factors", "item1a", source)
    )
    all_chunks.extend(
        build_section_chunks(blocks[mda_start:mda_end], "Item 7 MD&A", "item7", source)
    )
    return postprocess_chunks(all_chunks)


def write_jsonl(records: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare JPMorganChase 2025 10-K RAG chunks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    chunks = build_chunks(args.input)
    write_jsonl(chunks, args.output)
    by_section: dict[str, int] = {}
    for chunk in chunks:
        by_section[chunk["section"]] = by_section.get(chunk["section"], 0) + 1
    print(f"Wrote {len(chunks)} chunks to {args.output}")
    for section, count in by_section.items():
        print(f"- {section}: {count}")


if __name__ == "__main__":
    main()
