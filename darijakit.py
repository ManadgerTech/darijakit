"""DarijaKit - a developer toolkit for Moroccan Darija on local LLMs.

Zero-dependency core: Arabizi transliteration, Darija/Arabic text
normalization, script detection, and a benchmark runner that works against
any OpenAI-compatible endpoint (Ollama, llama.cpp, vLLM, LM Studio).

Everything runs locally. Nothing leaves your machine except what you
explicitly send to the endpoint you choose.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata  # noqa: F401  (reserved for future normalization rules)
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------

_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN_RE = re.compile(r"[A-Za-z]")
# Digits that carry letter meaning in Arabizi: 2=hamza, 3=ain, 5=kha,
# 7=hha, 8=gha, 9=qaf.
_ARABIZI_DIGIT_RE = re.compile(r"[235789]")


def detect_script(text: str) -> str:
    """Return 'arabic', 'arabizi', 'mixed' or 'other' for *text*.

    'arabizi' means Latin script that carries Arabizi digits or known
    Arabizi words; plain French/English Latin text is reported as 'other'
    so it is not sent through the transliterator by mistake.
    """
    has_arabic = bool(_ARABIC_RE.search(text))
    has_latin = bool(_LATIN_RE.search(text))
    has_arabizi_digit = bool(_ARABIZI_DIGIT_RE.search(text))
    if has_arabic and (has_latin or has_arabizi_digit):
        return "mixed"
    if has_arabic:
        return "arabic"
    if has_arabizi_digit:
        return "arabizi"
    if has_latin:
        padded = " " + text.lower() + " "
        for word in _ARABIZI_WORD_MAP:
            if (f" {word} " in padded or f" {word}." in padded
                    or f" {word}," in padded or f" {word}?" in padded):
                return "arabizi"
        return "other"
    return "other"


# ---------------------------------------------------------------------------
# Arabizi -> Arabic transliteration
# ---------------------------------------------------------------------------

# Word-level map for high-frequency Darija words. Word lookup happens first;
# anything unknown falls back to character-level transliteration.
_ARABIZI_WORD_MAP: Dict[str, str] = {
    "salam": "سلام", "salamou": "سلام", "alikoum": "عليكم", "alaykoum": "عليكم",
    "labas": "لاباس", "hamdoulah": "الحمد لله", "hamdullah": "الحمد لله",
    "chno": "شنو", "chnou": "شنو", "ash": "أش", "ach": "أش",
    "wach": "واش", "wash": "واش", "fin": "فين", "fine": "فين",
    "chhal": "شحال", "ch7al": "شحال", "7al": "حال",
    "daba": "دابا", "drari": "دراري",
    "bghit": "بغيت", "bghina": "بغينا", "bgha": "بغا", "bghaw": "بغاو",
    "bezzaf": "بزاف", "bezaf": "بزاف", "bzaf": "بزاف",
    "wakha": "واخا", "wak7a": "واخا",
    "mezyan": "مزيان", "mzyan": "مزيان",
    "khoya": "خويا", "khti": "ختي",
    "hadi": "هادي", "hadak": "هاداك", "hadchi": "هادشي", "hadakchi": "هاداكشي",
    "kifach": "كيفاش", "kidayr": "كيداير", "kidayra": "كيدايرة",
    "imta": "إمتى", "3lach": "علاش", "alach": "علاش",
    "machi": "ماشي", "ma3lich": "ماعليش",
    "safi": "صافي", "baraka": "بركة", "choukran": "شكرا", "chokran": "شكرا",
    "3afak": "عافاك", "afak": "عافاك",
    "inchallah": "إن شاء الله",
    "nhar": "نهار", "lila": "ليلة",
    "zwin": "زوين", "zwina": "زوينة", "7aja": "حاجة", "haja": "حاجة",
    "flous": "فلوس", "derham": "دراهم", "drahem": "دراهم",
    "sir": "سير", "siri": "سيري",
    "ghad": "غاد", "ghadi": "غادي", "9ahwa": "قهوة", "atay": "أتاي",
    "a3ziz": "عزيز", "aziz": "عزيز",
    "bnt": "بنت", "wld": "ولد", "rajel": "راجل", "mra": "مرا",
    "lmaghrib": "المغرب", "maghrib": "مغرب",
    "men": "من", "m3a": "مع", "m3aya": "معايا",
    "kant": "كنت", "kont": "كنت",
    "sa3a": "ساعة", "khasni": "خاصني", "khassni": "خاصني",
    "nkhdem": "نخدم", "nakol": "ناكل", "tajine": "طاجين",
}

# Digraphs first (longest match wins), then single chars.
_ARABIZI_DIGRAPHS: Dict[str, str] = {
    "ch": "ش", "sh": "ش", "kh": "خ", "gh": "غ", "th": "ث", "dh": "ذ",
    "dj": "ج", "ou": "و", "oo": "و", "aa": "ا", "ee": "ي", "ii": "ي",
    "ph": "ف", "ss": "س", "tt": "ت", "dd": "د", "rr": "ر", "ll": "ل",
}

_ARABIZI_CHARS: Dict[str, str] = {
    "a": "ا", "b": "ب", "c": "ك", "d": "د", "e": "ي", "f": "ف", "g": "ڭ",
    "h": "ه", "i": "ي", "j": "ج", "k": "ك", "l": "ل", "m": "م", "n": "ن",
    "o": "و", "p": "ب", "q": "ك", "r": "ر", "s": "س", "t": "ت", "u": "و",
    "v": "ڥ", "w": "و", "x": "كس", "y": "ي", "z": "ز",
    "2": "أ", "3": "ع", "5": "خ", "7": "ح", "8": "غ", "9": "ق",
    "'": "ء",
}


def _transliterate_word(word: str) -> str:
    """Character-level Arabizi -> Arabic for a single lowercase word."""
    out: List[str] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i : i + 2] in _ARABIZI_DIGRAPHS:
            out.append(_ARABIZI_DIGRAPHS[word[i : i + 2]])
            i += 2
        elif word[i] in _ARABIZI_CHARS:
            out.append(_ARABIZI_CHARS[word[i]])
            i += 1
        else:
            out.append(word[i])
            i += 1
    return "".join(out)


def arabizi_to_arabic(text: str) -> str:
    """Transliterate Arabizi (Latin-script Darija) to Arabic script.

    High-frequency Darija words use a curated word map; the rest falls back
    to character-level rules.
    """

    def convert(match: "re.Match[str]") -> str:
        token = match.group(0)
        lowered = token.lower()
        if lowered in _ARABIZI_WORD_MAP:
            return _ARABIZI_WORD_MAP[lowered]
        if not _LATIN_RE.search(token):
            return token
        return _transliterate_word(lowered)

    return re.sub(r"[A-Za-z0-9']+", convert, text)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_ARABIC_DIACRITICS = (
    "ًٌٍَُِّْٰٖٗـ"  # tanwin, harakat, shadda, sukun, maddah, tatweel
)

_ALEF_VARIANTS = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"}
_YA_VARIANTS = {"ى": "ي", "ئ": "ي"}
_TAA_MARBUTA = {"ة": "ه"}

# Common Darija spelling variants -> canonical form. Conservative on purpose:
# only spellings that never change meaning.
DARIJA_VARIANTS: Dict[str, str] = {
    "بش": "باش",
    "فاين": "فين",
    "اشنو": "شنو", "أشنو": "شنو", "شنوو": "شنو",
    "شال": "شحال",
    "شكن": "شكون",
    "كيفش": "كيفاش", "كيفاس": "كيفاش",
    "داب": "دابا",
    "بززاف": "بزاف",
    "بغيث": "بغيت",
    "مزين": "مزيان",
    "واخخا": "واخا",
    "شكراً": "شكرا",
    "عافك": "عافاك",
    "غاد": "غادي",
    "معنديش": "ماعنديش", "ما عنديش": "ماعنديش",
    "مابغيثش": "مابغيتش", "ما بغيتش": "مابغيتش",
}


def strip_diacritics(text: str) -> str:
    """Remove Arabic harakat, tanwin, shadda, sukun and tatweel."""
    for mark in _ARABIC_DIACRITICS:
        text = text.replace(mark, "")
    return text


def normalize_arabic(
    text: str,
    unify_alef: bool = True,
    unify_ya: bool = True,
    taa_marbuta_to_ha: bool = False,
    strip_marks: bool = True,
) -> str:
    """Normalize Arabic-script text for comparison or model input.

    By default: strips diacritics and tatweel, unifies alef variants to bare
    alef and alef maksura/hamza-on-ya to ya. Taa marbuta is kept unless
    ``taa_marbuta_to_ha`` is set (useful for strict answer matching).
    """
    if strip_marks:
        text = strip_diacritics(text)
    if unify_alef:
        for src, dst in _ALEF_VARIANTS.items():
            text = text.replace(src, dst)
    if unify_ya:
        for src, dst in _YA_VARIANTS.items():
            text = text.replace(src, dst)
    if taa_marbuta_to_ha:
        for src, dst in _TAA_MARBUTA.items():
            text = text.replace(src, dst)
    text = re.sub(r"[‎‏‪-‮]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_darija(text: str) -> str:
    """Full pipeline: Arabic normalization + canonical Darija spellings."""
    text = normalize_arabic(text)
    words = text.split(" ")
    words = [DARIJA_VARIANTS.get(w, w) for w in words]
    return " ".join(words)


def normalize(text: str) -> str:
    """Smart normalize: detect script, transliterate Arabizi, normalize."""
    script = detect_script(text)
    if script == "arabizi":
        text = arabizi_to_arabic(text)
    return normalize_darija(text)


# ---------------------------------------------------------------------------
# Benchmarking against any OpenAI-compatible endpoint
# ---------------------------------------------------------------------------


@dataclass
class EvalItem:
    """One benchmark item. ``kind`` decides scoring:

    - ``mcq``: prompt presents options; the reply must contain the expected
      option letter (Arabic أ/ب/ج/د or Latin A-D).
    - ``translation`` / ``contains``: normalized reply must contain any of
      the expected strings.
    - ``exact``: normalized reply must equal one expected string.
    """

    id: str
    kind: str  # mcq | translation | contains | exact
    prompt: str
    expected: List[str]
    category: str = "general"
    system: Optional[str] = None


@dataclass
class EvalResult:
    item: EvalItem
    response: str
    passed: bool
    score_detail: str = ""


@dataclass
class EvalReport:
    model: str
    endpoint: str
    results: List[EvalResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    def by_category(self) -> Dict[str, Tuple[int, int]]:
        out: Dict[str, List[int]] = {}
        for r in self.results:
            bucket = out.setdefault(r.item.category, [0, 0])
            bucket[1] += 1
            if r.passed:
                bucket[0] += 1
        return {k: (v[0], v[1]) for k, v in out.items()}


def load_evalset(path: str) -> List[EvalItem]:
    """Load a JSONL eval set. Each line: id, kind, prompt, expected, ..."""
    items: List[EvalItem] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            try:
                items.append(
                    EvalItem(
                        id=raw["id"],
                        kind=raw["kind"],
                        prompt=raw["prompt"],
                        expected=list(raw["expected"]),
                        category=raw.get("category", "general"),
                        system=raw.get("system"),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{lineno}: missing field {exc}") from exc
    return items


def _norm_for_match(text: str) -> str:
    return normalize_darija(text).lower()


def score_item(item: EvalItem, response: str) -> Tuple[bool, str]:
    """Score one response. Returns (passed, detail)."""
    resp = _norm_for_match(response)
    if item.kind == "mcq":
        for letter in [_norm_for_match(e) for e in item.expected]:
            if re.search(
                rf"(?:^|[\s\(\[.,؛:\-]){re.escape(letter)}(?:$|[\s\)\].,؛:\-])",
                resp,
            ):
                return True, f"found option '{letter}'"
        return False, f"none of {item.expected} found"
    if item.kind == "exact":
        expected = [_norm_for_match(e) for e in item.expected]
        ok = resp in expected
        return ok, "exact match" if ok else f"got: {resp[:80]}"
    # translation / contains
    for exp in item.expected:
        if _norm_for_match(exp) in resp:
            return True, f"contains '{exp}'"
    return False, f"none of {item.expected} found"


def default_client(
    endpoint: str, model: str, timeout: float = 120.0
) -> Callable[[str, Optional[str]], str]:
    """OpenAI-compatible chat client over stdlib urllib.

    Works with Ollama (http://localhost:11434/v1), llama.cpp server,
    LM Studio and vLLM out of the box.
    """

    def call(prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = json.dumps(
            {"model": model, "messages": messages, "temperature": 0.0}
        ).encode("utf-8")
        req = urllib.request.Request(
            endpoint.rstrip("/") + "/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"] or ""

    return call


def run_eval(
    items: Sequence[EvalItem],
    model: str,
    endpoint: str = "http://localhost:11434/v1",
    client: Optional[Callable[[str, Optional[str]], str]] = None,
    timeout: float = 120.0,
    progress: bool = False,
) -> EvalReport:
    """Run *items* against *model* and return a report.

    ``client`` lets tests and integrations inject any
    ``(prompt, system) -> response`` callable; by default an
    OpenAI-compatible HTTP client is used.
    """
    if client is None:
        client = default_client(endpoint, model, timeout=timeout)
    report = EvalReport(model=model, endpoint=endpoint)
    for idx, item in enumerate(items, 1):
        if progress:
            print(f"[{idx}/{len(items)}] {item.id} ...", file=sys.stderr)
        try:
            response = client(item.prompt, item.system)
        except Exception as exc:  # network errors should not kill a run
            report.results.append(
                EvalResult(item=item, response="", passed=False,
                           score_detail=f"error: {exc}")
            )
            continue
        passed, detail = score_item(item, response)
        report.results.append(
            EvalResult(item=item, response=response, passed=passed,
                       score_detail=detail)
        )
    return report


def report_to_markdown(report: EvalReport) -> str:
    lines = [
        f"# DarijaKit benchmark: `{report.model}`",
        "",
        f"Endpoint: `{report.endpoint}`",
        "",
        f"**Score: {report.passed}/{report.total} "
        f"({100.0 * report.passed / max(report.total, 1):.0f}%)**",
        "",
        "| Category | Passed | Total |",
        "| --- | --- | --- |",
    ]
    for cat, (ok, total) in sorted(report.by_category().items()):
        lines.append(f"| {cat} | {ok} | {total} |")
    lines += ["", "| # | Item | Result | Detail |",
              "| --- | --- | --- | --- |"]
    for i, r in enumerate(report.results, 1):
        mark = "PASS" if r.passed else "FAIL"
        lines.append(f"| {i} | {r.item.id} | {mark} | {r.score_detail} |")
    return "\n".join(lines) + "\n"


def report_to_json(report: EvalReport) -> str:
    return json.dumps(
        {
            "model": report.model,
            "endpoint": report.endpoint,
            "passed": report.passed,
            "total": report.total,
            "by_category": {
                k: {"passed": v[0], "total": v[1]}
                for k, v in report.by_category().items()
            },
            "results": [
                {
                    "id": r.item.id,
                    "category": r.item.category,
                    "kind": r.item.kind,
                    "passed": r.passed,
                    "detail": r.score_detail,
                    "response": r.response,
                }
                for r in report.results
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_stdin_or(value: Optional[str]) -> str:
    if value is not None:
        return value
    return sys.stdin.read()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="darijakit",
        description="Tools for Moroccan Darija on local LLMs.",
    )
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_norm = sub.add_parser("normalize", help="Normalize Darija/Arabic text")
    p_norm.add_argument("text", nargs="?", help="Text (default: stdin)")

    p_ar = sub.add_parser("arabizi",
                          help="Transliterate Arabizi to Arabic script")
    p_ar.add_argument("text", nargs="?", help="Text (default: stdin)")

    p_det = sub.add_parser(
        "detect", help="Detect script: arabic | arabizi | mixed | other")
    p_det.add_argument("text", nargs="?", help="Text (default: stdin)")

    p_eval = sub.add_parser(
        "eval", help="Benchmark a local model on the Darija eval set")
    p_eval.add_argument("--model", required=True,
                        help="Model name, e.g. qwen2.5:7b")
    p_eval.add_argument("--endpoint", default="http://localhost:11434/v1",
                        help="OpenAI-compatible base URL (default: Ollama)")
    p_eval.add_argument("--evalset", default="evalset.jsonl",
                        help="Path to JSONL eval set")
    p_eval.add_argument("--timeout", type=float, default=120.0)
    p_eval.add_argument("--json", dest="as_json", action="store_true",
                        help="Emit JSON instead of Markdown")
    p_eval.add_argument("--quiet", action="store_true",
                        help="No progress output")

    args = parser.parse_args(argv)

    if args.command == "normalize":
        print(normalize(_read_stdin_or(args.text)))
        return 0
    if args.command == "arabizi":
        print(arabizi_to_arabic(_read_stdin_or(args.text)))
        return 0
    if args.command == "detect":
        print(detect_script(_read_stdin_or(args.text)))
        return 0
    if args.command == "eval":
        items = load_evalset(args.evalset)
        report = run_eval(
            items,
            model=args.model,
            endpoint=args.endpoint,
            timeout=args.timeout,
            progress=not args.quiet,
        )
        print(report_to_json(report) if args.as_json
              else report_to_markdown(report))
        return 0 if report.passed == report.total else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
