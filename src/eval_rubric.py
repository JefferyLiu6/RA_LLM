"""
Rubric-based quality scorer for LoRA experiment outputs.

Scores each model output on four dimensions (0–3 each, max 12):
  1. Summary depth        — length and information density
  2. Bullet depth         — average word count per key-point bullet
  3. Limitation specificity — how concrete and technical the limitation is
  4. Follow-up quality    — whether the question is specific and well-formed

Scores are computed on cached outputs (outputs/content_outputs.json) so
no model needs to be loaded. Results are written to outputs/rubric_scores.md.

Usage:
  python src/eval_rubric.py
"""

import json
import re
import sys
from pathlib import Path

ROOT          = Path(__file__).parent.parent
CONTENT_CACHE = ROOT / "outputs" / "content_outputs.json"
RESULTS_PATH  = ROOT / "outputs" / "rubric_scores.md"

_GENERIC_LIMITATION = {
    "none", "none identified", "not identified", "no limitation",
    "no limitations", "n/a", "none mentioned",
}
_FILLER = {
    "the","a","an","is","are","was","were","be","been","being",
    "it","its","this","that","these","those","and","or","but",
    "in","on","at","to","for","of","with","by","from","as",
    "can","may","also","more","such","each","their","which",
    "both","about","have","has","not","no","so","when","how",
}


# ---------------------------------------------------------------------------
# Section parsing (no external deps)
# ---------------------------------------------------------------------------

def parse_sections(text: str) -> dict:
    out = {}
    m = re.search(r"Summary\s*:\s*(.*?)(?=\nKey Points\s*:|\Z)", text, re.DOTALL | re.IGNORECASE)
    out["summary"] = m.group(1).strip() if m else ""
    m = re.search(r"Key Points\s*:(.*?)(?=\nLimitation\s*:|\Z)", text, re.DOTALL | re.IGNORECASE)
    out["key_points"] = re.findall(r"^\s*[-*]\s+(.+)", m.group(1), re.MULTILINE) if m else []
    m = re.search(r"Limitation\s*:\s*(.*?)(?=\nFollow-up Question\s*:|\Z)", text, re.DOTALL | re.IGNORECASE)
    out["limitation"] = m.group(1).strip() if m else ""
    m = re.search(r"Follow-up Question\s*:\s*(.*?)$", text, re.DOTALL | re.IGNORECASE)
    out["followup"] = m.group(1).strip() if m else ""
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wc(s: str) -> int:
    return len(s.split()) if s.strip() else 0


def _specificity(s: str) -> float:
    tokens = re.findall(r"\b[a-z]+\b", s.lower())
    if not tokens:
        return 0.0
    content = [t for t in tokens if t not in _FILLER]
    return len(content) / len(tokens)


# ---------------------------------------------------------------------------
# Dimension scorers (each returns 0–3)
# ---------------------------------------------------------------------------

def score_summary(sec: dict) -> int:
    """
    0 — missing or < 8 words
    1 — 8–14 words
    2 — ≥ 15 words
    3 — ≥ 15 words AND specificity ≥ 0.55
    """
    s  = sec.get("summary", "")
    wc = _wc(s)
    sp = _specificity(s)
    if wc < 8:
        return 0
    if wc < 15:
        return 1
    return 3 if sp >= 0.55 else 2


def score_bullets(sec: dict) -> int:
    """
    Average words per key-point bullet:
    0 — < 5   (too terse / missing)
    1 — 5–9
    2 — 10–14
    3 — ≥ 15  (detailed)
    """
    pts = sec.get("key_points", [])
    if not pts:
        return 0
    avg = sum(_wc(p) for p in pts) / len(pts)
    if avg < 5:
        return 0
    if avg < 10:
        return 1
    if avg < 15:
        return 2
    return 3


def score_limitation(sec: dict) -> int:
    """
    0 — missing, 'None', or boilerplate
    1 — present but < 8 words
    2 — 8–18 words
    3 — > 18 words with high specificity (≥ 0.50)
    """
    lim = sec.get("limitation", "").strip()
    if not lim or lim.lower().rstrip(".") in _GENERIC_LIMITATION:
        return 0
    wc = _wc(lim)
    sp = _specificity(lim)
    if wc < 8:
        return 1
    if wc <= 18:
        return 2
    return 3 if sp >= 0.50 else 2


def score_followup(sec: dict) -> int:
    """
    0 — missing
    1 — present but does not end with '?'
    2 — ends with '?' but < 8 words
    3 — ends with '?' and ≥ 8 words
    """
    fu = sec.get("followup", "").strip()
    if not fu:
        return 0
    if not fu.endswith("?"):
        return 1
    return 3 if _wc(fu) >= 8 else 2


def rubric_score(text: str) -> dict:
    sec   = parse_sections(text)
    s_sum = score_summary(sec)
    s_bul = score_bullets(sec)
    s_lim = score_limitation(sec)
    s_fu  = score_followup(sec)
    return {
        "summary":    s_sum,
        "bullets":    s_bul,
        "limitation": s_lim,
        "followup":   s_fu,
        "total":      s_sum + s_bul + s_lim + s_fu,
    }


def avg_rubric(scores: list[dict]) -> dict:
    if not scores:
        return {}
    keys = ["summary", "bullets", "limitation", "followup", "total"]
    return {k: round(sum(s[k] for s in scores) / len(scores), 2) for k in keys}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(all_avg: dict[str, dict]) -> None:
    print("\n" + "=" * 72)
    print("RUBRIC SCORES  (avg across test prompts, max 3 per dim / max 12 total)")
    print("=" * 72)
    print(f"{'Model':<14} {'Summary':>8} {'Bullets':>8} {'Limit.':>8} {'Follow-up':>10} {'Total/12':>9}")
    print("-" * 62)
    for name, r in all_avg.items():
        print(
            f"{name:<14} "
            f"{r.get('summary',0):>8.2f} "
            f"{r.get('bullets',0):>8.2f} "
            f"{r.get('limitation',0):>8.2f} "
            f"{r.get('followup',0):>10.2f} "
            f"{r.get('total',0):>9.2f}"
        )
    print("=" * 72)


def write_markdown(all_avg: dict[str, dict]) -> None:
    lines = [
        "# Rubric Quality Scores",
        "",
        "Heuristic rubric applied to cached model outputs (no model reload needed). "
        "Each dimension is scored 0–3; total is out of 12.",
        "",
        "| Model | Summary (0–3) | Bullets (0–3) | Limitation (0–3) | Follow-up (0–3) | **Total / 12** |",
        "|-------|:---:|:---:|:---:|:---:|:---:|",
    ]
    for name, r in all_avg.items():
        lines.append(
            f"| {name} "
            f"| {r.get('summary',0):.2f} "
            f"| {r.get('bullets',0):.2f} "
            f"| {r.get('limitation',0):.2f} "
            f"| {r.get('followup',0):.2f} "
            f"| **{r.get('total',0):.2f}** |"
        )
    lines += [
        "",
        "## Dimension guide",
        "",
        "| Dimension | 0 | 1 | 2 | 3 |",
        "|-----------|---|---|---|---|",
        "| **Summary** | Missing / < 8 words | 8–14 words | ≥ 15 words | ≥ 15 words + high specificity |",
        "| **Bullets** | Missing or < 5 w/bullet | 5–9 w/bullet | 10–14 w/bullet | ≥ 15 w/bullet |",
        "| **Limitation** | Missing / 'None' | Present, < 8 words | 8–18 words | > 18 words + technical |",
        "| **Follow-up** | Missing | Present, no '?' | Ends '?', < 8 words | Ends '?', ≥ 8 words |",
        "",
    ]
    RESULTS_PATH.write_text("\n".join(lines))
    print(f"Rubric report written to: {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CONTENT_CACHE.exists():
        print(f"No cached outputs at {CONTENT_CACHE}.")
        print("Run `python src/eval_content.py` first.")
        sys.exit(1)

    outputs = json.loads(CONTENT_CACHE.read_text())
    ordered = ["base"] + sorted(k for k in outputs if k != "base")

    all_avg: dict[str, dict] = {}
    for name in ordered:
        if name not in outputs:
            continue
        scores = [rubric_score(text) for text in outputs[name]]
        all_avg[name] = avg_rubric(scores)

    print_table(all_avg)
    write_markdown(all_avg)


if __name__ == "__main__":
    main()
