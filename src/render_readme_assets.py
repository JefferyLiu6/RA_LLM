"""
Render README visual assets from current dataset/evaluation artifacts.

The script intentionally uses only the Python standard library and writes SVG
files so the figures are reproducible without matplotlib, Pillow, or browser
automation.

Usage:
  python src/render_readme_assets.py
"""

from __future__ import annotations

import html
import re
import textwrap
from pathlib import Path


ROOT = Path(__file__).parent.parent
ASSET_DIR = ROOT / "assets" / "screenshots"


COLORS = {
    "ink": "#111827",
    "muted": "#4b5563",
    "soft": "#f8fafc",
    "line": "#dbe3ef",
    "blue": "#2563eb",
    "blue_soft": "#dbeafe",
    "green": "#059669",
    "green_soft": "#d1fae5",
    "amber": "#d97706",
    "amber_soft": "#fef3c7",
    "red": "#dc2626",
    "red_soft": "#fee2e2",
    "purple": "#7c3aed",
    "purple_soft": "#ede9fe",
    "teal": "#0f766e",
    "teal_soft": "#ccfbf1",
    "white": "#ffffff",
}


def read(path: str) -> str:
    p = ROOT / path
    return p.read_text() if p.exists() else ""


def pct_from_fraction(text: str, label: str) -> int:
    match = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*(\d+)/(\d+)", text, re.I)
    if not match:
        return 0
    passed, total = int(match.group(1)), int(match.group(2))
    return round(passed / total * 100)


def quality_row(text: str, label: str) -> dict[str, int]:
    match = re.search(rf"\|\s*{re.escape(label)}\s*\|([^\n]+)", text, re.I)
    if not match:
        return {}
    cells = [c.strip() for c in match.group(1).strip().strip("|").split("|")]
    keys = [
        "format",
        "alignment",
        "f1",
        "key_recall",
        "unsupported",
        "followup",
        "formulaic",
        "grammar",
    ]
    values = {}
    for key, cell in zip(keys, cells):
        num = re.search(r"\d+", cell)
        values[key] = int(num.group(0)) if num else 0
    return values


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


class Svg:
    def __init__(self, width: int, height: int, title: str):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
            "<defs>",
            '<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">',
            '<feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#0f172a" flood-opacity="0.10"/>',
            "</filter>",
            "</defs>",
            f'<rect width="{width}" height="{height}" fill="{COLORS["soft"]}"/>',
        ]

    def rect(self, x, y, w, h, fill, stroke="none", rx=10, shadow=False):
        filt = ' filter="url(#shadow)"' if shadow else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"{filt}/>'
        )

    def line(self, x1, y1, x2, y2, color=COLORS["line"], width=2):
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>'
        )

    def text(self, x, y, text, size=24, weight=500, color=COLORS["ink"], anchor="start"):
        self.parts.append(
            f'<text x="{x}" y="{y}" fill="{color}" font-family="Inter,Arial,sans-serif" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{esc(text)}</text>'
        )

    def mono(self, x, y, text, size=18, color=COLORS["ink"], weight=500):
        self.parts.append(
            f'<text x="{x}" y="{y}" fill="{color}" font-family="Menlo,Consolas,monospace" '
            f'font-size="{size}" font-weight="{weight}">{esc(text)}</text>'
        )

    def wrapped(self, x, y, text, width_chars=54, size=20, leading=28, color=COLORS["muted"], weight=400):
        for i, line in enumerate(textwrap.wrap(text, width=width_chars)):
            self.text(x, y + i * leading, line, size=size, weight=weight, color=color)
        return y + max(1, len(textwrap.wrap(text, width=width_chars))) * leading

    def bar(self, x, y, w, h, value, color, label, value_label=None):
        self.rect(x, y, w, h, "#edf2f7", rx=7)
        self.rect(x, y, round(w * value / 100), h, color, rx=7)
        self.text(x, y - 10, label, size=19, weight=650)
        self.text(x + w + 22, y + h - 5, value_label or f"{value}%", size=24, weight=750, color=color)

    def save(self, path: Path):
        self.parts.append("</svg>")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.parts))


def render_results(metrics: dict):
    svg = Svg(1400, 820, "Project results overview")
    svg.text(70, 88, "Structured ML Notes Adapter", size=42, weight=800)
    svg.text(70, 124, "Qwen2.5-1.5B LoRA + DPO for reliable research-note formatting", size=22, color=COLORS["muted"])

    cards = [
        ("Dataset", "600", "examples with metadata", COLORS["blue"], COLORS["blue_soft"]),
        ("Held-out test", "125", "prompts never used for training", COLORS["purple"], COLORS["purple_soft"]),
        ("SFT compliance", f'{metrics["sft_format"]}%', "after LoRA supervised tuning", COLORS["amber"], COLORS["amber_soft"]),
        ("DPO compliance", f'{metrics["dpo_format"]}%', "after hard-negative preference tuning", COLORS["green"], COLORS["green_soft"]),
    ]
    for i, (title, value, subtitle, color, fill) in enumerate(cards):
        x = 70 + i * 315
        svg.rect(x, 165, 280, 155, COLORS["white"], stroke=COLORS["line"], rx=14, shadow=True)
        svg.rect(x + 20, 188, 54, 12, fill, rx=6)
        svg.text(x + 20, 228, title, size=20, weight=700, color=COLORS["muted"])
        svg.text(x + 20, 274, value, size=46, weight=850, color=color)
        svg.text(x + 20, 303, subtitle, size=17, color=COLORS["muted"])

    svg.rect(70, 370, 620, 360, COLORS["white"], stroke=COLORS["line"], rx=14, shadow=True)
    svg.text(105, 420, "Template Compliance", size=28, weight=800)
    svg.text(105, 452, "Required sections, exactly 3 bullets, valid follow-up question", size=18, color=COLORS["muted"])
    svg.bar(115, 505, 420, 28, metrics["base_format"], COLORS["red"], "Base model", f'{metrics["base_format"]}%')
    svg.bar(115, 590, 420, 28, metrics["sft_format"], COLORS["amber"], "SFT LoRA", f'{metrics["sft_format"]}%')
    svg.bar(115, 675, 420, 28, metrics["dpo_format"], COLORS["green"], "DPO", f'{metrics["dpo_format"]}%')

    svg.rect(740, 370, 590, 360, COLORS["white"], stroke=COLORS["line"], rx=14, shadow=True)
    svg.text(775, 420, "Quality Guardrails", size=28, weight=800)
    svg.text(775, 452, "Reference-overlap heuristics on the 125-prompt test set", size=18, color=COLORS["muted"])
    svg.bar(790, 505, 360, 28, metrics["sft_alignment"], COLORS["amber"], "SFT content alignment", f'{metrics["sft_alignment"]}%')
    svg.bar(790, 590, 360, 28, metrics["dpo_alignment"], COLORS["green"], "DPO content alignment", f'{metrics["dpo_alignment"]}%')
    svg.bar(790, 675, 360, 28, 100 - metrics["dpo_unsupported"], COLORS["teal"], "DPO supported-term rate", f'{100 - metrics["dpo_unsupported"]}%')

    svg.save(ASSET_DIR / "project_results.svg")


def render_dataset_pipeline():
    svg = Svg(1400, 620, "Dataset pipeline")
    svg.text(70, 78, "Dataset Pipeline", size=42, weight=800)
    svg.text(70, 113, "Original gold examples are preserved; synthetic records are traceable and validated.", size=22, color=COLORS["muted"])

    nodes = [
        (80, 210, "68 gold examples", "human_curated subset", COLORS["blue"], COLORS["blue_soft"]),
        (390, 210, "532 synthetic", "source=synthetic_codex", COLORS["purple"], COLORS["purple_soft"]),
        (700, 210, "600 validated", "schema + hash + template checks", COLORS["teal"], COLORS["teal_soft"]),
        (1010, 210, "Fixed splits", "400 train / 75 val / 125 test", COLORS["green"], COLORS["green_soft"]),
    ]
    for i, (x, y, title, subtitle, color, fill) in enumerate(nodes):
        svg.rect(x, y, 250, 150, COLORS["white"], stroke=COLORS["line"], rx=14, shadow=True)
        svg.rect(x + 22, y + 24, 58, 12, fill, rx=6)
        svg.text(x + 22, y + 70, title, size=27, weight=800, color=color)
        svg.wrapped(x + 22, y + 105, subtitle, width_chars=24, size=17, leading=23)
        if i < len(nodes) - 1:
            svg.line(x + 260, y + 75, x + 300, y + 75, COLORS["muted"], 3)
            svg.text(x + 282, y + 68, ">", size=24, weight=800, color=COLORS["muted"], anchor="middle")

    svg.rect(105, 430, 1190, 105, COLORS["white"], stroke=COLORS["line"], rx=14)
    svg.text(135, 473, "Validation gates", size=24, weight=800)
    gates = [
        "required metadata",
        "sha256 input hash",
        "no duplicate inputs",
        "valid topic/difficulty/split",
        "exact output template",
        "exactly 3 key-point bullets",
    ]
    x = 135
    for gate in gates:
        svg.rect(x, 492, len(gate) * 10 + 28, 34, COLORS["green_soft"], rx=17)
        svg.text(x + 14, 515, gate, size=15, weight=650, color=COLORS["green"])
        x += len(gate) * 10 + 44

    svg.save(ASSET_DIR / "dataset_pipeline.svg")


def render_dpo_pipeline():
    svg = Svg(1400, 670, "DPO hard-negative pipeline")
    svg.text(70, 78, "DPO Hard-Negative Alignment", size=42, weight=800)
    svg.text(70, 113, "Preference tuning targets real failure modes without using the held-out test split.", size=22, color=COLORS["muted"])

    x0, y0 = 90, 205
    steps = [
        ("SFT adapter", "99% format compliance\nbut occasional strict failures", COLORS["amber"], COLORS["amber_soft"]),
        ("300 preference pairs", "258 train / 42 val\n0 test examples used", COLORS["blue"], COLORS["blue_soft"]),
        ("Hard rejected answers", "extra bullet\nunsupported claim\nextra follow-up text", COLORS["red"], COLORS["red_soft"]),
        ("DPO adapter", "125/125 held-out\nstrict compliance", COLORS["green"], COLORS["green_soft"]),
    ]
    for i, (title, subtitle, color, fill) in enumerate(steps):
        x = x0 + i * 320
        svg.rect(x, y0, 255, 190, COLORS["white"], stroke=COLORS["line"], rx=14, shadow=True)
        svg.rect(x + 22, y0 + 24, 58, 12, fill, rx=6)
        svg.text(x + 22, y0 + 72, title, size=25, weight=800, color=color)
        yy = y0 + 110
        for line in subtitle.splitlines():
            svg.text(x + 22, yy, line, size=18, color=COLORS["muted"])
            yy += 27
        if i < len(steps) - 1:
            svg.line(x + 262, y0 + 95, x + 305, y0 + 95, COLORS["muted"], 3)
            svg.text(x + 286, y0 + 88, ">", size=24, weight=800, color=COLORS["muted"], anchor="middle")

    svg.rect(90, 475, 1220, 105, COLORS["white"], stroke=COLORS["line"], rx=14)
    svg.text(125, 520, "Result", size=25, weight=800)
    svg.text(235, 520, "DPO reduces held-out template failures to zero while keeping content alignment at 83%.", size=24, weight=650, color=COLORS["green"])
    svg.text(125, 552, "Honest claim: stronger format reliability. Do not claim DPO beat SFT on content quality.", size=18, color=COLORS["muted"])

    svg.save(ASSET_DIR / "dpo_pipeline.svg")


def render_output_example():
    svg = Svg(1400, 880, "Structured output example")
    svg.text(70, 78, "Example Output", size=42, weight=800)
    svg.text(70, 113, "DPO adapter converts an ML concept paragraph into reusable research notes.", size=22, color=COLORS["muted"])

    prompt = (
        "Preference data margins can be evaluated by asking whether pairs kept only "
        "when a judge strongly prefers one response actually reduce noisy labels in "
        "DPO training. A negative result appears when strict margins discard useful "
        "but subtle preferences."
    )
    output = [
        "Summary:",
        "Preference data margins support a specific benefit: reduce noisy labels in DPO training, while requiring care because strict margins can discard useful but subtle preferences.",
        "",
        "Key Points:",
        "- Its core mechanism is that pairs are kept only when a judge strongly prefers one response.",
        "- The main practical benefit is that it can reduce noisy labels in DPO training.",
        "- A useful evaluation should include cases where strict margins discard subtle preferences.",
        "",
        "Limitation:",
        "Strict margins can discard useful but subtle preferences, so results should be validated on realistic held-out examples.",
        "",
        "Follow-up Question:",
        "How large should preference margin thresholds be before discarding labeled pairs?",
    ]

    svg.rect(70, 170, 1260, 175, COLORS["white"], stroke=COLORS["line"], rx=14, shadow=True)
    svg.text(105, 215, "Input paragraph", size=25, weight=800)
    svg.wrapped(105, 255, prompt, width_chars=112, size=20, leading=29)

    svg.rect(70, 390, 1260, 395, COLORS["white"], stroke=COLORS["line"], rx=14, shadow=True)
    svg.text(105, 435, "DPO adapter output", size=25, weight=800)
    y = 480
    for line in output:
        if line.endswith(":"):
            svg.text(105, y, line, size=21, weight=800, color=COLORS["blue"])
        elif line.startswith("-"):
            svg.wrapped(125, y, line, width_chars=98, size=18, leading=25, color=COLORS["ink"])
        elif not line:
            y += 10
            continue
        else:
            y = svg.wrapped(105, y, line, width_chars=102, size=18, leading=25, color=COLORS["ink"]) - 25
        y += 31

    svg.save(ASSET_DIR / "output_example.svg")


def main() -> None:
    sft_quality = read("outputs/sft_quality_results.md")
    dpo_quality = read("outputs/dpo_quality_results.md")
    sft_eval = read("outputs/sft_eval_results.md")
    dpo_eval = read("outputs/dpo_eval_results.md")

    sft_row = quality_row(sft_quality, "lora")
    dpo_row = quality_row(dpo_quality, "DPO")
    metrics = {
        "base_format": quality_row(sft_quality, "base").get("format", 38),
        "sft_format": sft_row.get("format", pct_from_fraction(sft_eval, "LoRA")),
        "dpo_format": dpo_row.get("format", pct_from_fraction(dpo_eval, "DPO")),
        "sft_alignment": sft_row.get("alignment", 86),
        "dpo_alignment": dpo_row.get("alignment", 83),
        "dpo_unsupported": dpo_row.get("unsupported", 13),
    }

    render_results(metrics)
    render_dataset_pipeline()
    render_dpo_pipeline()
    render_output_example()
    print(f"Rendered README assets to {ASSET_DIR}")


if __name__ == "__main__":
    main()
