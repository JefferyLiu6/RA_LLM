"""
Generate a self-contained HTML dashboard for comparing LoRA experiments.

Usage:
  python src/dashboard.py           # generate + open in browser
  python src/dashboard.py --no-open # generate only
"""

import argparse
import json
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

EXPERIMENTS_DIR   = ROOT / "outputs" / "experiments"
CONTENT_CACHE     = ROOT / "outputs" / "content_outputs.json"
TEST_PROMPTS_PATH = ROOT / "data" / "test_prompts.jsonl"
DASHBOARD_PATH    = ROOT / "outputs" / "dashboard.html"

EXP_COLORS = {
    "baseline": "#6366f1",
    "rank_8":   "#f59e0b",
    "rank_32":  "#10b981",
    "rank_64":  "#ef4444",
    "epochs_5": "#8b5cf6",
    "lr_1e-4":  "#06b6d4",
    "lr_5e-4":  "#f97316",
}
FALLBACK_COLORS = ["#6366f1","#f59e0b","#10b981","#ef4444","#8b5cf6","#06b6d4","#f97316","#ec4899"]
BASE_COLOR = "#94a3b8"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data() -> dict:
    prompts = []
    if TEST_PROMPTS_PATH.exists():
        with open(TEST_PROMPTS_PATH) as f:
            prompts = [json.loads(l) for l in f if l.strip()]

    experiments = {}
    if EXPERIMENTS_DIR.exists():
        for d in sorted(EXPERIMENTS_DIR.iterdir()):
            if not d.is_dir():
                continue
            meta_p = d / "training_meta.json"
            loss_p = d / "loss_log.json"
            if not meta_p.exists():
                continue
            meta = json.loads(meta_p.read_text())
            log  = json.loads(loss_p.read_text()) if loss_p.exists() else []
            train_e = [e for e in log if "loss" in e and "eval_loss" not in e]
            val_e   = [e for e in log if "eval_loss" in e]
            experiments[d.name] = {
                "meta":             meta,
                "train_curve":      [{"x": round(e.get("epoch",0),3), "y": round(e["loss"],4)} for e in train_e],
                "val_curve":        [{"x": round(e.get("epoch",0),3), "y": round(e["eval_loss"],4)} for e in val_e],
                "final_train_loss": round(train_e[-1]["loss"],4) if train_e else None,
                "final_val_loss":   round(val_e[-1]["eval_loss"],4) if val_e else None,
            }

    outputs, has_content = {}, False
    if CONTENT_CACHE.exists():
        outputs     = json.loads(CONTENT_CACHE.read_text())
        has_content = True

    compliance, content_metrics_data = {}, {}
    if has_content and prompts:
        try:
            from eval_template import check_compliance
            from eval_content   import avg_metrics, content_metrics
            n = len(prompts)
            for name in ["base"] + list(experiments.keys()):
                if name not in outputs:
                    continue
                checks = [check_compliance(o) for o in outputs[name]]
                compliance[name] = {
                    "pass": sum(1 for c in checks if c["all_pass"]),
                    "n":    n,
                    "pct":  round(sum(1 for c in checks if c["all_pass"]) / n * 100),
                    "section_scores": {
                        k: sum(1 for c in checks if c.get(k, False))
                        for k in ("summary","key_points","three_bullets","limitation","follow_up")
                    },
                }
                per_prompt = [content_metrics(t) for t in outputs[name]]
                content_metrics_data[name] = avg_metrics(per_prompt)
        except Exception as exc:
            print(f"Warning: metrics computation failed — {exc}")

    return dict(
        experiments=experiments,
        prompts=prompts,
        outputs=outputs,
        compliance=compliance,
        content_metrics=content_metrics_data,
        has_content=has_content,
    )


def compute_summary(data: dict) -> dict:
    comp = {k: v for k, v in data["compliance"].items() if k != "base"}
    exps = data["experiments"]

    best_comp = max(comp.items(), key=lambda x: x[1]["pct"], default=None)
    best_pct  = best_comp[1]["pct"] if best_comp else 0

    val_losses = {k: v["final_val_loss"] for k, v in exps.items() if v["final_val_loss"] is not None}
    best_val   = min(val_losses.items(), key=lambda x: x[1], default=None)

    candidates = [(n, exps[n]["meta"].get("lora_r", 9999))
                  for n, c in comp.items() if c["pct"] >= best_pct and n in exps]
    efficient  = min(candidates, key=lambda x: x[1], default=None)

    return {
        "best_compliance": {"name": best_comp[0] if best_comp else "n/a",
                            "value": f"{best_pct}%"            if best_comp else "n/a"},
        "best_val_loss":   {"name": best_val[0]  if best_val  else "n/a",
                            "value": f"{best_val[1]:.4f}"      if best_val  else "n/a"},
        "most_efficient":  {"name": efficient[0]  if efficient else "n/a",
                            "value": f"r={efficient[1]}"       if efficient else "n/a"},
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(data: dict) -> str:
    exp_names = list(data["experiments"].keys())
    colors = {n: EXP_COLORS.get(n, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
              for i, n in enumerate(exp_names)}
    colors["base"] = BASE_COLOR

    summary  = compute_summary(data)
    n_exp    = len(exp_names)

    dash = {
        "experimentNames": exp_names,
        "colors": colors,
        "summary": summary,
        "compliance": data["compliance"],
        "experiments": {
            k: {
                "meta":           v["meta"],
                "trainCurve":     v["train_curve"],
                "valCurve":       v["val_curve"],
                "finalTrainLoss": v["final_train_loss"],
                "finalValLoss":   v["final_val_loss"],
            } for k, v in data["experiments"].items()
        },
        "contentMetrics": data["content_metrics"],
        "outputs": data["outputs"],
        "prompts": [{"id": p.get("id", f"test_{i+1:02d}"), "input": p["input"]}
                    for i, p in enumerate(data["prompts"])],
        "hasContent": data["has_content"],
        "nExp": n_exp,
    }
    data_json = json.dumps(dash, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LoRA Fine-Tuning — Experiment Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;1,14..32,400&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
  --bg:        #f8fafc;
  --surface:   #ffffff;
  --border:    #e2e8f0;
  --border-sm: #f1f5f9;
  --text:      #0f172a;
  --text-2:    #334155;
  --muted:     #94a3b8;
  --accent:    #6366f1;
  --accent-lo: #eef2ff;
  --accent-md: rgba(99,102,241,0.15);
  --green:     #10b981;
  --amber:     #f59e0b;
  --red:       #ef4444;
  --shadow-sm: 0 1px 2px rgba(0,0,0,.05);
  --shadow:    0 1px 3px rgba(0,0,0,.07), 0 2px 8px rgba(0,0,0,.05);
  --radius:    8px;
  --radius-lg: 12px;
  --mono:      'JetBrains Mono', 'Fira Code', monospace;
}}

body {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  background: var(--bg);
  color: var(--text);
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
  min-height: 100vh;
}}

/* ─── Scrollbar ─────────────────────────────── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--muted); }}

/* ─── Top bar ───────────────────────────────── */
.topbar {{
  position: sticky; top: 0; z-index: 100;
  background: rgba(255,255,255,0.9);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  height: 52px;
  display: flex; align-items: center;
  padding: 0 24px; gap: 0;
}}
.brand {{
  display: flex; align-items: center; gap: 10px;
  margin-right: 32px; flex-shrink: 0;
}}
.brand-icon {{
  width: 30px; height: 30px;
  background: var(--accent);
  border-radius: 7px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}}
.brand-icon svg {{ width: 16px; height: 16px; }}
.brand-name {{
  font-size: 14px; font-weight: 600; letter-spacing: -.2px;
  color: var(--text); line-height: 1.2;
}}
.brand-sub {{
  font-family: var(--mono); font-size: 11px;
  color: var(--muted); line-height: 1;
}}
.tabs {{ display: flex; gap: 2px; flex: 1; }}
.tab {{
  padding: 5px 14px; border-radius: 6px;
  border: none; background: none; cursor: pointer;
  font-family: inherit; font-size: 13px; font-weight: 500;
  color: var(--muted); transition: all .15s; white-space: nowrap;
}}
.tab:hover {{ color: var(--text-2); background: var(--bg); }}
.tab.active {{ color: var(--accent); background: var(--accent-lo); }}
.topbar-right {{
  margin-left: auto; flex-shrink: 0;
  font-family: var(--mono); font-size: 11px;
  color: var(--muted); padding-left: 16px;
  border-left: 1px solid var(--border);
}}

/* ─── Hero strip ────────────────────────────── */
.hero {{
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 16px 24px;
  display: flex; gap: 12px; flex-wrap: wrap; align-items: stretch;
}}
.kpi {{
  flex: 1; min-width: 130px;
  padding: 12px 16px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg);
}}
.kpi.highlight {{
  background: var(--accent-lo);
  border-color: rgba(99,102,241,0.2);
}}
.kpi-label {{
  font-size: 10px; font-weight: 600;
  text-transform: uppercase; letter-spacing: .07em;
  color: var(--muted); margin-bottom: 6px;
}}
.kpi-value {{
  font-family: var(--mono); font-size: 22px; font-weight: 600;
  color: var(--text); line-height: 1; margin-bottom: 4px;
}}
.kpi.highlight .kpi-value {{ color: var(--accent); }}
.kpi-name {{
  font-size: 11px; color: var(--muted);
  font-family: var(--mono);
}}
.kpi-divider {{
  width: 1px; background: var(--border);
  margin: 4px 0; flex-shrink: 0;
  display: none;
}}
@media (min-width: 640px) {{ .kpi-divider {{ display: block; }} }}

/* ─── Main ──────────────────────────────────── */
.main {{
  max-width: 1320px; margin: 0 auto;
  padding: 24px; display: flex; flex-direction: column; gap: 16px;
}}
.panel {{ display: none; flex-direction: column; gap: 16px; }}
.panel.active {{
  display: flex;
  animation: fade-up .2s ease both;
}}
@keyframes fade-up {{
  from {{ opacity: 0; transform: translateY(6px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
.row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.row > * {{ flex: 1; min-width: 0; }}
.row-fixed > :first-child {{ flex: 0 0 55%; }}
.row-fixed > :last-child  {{ flex: 1; }}

/* ─── Card ──────────────────────────────────── */
.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px;
  box-shadow: var(--shadow-sm);
}}
.card-hd {{
  display: flex; align-items: flex-start;
  justify-content: space-between; gap: 12px;
  margin-bottom: 16px;
}}
.card-title {{
  font-size: 13px; font-weight: 600; color: var(--text);
}}
.card-desc {{
  font-size: 11px; color: var(--muted); margin-top: 2px;
}}
.chart-wrap {{ position: relative; }}
.h240 {{ height: 240px; }}
.h280 {{ height: 280px; }}
.h320 {{ height: 320px; }}
.h300 {{ height: 300px; }}

/* ─── Legend ────────────────────────────────── */
.legend {{
  display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px;
}}
.legend-item {{
  display: flex; align-items: center; gap: 5px;
  font-size: 11px; color: var(--muted);
  font-family: var(--mono);
}}
.legend-dot {{
  width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
}}

/* ─── Table ─────────────────────────────────── */
.tbl-wrap {{
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden; overflow-x: auto;
  background: var(--surface);
  box-shadow: var(--shadow-sm);
}}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
thead th {{
  padding: 10px 16px;
  text-align: left; white-space: nowrap;
  font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: .06em;
  color: var(--muted); background: var(--bg);
  border-bottom: 1px solid var(--border);
}}
tbody tr {{ border-bottom: 1px solid var(--border-sm); transition: background .1s; }}
tbody tr:last-child {{ border-bottom: none; }}
tbody tr:hover {{ background: var(--bg); }}
td {{ padding: 11px 16px; color: var(--text-2); vertical-align: middle; }}
.td-exp {{
  display: flex; align-items: center; gap: 8px;
  font-weight: 500; color: var(--text); font-family: var(--mono);
  font-size: 12px;
}}
.td-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
.td-mono {{ font-family: var(--mono); font-size: 12px; }}
.badge-best {{
  font-size: 10px; font-weight: 700; letter-spacing: .04em;
  padding: 2px 7px; border-radius: 4px;
  background: #fef9c3; color: #854d0e;
  border: 1px solid #fde68a;
}}
.bar-cell {{ display: flex; align-items: center; gap: 8px; }}
.mini-bar {{
  height: 4px; width: 72px; background: var(--border);
  border-radius: 2px; overflow: hidden; flex-shrink: 0;
}}
.mini-fill {{ height: 100%; border-radius: 2px; }}

/* ─── Output viewer ─────────────────────────── */
.viewer-ctrl {{
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
}}
.viewer-lbl {{
  font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: .06em;
  color: var(--muted); white-space: nowrap;
}}
select {{
  font-family: var(--mono); font-size: 12px;
  padding: 7px 32px 7px 12px; min-width: 200px;
  border: 1px solid var(--border); border-radius: var(--radius);
  background: var(--surface) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%2394a3b8'/%3E%3C/svg%3E") no-repeat right 10px center;
  -webkit-appearance: none; appearance: none;
  color: var(--text); cursor: pointer; outline: none;
  transition: border-color .15s;
}}
select:focus {{ border-color: var(--accent); }}
.prompt-quote {{
  background: var(--bg); border-left: 3px solid var(--accent);
  border-radius: 0 var(--radius) var(--radius) 0;
  padding: 12px 16px;
  font-size: 13px; color: var(--text-2); line-height: 1.65;
  display: none;
}}
.prompt-quote.visible {{ display: block; }}
.out-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 12px;
}}
.out-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  box-shadow: var(--shadow-sm);
  transition: border-color .15s, box-shadow .15s;
}}
.out-card:hover {{
  border-color: #cbd5e1;
  box-shadow: var(--shadow);
}}
.out-hd {{
  padding: 9px 14px; background: var(--bg);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 7px;
}}
.out-hd-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
.out-hd-name {{
  font-family: var(--mono); font-size: 11px; font-weight: 600;
  color: var(--text);
}}
.out-body {{
  padding: 14px; font-family: var(--mono); font-size: 11.5px;
  line-height: 1.75; color: var(--text-2); white-space: pre-wrap;
  max-height: 400px; overflow-y: auto;
}}
.out-sec {{ font-weight: 700; color: var(--text); }}

/* ─── Empty state ───────────────────────────── */
.empty {{
  border: 1px dashed var(--border); border-radius: var(--radius-lg);
  background: var(--surface); text-align: center;
  padding: 48px 24px; color: var(--muted);
}}
.empty strong {{ display: block; font-size: 14px; color: var(--text-2); margin-bottom: 6px; font-weight: 500; }}
.empty code {{
  display: inline-block; margin-top: 12px;
  font-family: var(--mono); font-size: 12px;
  background: var(--bg); border: 1px solid var(--border);
  border-radius: 5px; padding: 5px 12px; color: var(--text);
}}

/* ─── Before/After chat UI ──────────────────── */
.compare-ctrl {{
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  padding: 16px 20px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-lg); margin-bottom: 16px;
}}
.chat-cols {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
  align-items: start;
}}
.chat-win {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  box-shadow: var(--shadow);
  display: flex; flex-direction: column;
}}
.chat-header {{
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 10px;
  background: var(--bg);
}}
.chat-avatar {{
  width: 28px; height: 28px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; font-size: 12px; font-weight: 700; color: #fff;
}}
.chat-model-name {{
  font-size: 12px; font-weight: 600; color: var(--text); line-height: 1.2;
}}
.chat-model-sub {{
  font-family: var(--mono); font-size: 10px; color: var(--muted);
}}
.chat-badge {{
  margin-left: auto; flex-shrink: 0;
  font-size: 10px; font-weight: 600; padding: 2px 7px;
  border-radius: 10px; letter-spacing: .03em;
}}
/* badge colors set inline via style attr */
.chat-messages {{
  padding: 14px 12px; display: flex; flex-direction: column; gap: 10px;
  min-height: 200px; max-height: 600px; overflow-y: auto;
}}
.chat-msg {{
  display: flex; gap: 8px; align-items: flex-start;
}}
.chat-msg.user {{ flex-direction: row-reverse; }}
.msg-icon {{
  width: 24px; height: 24px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; margin-top: 1px;
}}
.msg-icon.user-icon {{
  background: var(--border); color: var(--muted);
}}
.msg-bubble {{
  border-radius: 14px; padding: 9px 13px;
  font-size: 12.5px; line-height: 1.65; max-width: 85%;
}}
.chat-msg.user .msg-bubble {{
  background: var(--accent); color: #fff;
  border-bottom-right-radius: 4px;
}}
.chat-msg.assistant .msg-bubble {{
  background: var(--bg); border: 1px solid var(--border);
  color: var(--text-2); border-bottom-left-radius: 4px;
  font-family: var(--mono); font-size: 11.5px; white-space: pre-wrap;
}}
.chat-sec {{ font-weight: 700; color: var(--text); }}
.chat-typing {{
  display: flex; gap: 4px; align-items: center; padding: 12px 14px;
}}
.chat-typing span {{
  width: 5px; height: 5px; background: var(--muted);
  border-radius: 50%; animation: blink 1.2s infinite;
}}
.chat-typing span:nth-child(2) {{ animation-delay: .2s; }}
.chat-typing span:nth-child(3) {{ animation-delay: .4s; }}
@keyframes blink {{
  0%, 80%, 100% {{ opacity: .3; transform: scale(.8); }}
  40% {{ opacity: 1; transform: scale(1); }}
}}
.chat-footer {{
  padding: 10px 12px;
  border-top: 1px solid var(--border);
  background: var(--bg);
  display: flex; align-items: center; gap: 8px;
}}
.chat-footer-input {{
  flex: 1; font-family: var(--mono); font-size: 11px;
  color: var(--muted); background: none; border: none; outline: none;
  cursor: default;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.send-btn {{
  width: 26px; height: 26px; border-radius: 50%;
  background: var(--accent); border: none; cursor: default;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; opacity: .5;
}}
.send-btn svg {{ width: 12px; height: 12px; }}

/* ─── Responsive ────────────────────────────── */
@media (max-width: 900px) {{
  .row {{ flex-direction: column; }}
  .row-fixed > :first-child, .row-fixed > :last-child {{ flex: 1; }}
}}
@media (max-width: 600px) {{
  .topbar {{ padding: 0 16px; }}
  .hero {{ padding: 12px 16px; }}
  .main {{ padding: 16px; }}
  .brand-sub {{ display: none; }}
}}
</style>
</head>
<body>

<!-- ── Top bar ───────────────────────────────────────────────────── -->
<header class="topbar">
  <div class="brand">
    <div class="brand-icon">
      <svg viewBox="0 0 16 16" fill="none">
        <path d="M2 8C2 4.686 4.686 2 8 2s6 2.686 6 6-2.686 6-6 6-6-2.686-6-6z" stroke="white" stroke-width="1.5"/>
        <path d="M5.5 8h5M8 5.5v5" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
      </svg>
    </div>
    <div>
      <div class="brand-name">LoRA Research</div>
      <div class="brand-sub">Qwen2.5-1.5B-Instruct</div>
    </div>
  </div>
  <nav class="tabs">
    <button class="tab active" onclick="switchTab('results')">Results</button>
    <button class="tab" onclick="switchTab('curves')">Training Curves</button>
    <button class="tab" onclick="switchTab('content')">Content Analysis</button>
    <button class="tab" onclick="switchTab('outputs')">Output Viewer</button>
    <button class="tab" onclick="switchTab('compare')">Before / After</button>
  </nav>
  <div class="topbar-right" id="exp-meta"></div>
</header>

<!-- ── Hero KPIs ─────────────────────────────────────────────────── -->
<section class="hero">
  <div class="kpi highlight">
    <div class="kpi-label">Best Compliance</div>
    <div class="kpi-value" id="kpi-compliance">—</div>
    <div class="kpi-name"  id="kpi-compliance-name">—</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Best Val Loss</div>
    <div class="kpi-value" id="kpi-val">—</div>
    <div class="kpi-name"  id="kpi-val-name">—</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Most Efficient</div>
    <div class="kpi-value" id="kpi-eff">—</div>
    <div class="kpi-name"  id="kpi-eff-name">—</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Base Compliance</div>
    <div class="kpi-value" id="kpi-base">—</div>
    <div class="kpi-name">before fine-tuning</div>
  </div>
</section>

<!-- ── Main content ───────────────────────────────────────────────── -->
<main class="main">

  <!-- Results panel -->
  <div id="panel-results" class="panel active">
    <div class="row">
      <div class="card">
        <div class="card-hd">
          <div>
            <div class="card-title">Format Compliance</div>
            <div class="card-desc">% of outputs passing all 4 template checks</div>
          </div>
        </div>
        <div class="chart-wrap h240"><canvas id="c-compliance"></canvas></div>
        <div class="legend" id="lg-compliance"></div>
      </div>
      <div class="card">
        <div class="card-hd">
          <div>
            <div class="card-title">Final Loss</div>
            <div class="card-desc">Train vs validation loss at end of training</div>
          </div>
        </div>
        <div class="chart-wrap h240"><canvas id="c-loss"></canvas></div>
        <div class="legend" id="lg-loss"></div>
      </div>
    </div>
    <div class="tbl-wrap">
      <table id="leaderboard">
        <thead>
          <tr>
            <th>Experiment</th>
            <th>LoRA R</th>
            <th>Learning Rate</th>
            <th>Epochs</th>
            <th>Train Loss</th>
            <th>Val Loss</th>
            <th>Compliance</th>
            <th></th>
          </tr>
        </thead>
        <tbody id="tbl-body"></tbody>
      </table>
    </div>
  </div>

  <!-- Curves panel -->
  <div id="panel-curves" class="panel">
    <div class="row">
      <div class="card">
        <div class="card-hd">
          <div>
            <div class="card-title">Training Loss</div>
            <div class="card-desc">Loss on training set over epochs</div>
          </div>
        </div>
        <div class="chart-wrap h300"><canvas id="c-train"></canvas></div>
        <div class="legend" id="lg-train"></div>
      </div>
      <div class="card">
        <div class="card-hd">
          <div>
            <div class="card-title">Validation Loss</div>
            <div class="card-desc">Loss on held-out set after each epoch</div>
          </div>
        </div>
        <div class="chart-wrap h300"><canvas id="c-val"></canvas></div>
        <div class="legend" id="lg-val"></div>
      </div>
    </div>
  </div>

  <!-- Content panel -->
  <div id="panel-content" class="panel">
    <div id="content-empty" class="empty" style="display:none">
      <strong>Content analysis not yet generated</strong>
      Run eval_content.py first, then regenerate the dashboard.
      <code>python src/eval_content.py</code>
    </div>
    <div id="content-charts" class="panel active" style="gap:16px">
      <div class="row">
        <div class="card">
          <div class="card-hd">
            <div>
              <div class="card-title">Multi-Metric Radar</div>
              <div class="card-desc">Normalised across words, diversity, specificity, detail, question validity</div>
            </div>
          </div>
          <div class="chart-wrap h300"><canvas id="c-radar"></canvas></div>
          <div class="legend" id="lg-radar"></div>
        </div>
        <div class="card">
          <div class="card-hd">
            <div>
              <div class="card-title">Avg Words per Key Point</div>
              <div class="card-desc">Higher = more detailed bullet explanations</div>
            </div>
          </div>
          <div class="chart-wrap h300"><canvas id="c-ptwords"></canvas></div>
        </div>
      </div>
      <div class="card">
        <div class="card-hd">
          <div>
            <div class="card-title">Output Specificity</div>
            <div class="card-desc">Content tokens ÷ total tokens — higher means denser, less filler</div>
          </div>
        </div>
        <div class="chart-wrap h240"><canvas id="c-specif"></canvas></div>
      </div>
    </div>
  </div>

  <!-- Outputs panel -->
  <div id="panel-outputs" class="panel">
    <div id="outputs-empty" class="empty" style="display:none">
      <strong>Content outputs not yet generated</strong>
      Run eval_content.py first, then regenerate the dashboard.
      <code>python src/eval_content.py</code>
    </div>
    <div id="outputs-ui">
      <div class="card" style="padding:16px 20px">
        <div class="viewer-ctrl">
          <span class="viewer-lbl">Prompt</span>
          <select id="prompt-sel" onchange="renderOutputs()">
            <option value="">Select a test prompt…</option>
          </select>
        </div>
      </div>
      <div class="prompt-quote" id="prompt-txt"></div>
      <div class="out-grid" id="out-grid"></div>
    </div>
  </div>

  <!-- Before / After panel -->
  <div id="panel-compare" class="panel">
    <div id="compare-empty" class="empty" style="display:none">
      <strong>Content outputs not yet generated</strong>
      Run eval_content.py first, then regenerate the dashboard.
      <code>python src/eval_content.py</code>
    </div>
    <div id="compare-ui">
      <div class="compare-ctrl">
        <span class="viewer-lbl">Prompt</span>
        <select id="compare-sel" onchange="renderCompare()">
          <option value="">Select a test prompt…</option>
        </select>
      </div>
      <div class="chat-cols" id="chat-cols"></div>
    </div>
  </div>

</main>

<script>
const DASH = {data_json};

// ── Utils ──────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

function hex2rgba(hex, a) {{
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${{r}},${{g}},${{b}},${{a}})`;
}}

function fmt(v, dec=4) {{
  return v != null ? Number(v).toFixed(dec) : '—';
}}

// ── Chart defaults ─────────────────────────────────────────────────
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size   = 12;
Chart.defaults.color       = '#94a3b8';

const GRID = {{ color: '#f1f5f9', drawBorder: false }};
const TICK = {{ color: '#94a3b8', font: {{ size: 11 }} }};

// ── Tab switching ──────────────────────────────────────────────────
const PANELS = ['results','curves','content','outputs','compare'];
function switchTab(name) {{
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', PANELS[i]===name));
  PANELS.forEach(p => $('panel-'+p).classList.toggle('active', p===name));
}}

// ── Legend ─────────────────────────────────────────────────────────
function makeLegend(id, items) {{
  $(id).innerHTML = items.map(it =>
    `<span class="legend-item">
       <span class="legend-dot" style="background:${{it.c}}"></span>
       ${{it.l}}
     </span>`).join('');
}}

// ── KPIs ───────────────────────────────────────────────────────────
function buildKPIs() {{
  const s = DASH.summary;
  $('kpi-compliance').textContent      = s.best_compliance.value;
  $('kpi-compliance-name').textContent = s.best_compliance.name;
  $('kpi-val').textContent             = s.best_val_loss.value;
  $('kpi-val-name').textContent        = s.best_val_loss.name;
  $('kpi-eff').textContent             = s.most_efficient.value;
  $('kpi-eff-name').textContent        = s.most_efficient.name;

  const base = DASH.compliance['base'];
  $('kpi-base').textContent = base ? base.pct + '%' : '—';
  $('exp-meta').textContent = `${{DASH.nExp}} experiments`;
}}

// ── Compliance bar chart ───────────────────────────────────────────
function buildCompliance() {{
  const names  = DASH.experimentNames;
  const comp   = DASH.compliance;
  const colors = names.map(n => DASH.colors[n]);
  const lora   = names.map(n => comp[n]?.pct ?? 0);
  const base   = comp['base']?.pct ?? 0;

  new Chart($('c-compliance'), {{
    type: 'bar',
    data: {{
      labels: names,
      datasets: [
        {{
          label: 'LoRA',
          data: lora,
          backgroundColor: colors.map(c => hex2rgba(c, 0.85)),
          borderColor: colors,
          borderWidth: 1.5,
          borderRadius: 5,
          borderSkipped: false,
        }},
        {{
          label: 'Base model',
          data: names.map(() => base),
          backgroundColor: hex2rgba('#94a3b8', 0.3),
          borderColor: '#94a3b8',
          borderWidth: 1.5,
          borderRadius: 5,
          borderSkipped: false,
        }},
      ],
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: c => ` ${{c.dataset.label}}: ${{c.raw}}%` }} }},
      }},
      scales: {{
        x: {{ grid: GRID, ticks: TICK }},
        y: {{ grid: GRID, ticks: {{ ...TICK, callback: v => v+'%' }}, max: 100, beginAtZero: true }},
      }},
    }},
  }});

  makeLegend('lg-compliance', [
    {{ c: '#6366f1', l: 'LoRA (per experiment)' }},
    {{ c: '#94a3b8', l: 'Base model (no fine-tuning)' }},
  ]);
}}

// ── Loss bar chart ─────────────────────────────────────────────────
function buildLoss() {{
  const names = DASH.experimentNames;
  const exps  = DASH.experiments;

  new Chart($('c-loss'), {{
    type: 'bar',
    data: {{
      labels: names,
      datasets: [
        {{
          label: 'Train loss',
          data: names.map(n => exps[n]?.finalTrainLoss ?? null),
          backgroundColor: hex2rgba('#6366f1', 0.7),
          borderColor: '#6366f1',
          borderWidth: 1.5, borderRadius: 5, borderSkipped: false,
        }},
        {{
          label: 'Val loss',
          data: names.map(n => exps[n]?.finalValLoss ?? null),
          backgroundColor: hex2rgba('#10b981', 0.7),
          borderColor: '#10b981',
          borderWidth: 1.5, borderRadius: 5, borderSkipped: false,
        }},
      ],
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: c => ` ${{c.dataset.label}}: ${{fmt(c.raw)}}` }} }},
      }},
      scales: {{
        x: {{ grid: GRID, ticks: TICK }},
        y: {{ grid: GRID, ticks: TICK, beginAtZero: false }},
      }},
    }},
  }});

  makeLegend('lg-loss', [
    {{ c: '#6366f1', l: 'Train loss' }},
    {{ c: '#10b981', l: 'Val loss'   }},
  ]);
}}

// ── Leaderboard table ──────────────────────────────────────────────
function buildTable() {{
  const names = DASH.experimentNames;
  const comp  = DASH.compliance;
  const exps  = DASH.experiments;
  const bestPct = Math.max(...names.map(n => comp[n]?.pct ?? 0));

  const rows = names.map(n => {{
    const c   = comp[n] ?? {{}};
    const e   = exps[n] ?? {{}};
    const m   = e.meta ?? {{}};
    const pct = c.pct ?? 0;
    const col = DASH.colors[n];
    const isBest = pct === bestPct && bestPct > 0;

    return `<tr>
      <td><span class="td-exp">
        <span class="td-dot" style="background:${{col}}"></span>
        ${{n}}
      </span></td>
      <td class="td-mono">${{m.lora_r ?? '—'}}</td>
      <td class="td-mono">${{m.learning_rate ? Number(m.learning_rate).toExponential(0) : '—'}}</td>
      <td class="td-mono">${{m.epochs ?? '—'}}</td>
      <td class="td-mono">${{fmt(e.finalTrainLoss)}}</td>
      <td class="td-mono">${{fmt(e.finalValLoss)}}</td>
      <td>
        <div class="bar-cell">
          <span class="td-mono">${{pct}}%</span>
          <div class="mini-bar">
            <div class="mini-fill" style="width:${{pct}}%;background:${{col}}"></div>
          </div>
        </div>
      </td>
      <td>${{isBest ? '<span class="badge-best">★ Best</span>' : ''}}</td>
    </tr>`;
  }});

  $('tbl-body').innerHTML = rows.join('');
}}

// ── Curve charts ───────────────────────────────────────────────────
function buildCurve(id, lgId, key, yLabel) {{
  const names    = DASH.experimentNames;
  const datasets = names.map(n => ({{
    label: n,
    data:  DASH.experiments[n]?.[key] ?? [],
    borderColor:     DASH.colors[n],
    backgroundColor: hex2rgba(DASH.colors[n], 0.06),
    borderWidth: 2, tension: 0.35,
    pointRadius: 2.5, pointHoverRadius: 5,
    fill: false,
  }}));

  new Chart($(id), {{
    type: 'line',
    data: {{ datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      parsing: {{ xAxisKey: 'x', yAxisKey: 'y' }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{ label: c => ` ${{c.dataset.label}}: ${{fmt(c.raw.y)}}` }}
        }},
      }},
      scales: {{
        x: {{ type: 'linear', grid: GRID, ticks: TICK,
              title: {{ display: true, text: 'Epoch', color: '#94a3b8', font: {{ size: 11 }} }} }},
        y: {{ grid: GRID, ticks: TICK,
              title: {{ display: true, text: yLabel, color: '#94a3b8', font: {{ size: 11 }} }} }},
      }},
    }},
  }});

  makeLegend(lgId, names.map(n => ({{ c: DASH.colors[n], l: n }})));
}}

// ── Radar chart ────────────────────────────────────────────────────
function buildRadar() {{
  const cm    = DASH.contentMetrics;
  const all   = ['base', ...DASH.experimentNames].filter(n => n in cm);
  const KEYS  = ['total_words','lexical_diversity','specificity','avg_point_words','followup_is_q'];
  const LBLS  = ['Total Words','Lex Diversity','Specificity','Pt Detail','FU Valid'];

  const allVals = KEYS.map(k => all.map(n => cm[n]?.[k] ?? 0));
  const mins    = allVals.map(v => Math.min(...v));
  const maxs    = allVals.map(v => Math.max(...v));
  const norm    = n => KEYS.map((k,i) => {{
    const range = maxs[i] - mins[i];
    return range === 0 ? 0.5 : ((cm[n]?.[k] ?? 0) - mins[i]) / range;
  }});

  const datasets = all.map(n => ({{
    label: n,
    data:  norm(n),
    borderColor:     DASH.colors[n] || '#94a3b8',
    backgroundColor: hex2rgba(DASH.colors[n] || '#94a3b8', 0.07),
    borderWidth: 1.5, pointRadius: 2.5,
  }}));

  new Chart($('c-radar'), {{
    type: 'radar',
    data: {{ labels: LBLS, datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        r: {{
          min: 0, max: 1,
          ticks: {{ display: false }},
          grid: {{ color: '#e2e8f0' }},
          pointLabels: {{ font: {{ size: 11 }}, color: '#64748b' }},
        }},
      }},
    }},
  }});

  makeLegend('lg-radar', all.map(n => ({{ c: DASH.colors[n] || '#94a3b8', l: n }})));
}}

// ── Horizontal bar ─────────────────────────────────────────────────
function buildHBar(id, metricKey, label) {{
  const cm    = DASH.contentMetrics;
  const names = ['base', ...DASH.experimentNames].filter(n => n in cm);
  const data  = names.map(n => +(cm[n]?.[metricKey] ?? 0).toFixed(3));
  const cols  = names.map(n => DASH.colors[n] || '#94a3b8');

  new Chart($(id), {{
    type: 'bar',
    data: {{
      labels: names,
      datasets: [{{
        label,
        data,
        backgroundColor: cols.map(c => hex2rgba(c, 0.8)),
        borderColor: cols,
        borderWidth: 1.5, borderRadius: 5, borderSkipped: false,
      }}],
    }},
    options: {{
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }},
                  tooltip: {{ callbacks: {{ label: c => ` ${{c.raw}}` }} }} }},
      scales: {{
        x: {{ grid: GRID, ticks: TICK, beginAtZero: true }},
        y: {{ grid: {{ display: false }}, ticks: {{ ...TICK, font: {{ family: "'JetBrains Mono', monospace", size: 11 }} }} }},
      }},
    }},
  }});
}}

// ── Output viewer ──────────────────────────────────────────────────
function buildPromptDropdown() {{
  const sel = $('prompt-sel');
  DASH.prompts.forEach(p => {{
    const o = document.createElement('option');
    o.value = p.id;
    o.textContent = p.id + ' — ' + p.input.slice(0, 65) + (p.input.length > 65 ? '…' : '');
    sel.appendChild(o);
  }});
}}

function hlSections(text) {{
  return text
    .replace(/(Summary:)/g,           '<span class="out-sec">$1</span>')
    .replace(/(Key Points:)/g,        '<span class="out-sec">$1</span>')
    .replace(/(Limitation:)/g,        '<span class="out-sec">$1</span>')
    .replace(/(Follow-up Question:)/g,'<span class="out-sec">$1</span>');
}}

function renderOutputs() {{
  const pid  = $('prompt-sel').value;
  const grid = $('out-grid');
  const ptxt = $('prompt-txt');
  if (!pid) {{ grid.innerHTML = ''; ptxt.classList.remove('visible'); return; }}

  const p   = DASH.prompts.find(x => x.id === pid);
  const idx = DASH.prompts.findIndex(x => x.id === pid);
  ptxt.textContent = p?.input ?? '';
  ptxt.classList.add('visible');

  const all = ['base', ...DASH.experimentNames];
  grid.innerHTML = all.map(name => {{
    const text  = DASH.outputs[name]?.[idx] ?? '(no output)';
    const color = DASH.colors[name] || '#94a3b8';
    return `<div class="out-card">
      <div class="out-hd">
        <span class="out-hd-dot" style="background:${{color}}"></span>
        <span class="out-hd-name">${{name}}</span>
      </div>
      <div class="out-body">${{hlSections(text)}}</div>
    </div>`;
  }}).join('');
}}

// ── Before / After chat comparison ────────────────────────────────
function pickCompareModels() {{
  // Tier 0: base (no adaptation)
  // Tier 1: lowest LoRA rank available → "low capacity" demo
  // Tier 2: highest compliance / lowest val loss → "trained well" demo
  const names = DASH.experimentNames;
  if (!names.length) return ['base', 'base', 'base'];
  const exps  = DASH.experiments;
  const comp  = DASH.compliance;

  // Prefer specific known names for clarity, fall back to data-driven picks
  const PREFER_LOW  = ['rank_8', 'lr_1e-4'];
  const PREFER_BEST = ['rank_64', 'epochs_5', 'rank_32', 'baseline'];

  const lowPick  = PREFER_LOW.find(n  => names.includes(n))
    || names.slice().sort((a,b) =>
        (exps[a]?.meta?.lora_r ?? 999) - (exps[b]?.meta?.lora_r ?? 999))[0];

  const bestPick = PREFER_BEST.find(n => names.includes(n))
    || (() => {{
      // sort by compliance desc, then val loss asc
      const byComp = names.filter(n => comp[n]).sort((a,b) => comp[b].pct - comp[a].pct);
      if (byComp.length) return byComp[0];
      return names.slice().sort((a,b) =>
        (exps[a]?.finalValLoss ?? 9) - (exps[b]?.finalValLoss ?? 9))[0];
    }})();

  return ['base', lowPick, bestPick];
}}

function chatHlSections(text) {{
  return text
    .replace(/(Summary:)/g,           '<span class="chat-sec">$1</span>')
    .replace(/(Key Points:)/g,        '<span class="chat-sec">$1</span>')
    .replace(/(Limitation:)/g,        '<span class="chat-sec">$1</span>')
    .replace(/(Follow-up Question:)/g,'<span class="chat-sec">$1</span>');
}}

function compBadge(name) {{
  const c = DASH.compliance[name];
  if (!c) return '';
  const col = c.pct >= 80 ? '#10b981' : c.pct >= 50 ? '#f59e0b' : '#ef4444';
  const bg  = col + '18';
  return `<span class="chat-badge" style="background:${{bg}};color:${{col}}">${{c.pct}}% compliant</span>`;
}}

function modelLabel(name) {{
  if (name === 'base') return {{ title: 'No Adaptation', sub: 'Qwen2.5-1.5B-Instruct · base' }};
  const meta = DASH.experiments[name]?.meta || {{}};
  const r    = meta.lora_r    ? `r=${{meta.lora_r}}`       : '';
  const lr   = meta.learning_rate ? `lr=${{meta.learning_rate.toExponential(0)}}` : '';
  const ep   = meta.epochs    ? `${{meta.epochs}} ep`      : '';
  const sub  = [r, lr, ep].filter(Boolean).join(' · ');
  return {{ title: name, sub }};
}}

function buildChatWin(name, promptText, outputText, tier) {{
  const color  = DASH.colors[name] || '#94a3b8';
  const lbl    = modelLabel(name);
  const badge  = compBadge(name);

  const tierTitle = tier === 0 ? 'No Adaptation'
                  : tier === 1 ? 'LoRA — Low Rank'
                  :              'LoRA — High Rank';
  const tierSub   = tier === 0 ? lbl.sub
                  : `${{lbl.sub}}`;
  const initials  = tier === 0 ? 'B'
                  : tier === 1 ? 'Lo'
                  :               'Hi';

  const tierLabel = tier === 0 ? 'No Adaptation (Base)'
                  : tier === 1 ? 'LoRA — low rank / low capacity'
                  :              'LoRA — higher rank / trained';

  return `<div class="chat-win">
    <div class="chat-header">
      <div class="chat-avatar" style="background:${{color}}">${{initials}}</div>
      <div>
        <div class="chat-model-name">${{tierTitle}}</div>
        <div class="chat-model-sub">${{tierSub}}</div>
      </div>
      ${{badge}}
    </div>
    <div class="chat-messages">
      <div class="chat-msg user">
        <div class="msg-icon user-icon">
          <svg viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="7" cy="4.5" r="2.5" stroke="#94a3b8" stroke-width="1.2"/>
            <path d="M2 12c0-2.761 2.239-5 5-5s5 2.239 5 5" stroke="#94a3b8" stroke-width="1.2" stroke-linecap="round"/>
          </svg>
        </div>
        <div class="msg-bubble">${{promptText}}</div>
      </div>
      <div class="chat-msg assistant">
        <div class="msg-icon" style="background:${{color}}20; color:${{color}}; font-size:9px; font-weight:700;">AI</div>
        <div class="msg-bubble">${{chatHlSections(outputText || '(no output generated)')}}</div>
      </div>
    </div>
    <div class="chat-footer">
      <div class="chat-footer-input">Ask a follow-up…</div>
      <div class="send-btn">
        <svg viewBox="0 0 12 12" fill="none">
          <path d="M1 6h10M7 2l4 4-4 4" stroke="white" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
    </div>
  </div>`;
}}

function buildCompareDropdown() {{
  const sel = $('compare-sel');
  DASH.prompts.forEach(p => {{
    const o = document.createElement('option');
    o.value = p.id;
    o.textContent = p.id + ' — ' + p.input.slice(0, 70) + (p.input.length > 70 ? '…' : '');
    sel.appendChild(o);
  }});
}}

function renderCompare() {{
  const pid  = $('compare-sel').value;
  const cols = $('chat-cols');
  if (!pid) {{ cols.innerHTML = ''; return; }}

  const p   = DASH.prompts.find(x => x.id === pid);
  const idx = DASH.prompts.findIndex(x => x.id === pid);

  const [baseKey, midKey, bestKey] = pickCompareModels();
  const trio = [baseKey, midKey, bestKey];

  cols.innerHTML = trio.map((name, tier) => {{
    const text = DASH.outputs[name]?.[idx] ?? '';
    return buildChatWin(name, p.input, text, tier);
  }}).join('');
}}

// ── Init ───────────────────────────────────────────────────────────
(function init() {{
  buildKPIs();
  buildCompliance();
  buildLoss();
  buildTable();
  buildCurve('c-train', 'lg-train', 'trainCurve', 'Loss');
  buildCurve('c-val',   'lg-val',   'valCurve',   'Loss');

  const noMetrics = !DASH.hasContent || Object.keys(DASH.contentMetrics).length === 0;
  const noOutputs = !DASH.hasContent || Object.keys(DASH.outputs).length === 0;

  if (noMetrics) {{
    $('content-empty').style.display  = 'block';
    $('content-charts').style.display = 'none';
  }} else {{
    buildRadar();
    buildHBar('c-ptwords', 'avg_point_words', 'Avg words / key point');
    buildHBar('c-specif',  'specificity',     'Specificity');
  }}

  if (noOutputs) {{
    $('outputs-empty').style.display  = 'block';
    $('outputs-ui').style.display     = 'none';
    $('compare-empty').style.display  = 'block';
    $('compare-ui').style.display     = 'none';
  }} else {{
    buildPromptDropdown();
    buildCompareDropdown();
  }}
}})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LoRA experiment dashboard.")
    parser.add_argument("--no-open", action="store_true", help="Don't open browser after generating.")
    args = parser.parse_args()

    print("Loading experiment data...")
    data = load_all_data()
    n_exp = len(data["experiments"])

    if n_exp == 0:
        print("No trained experiments found in outputs/experiments/")
        print("Run `python src/run_experiments.py` first.")
        sys.exit(1)

    print(f"Found {n_exp} experiments: {', '.join(data['experiments'].keys())}")
    print(f"Content outputs: {'yes' if data['has_content'] else 'no — run eval_content.py for full analysis'}")

    html = generate_html(data)
    DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    print(f"\nDashboard → {DASHBOARD_PATH}")

    if not args.no_open:
        url = DASHBOARD_PATH.resolve().as_uri()
        webbrowser.open(url)


if __name__ == "__main__":
    main()
