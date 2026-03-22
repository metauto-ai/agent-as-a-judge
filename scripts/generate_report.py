"""
Agent-as-a-Judge: Full 55-Task Multi-Framework Evaluation Report Generator
Reads existing judgment results for all DevAI tasks across all frameworks
(AaaJ gray-box + Human) and produces a rich, interactive HTML report.
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── Configuration ─────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark"
JUDGMENT_BASE = BENCHMARK_DIR / "judgment"
AAAJ_SETTING  = "agent_as_a_judge/gray_box"
HUMAN_SETTING = "human_as_a_judge"
FRAMEWORKS    = ["OpenHands", "MetaGPT", "GPT-Pilot"]
ALL_TASKS     = list(range(1, 56))           # tasks 01-55

OUTPUT_DIR  = Path(__file__).parent.parent / "reports"
OUTPUT_FILE = OUTPUT_DIR / "aaaj_full_report.html"
OUTPUT_DIR.mkdir(exist_ok=True)

FW_COLORS = {"OpenHands": "#6366f1", "MetaGPT": "#10b981", "GPT-Pilot": "#f59e0b"}
FW_BG     = {
    "OpenHands": "rgba(99,102,241,.13)",
    "MetaGPT":   "rgba(16,185,129,.13)",
    "GPT-Pilot": "rgba(245,158,11,.13)",
}

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_judgment(framework, setting, task_num):
    jdir = JUDGMENT_BASE / framework / setting
    if not jdir.exists():
        return None
    for f in sorted(jdir.glob("*.json")):
        try:
            if int(f.stem.split("_")[0]) == task_num:
                return json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, json.JSONDecodeError):
            continue
    return None


def score(data):
    reqs  = data.get("requirements", [])
    prefs = data.get("preferences",  [])
    total = len(reqs)
    sat   = sum(1 for r in reqs if r.get("satisfied") is True)
    pct   = round(sat / total * 100, 1) if total else 0.0

    by_cat = defaultdict(lambda: {"sat": 0, "tot": 0})
    for r in reqs:
        c = (r.get("category") or "Other").strip()
        by_cat[c]["tot"] += 1
        if r.get("satisfied") is True:
            by_cat[c]["sat"] += 1

    return {
        "sat": sat, "tot": total, "pct": pct,
        "by_cat": {k: dict(v) for k, v in by_cat.items()},
        "pref_sat": sum(1 for p in prefs if p.get("satisfied") is True),
        "pref_tot": len(prefs),
        "reqs":  [{"idx": r.get("requirement_id", i), "text": r.get("criteria", ""),
                   "cat": (r.get("category") or "").strip(), "ok": r.get("satisfied"),
                   "pre": r.get("prerequisites", [])}
                  for i, r in enumerate(reqs)],
        "prefs": [{"idx": p.get("preference_id", i), "text": p.get("criteria", ""),
                   "ok": p.get("satisfied")}
                  for i, p in enumerate(prefs)],
    }


# ── Build full dataset ────────────────────────────────────────────────────────

tasks = {}
for t in ALL_TASKS:
    info = {"name": None, "query": None, "tags": [], "aaaj": {}, "human": {}}
    for fw in FRAMEWORKS:
        for key, setting in [("aaaj", AAAJ_SETTING), ("human", HUMAN_SETTING)]:
            d = load_judgment(fw, setting, t)
            if not d:
                continue
            if info["name"] is None:
                info["name"]  = d.get("name", f"Task {t:02d}")
                info["query"] = d.get("query", "")
                info["tags"]  = d.get("tags", [])
            info[key][fw] = score(d)
    tasks[t] = info

# ── Aggregate helpers ─────────────────────────────────────────────────────────

def fw_aggregate(fw, source="aaaj"):
    pcts, sat_tot, req_tot, fully = [], 0, 0, 0
    for t in ALL_TASKS:
        sc = tasks[t][source].get(fw)
        if not sc:
            continue
        pcts.append(sc["pct"])
        sat_tot += sc["sat"]
        req_tot += sc["tot"]
        if sc["sat"] == sc["tot"] and sc["tot"] > 0:
            fully += 1
    avg = round(sum(pcts) / len(pcts), 1) if pcts else 0.0
    ovr = round(sat_tot / req_tot * 100, 1) if req_tot else 0.0
    return {"avg": avg, "ovr": ovr, "sat": sat_tot, "tot": req_tot,
            "fully": fully, "n": len(pcts)}


def alignment(fw):
    match = tot = 0
    for t in ALL_TASKS:
        a = tasks[t]["aaaj"].get(fw)
        h = tasks[t]["human"].get(fw)
        if not a or not h:
            continue
        for ra, rh in zip(a["reqs"], h["reqs"]):
            if ra["ok"] is not None and rh["ok"] is not None:
                tot += 1
                if ra["ok"] == rh["ok"]:
                    match += 1
    return {"match": match, "tot": tot,
            "pct": round(match / tot * 100, 1) if tot else 0.0}


def score_color(pct):
    if pct >= 80: return "#10b981"
    if pct >= 50: return "#f59e0b"
    if pct >= 20: return "#f97316"
    return "#ef4444"


# ── HTML helpers ──────────────────────────────────────────────────────────────

def badge(ok):
    if ok is True:  return '<span class="b pass">&#10003;</span>'
    if ok is False: return '<span class="b fail">&#10007;</span>'
    return '<span class="b unk">?</span>'


def bar(pct, color):
    return (f'<div class="barw"><div class="barf" style="width:{pct}%;background:{color}">'
            f'</div><span class="barl">{pct}%</span></div>')


def hm_cell(pct):
    if pct is None:
        return '<td class="hm-na">&#8212;</td>'
    c = score_color(pct)
    return f'<td class="hm-cell" style="background:{c}22;color:{c}">{pct}%</td>'


# ── Framework summary cards ───────────────────────────────────────────────────

def fw_summary_cards():
    html = '<div class="fw-sum-grid">'
    for fw in FRAMEWORKS:
        agg  = fw_aggregate(fw, "aaaj")
        hagg = fw_aggregate(fw, "human")
        aln  = alignment(fw)
        c    = FW_COLORS[fw]
        html += f"""
        <div class="fw-sum-card" style="border-top:5px solid {c}">
          <h3 style="color:{c}">{fw}</h3>
          <div class="fw-kpi-row">
            <div class="fw-kpi"><div class="fw-big">{agg["ovr"]}%</div><div class="fw-lbl">AaaJ Overall</div></div>
            <div class="fw-kpi"><div class="fw-big" style="color:#94a3b8">{hagg["ovr"]}%</div><div class="fw-lbl">Human Overall</div></div>
          </div>
          <hr class="fhr"/>
          <div class="fw-row"><span>Tasks covered</span><b>{agg["n"]}/55</b></div>
          <div class="fw-row"><span>Reqs satisfied (AaaJ)</span><b>{agg["sat"]}/{agg["tot"]}</b></div>
          <div class="fw-row"><span>Fully solved tasks</span><b>{agg["fully"]}</b></div>
          <div class="fw-row"><span>AaaJ&#8211;Human alignment</span>
            <b style="color:{score_color(aln["pct"])}">{aln["pct"]}%</b></div>
        </div>"""
    html += '</div>'
    return html


# ── Heatmap table ─────────────────────────────────────────────────────────────

def heatmap_table():
    hdr = "<tr><th>#</th><th>Task</th>"
    for fw in FRAMEWORKS:
        hdr += f'<th style="color:{FW_COLORS[fw]}">AaaJ&#xB7;{fw[:4]}</th>'
    for fw in FRAMEWORKS:
        hdr += f'<th style="color:#94a3b8">Human&#xB7;{fw[:4]}</th>'
    hdr += "</tr>"

    rows = ""
    for t in ALL_TASKS:
        info  = tasks[t]
        name  = (info["name"] or f"Task {t:02d}").replace("_", " ")
        cells = (f'<td class="task-id">{t:02d}</td>'
                 f'<td><a href="#t{t:02d}">{name[:52]}</a></td>')
        for source in ["aaaj", "human"]:
            for fw in FRAMEWORKS:
                sc = info[source].get(fw)
                cells += hm_cell(sc["pct"] if sc else None)
        rows += f"<tr>{cells}</tr>"

    avg_cells = "<td></td><td><strong>Average</strong></td>"
    for source in ["aaaj", "human"]:
        for fw in FRAMEWORKS:
            vals = [tasks[t][source][fw]["pct"] for t in ALL_TASKS
                    if fw in tasks[t][source]]
            avg  = round(sum(vals) / len(vals), 1) if vals else 0.0
            c    = score_color(avg)
            avg_cells += f'<td style="color:{c};font-weight:800">{avg}%</td>'
    rows += f"<tr class=\"avg-r\">{avg_cells}</tr>"

    return (f'<div class="hm-scroll"><table class="hm-tbl" id="hm-tbl">'
            f'<thead>{hdr}</thead><tbody>{rows}</tbody></table></div>')


# ── Per-task card ─────────────────────────────────────────────────────────────

def task_card(t):
    info  = tasks[t]
    name  = (info["name"] or f"Task {t:02d}").replace("_", " ")
    query = info["query"] or ""
    qshow = query[:300] + ("..." if len(query) > 300 else "")
    tags  = "".join(f'<span class="tag">{tg}</span>' for tg in info.get("tags", []))

    cards = ""
    for fw in FRAMEWORKS:
        sc  = info["aaaj"].get(fw)
        hsc = info["human"].get(fw)
        c   = FW_COLORS[fw]
        if not sc:
            cards += (f'<div class="fwc" style="border-top:4px solid #444">'
                      f'<h4 style="color:#777">{fw}</h4>'
                      f'<p style="color:var(--muted);font-size:.8rem">No data</p></div>')
            continue
        hline = ""
        if hsc:
            diff  = abs(hsc["pct"] - sc["pct"])
            sym   = "&#8776;" if diff <= 5 else ("&#9650;" if hsc["pct"] > sc["pct"] else "&#9660;")
            hline = (f'<div class="hline">&#128100; Human: <b style="color:#94a3b8">'
                     f'{hsc["pct"]}%</b> <span class="aln">{sym}</span></div>')
        cards += f"""
        <div class="fwc" style="border-top:4px solid {c};background:{FW_BG[fw]}">
          <h4 style="color:{c}">{fw}</h4>
          {bar(sc["pct"], c)}
          <div class="fwmeta"><span>&#10003; {sc["sat"]}/{sc["tot"]} reqs</span></div>
          {hline}
        </div>"""

    tables = ""
    for fw in FRAMEWORKS:
        a = info["aaaj"].get(fw)
        h = info["human"].get(fw)
        if not a:
            continue
        c = FW_COLORS[fw]
        rows = ""
        for req in a["reqs"]:
            h_ok = ""
            if h:
                hreq = next((r for r in h["reqs"] if r["idx"] == req["idx"]), None)
                if hreq:
                    h_ok = badge(hreq["ok"])
            rows += (f'<tr>'
                     f'<td class="ri" style="border-left:3px solid {c}">{req["idx"]}</td>'
                     f'<td class="rc">{req["text"]}</td>'
                     f'<td class="rcat">{req["cat"]}</td>'
                     f'<td>{badge(req["ok"])}</td>'
                     f'<td>{h_ok}</td>'
                     f'</tr>')
        tables += f"""
        <details class="rdet">
          <summary style="color:{c}">&#128203; {fw} &#8212; {a["sat"]}/{a["tot"]} satisfied (AaaJ)</summary>
          <table class="rt">
            <thead><tr><th>#</th><th>Criteria</th><th>Category</th><th>AaaJ</th><th>Human</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </details>"""

    return f"""
    <section class="tc" id="t{t:02d}">
      <div class="th">
        <span class="tn">Task {t:02d}</span>
        <h2 class="tt">{name}</h2>
        <div class="tags">{tags}</div>
      </div>
      <p class="tq">{qshow}</p>
      <div class="fwcs">{cards}</div>
      {tables}
    </section>"""


# ── Chart.js data helpers ─────────────────────────────────────────────────────

def js_per_task_data(fw):
    vals = [tasks[t]["aaaj"].get(fw, {}).get("pct", 0) for t in ALL_TASKS]
    return json.dumps(vals)


def js_aggregate_data():
    aaaj_vals  = [fw_aggregate(fw, "aaaj")["ovr"] for fw in FRAMEWORKS]
    human_vals = [fw_aggregate(fw, "human")["ovr"] for fw in FRAMEWORKS]
    ds = [
        {"label": "AaaJ (Gray-Box)",
         "data": aaaj_vals,
         "backgroundColor": [FW_COLORS[fw] + "cc" for fw in FRAMEWORKS],
         "borderColor": [FW_COLORS[fw] for fw in FRAMEWORKS],
         "borderWidth": 2, "borderRadius": 8},
        {"label": "Human",
         "data": human_vals,
         "backgroundColor": ["#94a3b855"] * 3,
         "borderColor": ["#94a3b8"] * 3,
         "borderWidth": 2, "borderRadius": 8},
    ]
    return json.dumps({"labels": FRAMEWORKS, "datasets": ds})


def js_alignment_data():
    vals = [alignment(fw)["pct"] for fw in FRAMEWORKS]
    ds = [{"label": "Alignment %",
           "data": vals,
           "backgroundColor": [FW_COLORS[fw] + "cc" for fw in FRAMEWORKS],
           "borderColor": [FW_COLORS[fw] for fw in FRAMEWORKS],
           "borderWidth": 2, "borderRadius": 8}]
    return json.dumps({"labels": FRAMEWORKS, "datasets": ds})


# ── Build HTML ────────────────────────────────────────────────────────────────

now = datetime.now().strftime("%Y-%m-%d %H:%M")

total_reqs = sum(tasks[t]["aaaj"].get(fw, {}).get("tot", 0)
                 for t in ALL_TASKS for fw in FRAMEWORKS)
total_sat  = sum(tasks[t]["aaaj"].get(fw, {}).get("sat", 0)
                 for t in ALL_TASKS for fw in FRAMEWORKS)
grand_pct  = round(total_sat / total_reqs * 100, 1) if total_reqs else 0.0

all_task_sections = "".join(task_card(t) for t in ALL_TASKS)
agg_data  = js_aggregate_data()
aln_data  = js_alignment_data()
task_labels_js = json.dumps([f"T{t:02d}" for t in ALL_TASKS])
oh_data   = js_per_task_data("OpenHands")
mg_data   = js_per_task_data("MetaGPT")
gp_data   = js_per_task_data("GPT-Pilot")

fw_labels_js    = json.dumps(FRAMEWORKS)
fw_colors_cc_js = json.dumps([FW_COLORS[fw] + "cc" for fw in FRAMEWORKS])
fw_colors_js    = json.dumps([FW_COLORS[fw] for fw in FRAMEWORKS])
avg_aaaj_js     = json.dumps([fw_aggregate(fw, "aaaj")["avg"] for fw in FRAMEWORKS])

CSS = """
:root {
  --bg:#0f1117; --surface:#1a1d27; --surface2:#222536;
  --border:#2e3149; --text:#e2e8f0; --muted:#8892a4; --accent:#6366f1;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);line-height:1.6}
.hero{background:linear-gradient(135deg,#1a1d27,#0f1117 55%,#1a1040);
  border-bottom:1px solid var(--border);padding:52px 40px 40px;text-align:center}
.hero h1{font-size:2.6rem;font-weight:800;background:linear-gradient(135deg,#a5b4fc,#818cf8,#6ee7b7);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px}
.hero p{color:var(--muted);font-size:.95rem;margin-bottom:16px}
.pills{display:flex;gap:10px;justify-content:center;flex-wrap:wrap}
.pill{padding:4px 14px;border-radius:20px;font-size:.78rem;font-weight:600;
  border:1px solid var(--border);background:var(--surface2)}
.kpi-row{display:flex;gap:24px;justify-content:center;margin-top:24px;flex-wrap:wrap}
.kpi{text-align:center}
.kpi-val{font-size:2rem;font-weight:800;color:#a5b4fc}
.kpi-lbl{font-size:.75rem;color:var(--muted);font-weight:500}
.container{max-width:1200px;margin:0 auto;padding:40px 24px}
.section-title{font-size:1.25rem;font-weight:800;margin:44px 0 18px;
  padding-bottom:10px;border-bottom:2px solid var(--border)}
.fw-sum-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px;margin-bottom:8px}
.fw-sum-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:22px}
.fw-sum-card h3{font-size:1.1rem;font-weight:800;margin-bottom:14px}
.fw-kpi-row{display:flex;gap:16px;margin-bottom:12px}
.fw-kpi{flex:1;text-align:center;background:var(--surface2);border-radius:8px;padding:10px 4px}
.fw-big{font-size:1.7rem;font-weight:800}
.fw-lbl{font-size:.7rem;color:var(--muted);font-weight:600;margin-top:2px}
.fhr{border:none;border-top:1px solid var(--border);margin:12px 0}
.fw-row{display:flex;justify-content:space-between;font-size:.82rem;padding:4px 0;color:var(--muted)}
.fw-row b{color:var(--text);font-weight:700}
.search-wrap{margin-bottom:14px}
#tbl-search{width:100%;padding:10px 14px;background:var(--surface);border:1px solid var(--border);
  border-radius:8px;color:var(--text);font-size:.88rem;outline:none}
#tbl-search:focus{border-color:var(--accent)}
.hm-scroll{overflow-x:auto;border-radius:12px;border:1px solid var(--border)}
.hm-tbl{width:100%;border-collapse:collapse;font-size:.8rem;white-space:nowrap}
.hm-tbl th{padding:10px 12px;background:var(--surface2);color:var(--muted);font-weight:700;
  position:sticky;top:0;z-index:2;text-align:center}
.hm-tbl th:nth-child(1),.hm-tbl th:nth-child(2){text-align:left}
.hm-tbl td{padding:8px 10px;border-bottom:1px solid var(--border);text-align:center}
.hm-tbl td:nth-child(2){text-align:left;max-width:260px;overflow:hidden;text-overflow:ellipsis}
.hm-tbl tr:hover td{background:var(--surface2)}
.task-id{font-weight:700;color:var(--muted);width:40px}
.hm-cell{font-weight:700;font-size:.82rem;border-radius:4px}
.hm-na{color:var(--muted)}
.avg-r td{background:var(--surface2)!important;border-top:2px solid var(--border);font-size:.82rem}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.charts-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;margin-bottom:32px}
@media(max-width:900px){.charts-grid{grid-template-columns:1fr}}
.chart-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px}
.chart-card h3{font-size:.95rem;font-weight:700;margin-bottom:14px;color:var(--muted)}
.chart-big{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:20px;margin-bottom:32px;overflow-x:auto}
.chart-big h3{font-size:.95rem;font-weight:700;margin-bottom:14px;color:var(--muted)}
.chart-big-inner{min-width:1400px}
.tc{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  margin-bottom:24px;overflow:hidden}
.th{display:flex;align-items:center;gap:12px;padding:18px 22px 12px;
  border-bottom:1px solid var(--border);flex-wrap:wrap}
.tn{font-size:.72rem;font-weight:700;background:var(--accent);color:#fff;
  padding:3px 10px;border-radius:6px;letter-spacing:.05em;white-space:nowrap}
.tt{font-size:1rem;font-weight:700}
.tags{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto}
.tag{font-size:.7rem;padding:2px 8px;border-radius:10px;background:var(--surface2);
  color:var(--muted);border:1px solid var(--border)}
.tq{padding:10px 22px;font-size:.83rem;color:var(--muted);border-bottom:1px solid var(--border);font-style:italic}
.fwcs{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;padding:18px 22px}
.fwc{border-radius:10px;padding:14px;border:1px solid var(--border)}
.fwc h4{font-size:.9rem;font-weight:700;margin-bottom:8px}
.fwmeta{display:flex;gap:8px;margin-top:6px;font-size:.76rem;color:var(--muted)}
.hline{font-size:.76rem;color:var(--muted);margin-top:4px}
.aln{font-weight:700}
.barw{position:relative;height:20px;background:var(--surface2);border-radius:10px;overflow:hidden;margin-bottom:4px}
.barf{height:100%;border-radius:10px}
.barl{position:absolute;right:7px;top:2px;font-size:.75rem;font-weight:700;color:#fff}
.rdet{padding:0 22px 14px}
.rdet summary{cursor:pointer;padding:9px 0;font-size:.85rem;font-weight:600}
.rt{width:100%;border-collapse:collapse;margin-top:8px;font-size:.79rem}
.rt th{text-align:left;padding:7px 9px;background:var(--surface2);color:var(--muted);
  font-weight:600;border-bottom:1px solid var(--border)}
.rt td{padding:7px 9px;border-bottom:1px solid var(--border);vertical-align:top}
.ri{font-weight:700;width:36px}
.rc{width:38%}
.rcat{font-size:.73rem;color:var(--muted);width:22%}
.b{display:inline-block;padding:1px 7px;border-radius:4px;font-size:.73rem;font-weight:700}
.pass{background:rgba(16,185,129,.2);color:#34d399}
.fail{background:rgba(239,68,68,.2);color:#f87171}
.unk{background:var(--surface2);color:var(--muted)}
footer{text-align:center;padding:28px;color:var(--muted);font-size:.78rem;border-top:1px solid var(--border)}
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>AaaJ Full Evaluation Report &#8212; 55 Tasks</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>{CSS}</style>
</head>
<body>

<div class="hero">
  <h1>&#129302; Agent-as-a-Judge</h1>
  <p>Full Evaluation Report &#8212; 55 DevAI Tasks &#215; 3 Frameworks</p>
  <div class="pills">
    <span class="pill">&#128197; {now}</span>
    <span class="pill">&#128194; 55 Tasks</span>
    <span class="pill">&#129302; OpenHands &middot; MetaGPT &middot; GPT-Pilot</span>
    <span class="pill">&#9881;&#65039; AaaJ Gray-Box vs Human</span>
  </div>
  <div class="kpi-row">
    <div class="kpi"><div class="kpi-val">55</div><div class="kpi-lbl">Tasks</div></div>
    <div class="kpi"><div class="kpi-val">3</div><div class="kpi-lbl">Frameworks</div></div>
    <div class="kpi"><div class="kpi-val">{total_reqs}</div><div class="kpi-lbl">Total Req Checks (AaaJ)</div></div>
    <div class="kpi"><div class="kpi-val">{grand_pct}%</div><div class="kpi-lbl">Overall AaaJ Satisfaction</div></div>
  </div>
</div>

<div class="container">

  <h2 class="section-title">&#127942; Framework Overview</h2>
  {fw_summary_cards()}

  <h2 class="section-title">&#128202; Aggregate Performance</h2>
  <div class="charts-grid">
    <div class="chart-card">
      <h3>Overall Requirement Satisfaction</h3>
      <canvas id="aggChart" height="220"></canvas>
    </div>
    <div class="chart-card">
      <h3>AaaJ &#8211; Human Alignment Rate</h3>
      <canvas id="alnChart" height="220"></canvas>
    </div>
    <div class="chart-card">
      <h3>Average Score per Framework (AaaJ)</h3>
      <canvas id="avgChart" height="220"></canvas>
    </div>
  </div>

  <h2 class="section-title">&#128200; Score per Task &#8212; All 55 Tasks (AaaJ Gray-Box)</h2>
  <div class="chart-big">
    <h3>Grouped bar chart &#8212; scroll right to see all tasks</h3>
    <div class="chart-big-inner">
      <canvas id="taskChart" height="260"></canvas>
    </div>
  </div>

  <h2 class="section-title">&#128293; Score Heatmap &#8212; All 55 Tasks</h2>
  <div class="search-wrap">
    <input id="tbl-search" type="text" placeholder="&#128269;  Filter tasks by name..."/>
  </div>
  {heatmap_table()}

  <h2 class="section-title">&#128269; Task-by-Task Breakdown</h2>
  {all_task_sections}

</div>

<footer>Agent-as-a-Judge Evaluation Report &middot; 55 DevAI Tasks &middot; Gray-Box (AaaJ) vs Human &middot; Generated {now}</footer>

<script>
const darkGrid = {{ color: '#2e3149' }};
const darkTick = {{ color: '#8892a4' }};
const legendCfg = {{ labels: {{ color: '#e2e8f0', font: {{ size: 12 }} }} }};

const aggData = {agg_data};
new Chart(document.getElementById('aggChart'), {{
  type: 'bar', data: aggData,
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: legendCfg }},
    scales: {{
      x: {{ ticks: darkTick, grid: darkGrid }},
      y: {{ min:0, max:100, ticks: {{...darkTick, callback: v => v + '%'}}, grid: darkGrid }}
    }}
  }}
}});

const alnData = {aln_data};
new Chart(document.getElementById('alnChart'), {{
  type: 'bar', data: alnData,
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: darkTick, grid: darkGrid }},
      y: {{ min:0, max:100, ticks: {{...darkTick, callback: v => v + '%'}}, grid: darkGrid }}
    }}
  }}
}});

new Chart(document.getElementById('avgChart'), {{
  type: 'bar',
  data: {{
    labels: {fw_labels_js},
    datasets: [{{
      label: 'Avg task score (AaaJ)',
      data: {avg_aaaj_js},
      backgroundColor: {fw_colors_cc_js},
      borderColor: {fw_colors_js},
      borderWidth: 2, borderRadius: 8
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: darkTick, grid: darkGrid }},
      y: {{ min:0, max:100, ticks: {{...darkTick, callback: v => v + '%'}}, grid: darkGrid }}
    }}
  }}
}});

new Chart(document.getElementById('taskChart'), {{
  type: 'bar',
  data: {{
    labels: {task_labels_js},
    datasets: [
      {{ label: 'OpenHands', data: {oh_data},
         backgroundColor: '#6366f1bb', borderColor: '#6366f1', borderWidth: 1.5, borderRadius: 3 }},
      {{ label: 'MetaGPT',   data: {mg_data},
         backgroundColor: '#10b981bb', borderColor: '#10b981', borderWidth: 1.5, borderRadius: 3 }},
      {{ label: 'GPT-Pilot', data: {gp_data},
         backgroundColor: '#f59e0bbb', borderColor: '#f59e0b', borderWidth: 1.5, borderRadius: 3 }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: legendCfg }},
    scales: {{
      x: {{ ticks: {{...darkTick, font:{{size:10}}}}, grid: darkGrid }},
      y: {{ min:0, max:100, ticks: {{...darkTick, callback: v => v + '%'}}, grid: darkGrid }}
    }}
  }}
}});

document.getElementById('tbl-search').addEventListener('input', function() {{
  const q = this.value.toLowerCase();
  document.querySelectorAll('#hm-tbl tbody tr').forEach(tr => {{
    const txt = tr.querySelector('td:nth-child(2)');
    if (!txt) return;
    tr.style.display = txt.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}});
</script>
</body>
</html>"""

with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
    fh.write(html)

print(f"Report -> {OUTPUT_FILE}")
print()
print("=" * 78)
print("   AGENT-AS-A-JUDGE  -  Full Evaluation Summary  -  55 DevAI Tasks")
print("=" * 78)
FW_W = 12
header = f"{'Task':<48}" + "".join(f"{fw:>{FW_W}}" for fw in FRAMEWORKS)
print(header)
print("-" * 78)
for t in ALL_TASKS:
    info = tasks[t]
    name = (info["name"] or f"Task {t:02d}").replace("_", " ")[:46]
    row  = f"{name:<48}"
    for fw in FRAMEWORKS:
        sc = info["aaaj"].get(fw)
        row += f"{'N/A':>{FW_W}}" if not sc else f"{sc['pct']:>{FW_W-1}.1f}%"
    print(row)
print("-" * 78)
avgs = {}
for fw in FRAMEWORKS:
    vals = [tasks[t]["aaaj"][fw]["pct"] for t in ALL_TASKS if fw in tasks[t]["aaaj"]]
    avgs[fw] = round(sum(vals) / len(vals), 1) if vals else 0.0
avg_row = f"{'AVERAGE':<48}" + "".join(f"{avgs[fw]:>{FW_W-1}.1f}%" for fw in FRAMEWORKS)
print(avg_row)
print("=" * 78)
print()
print("-- AaaJ vs Human Alignment --")
for fw in FRAMEWORKS:
    aln = alignment(fw)
    print(f"  {fw:<14}  {aln['pct']:5.1f}%  ({aln['match']}/{aln['tot']} requirements)")
print()
