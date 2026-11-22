"""
dashboard/app.py — IoT Anomaly Detection Dashboard
====================================================
Real-time network anomaly detection using CNN-LSTM + LLM analysis.
Run:  streamlit run dashboard/app.py
"""

import streamlit as st
import numpy as np
import torch
import sys, os, time, threading
import traceback
from datetime import datetime
from collections import deque
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "models"))
sys.path.insert(0, os.path.join(BASE_DIR, "simulation"))
sys.path.insert(0, os.path.join(BASE_DIR, "llm"))

from cnn_lstm          import CNNLSTM
from traffic_simulator import generate_normal, generate_ddos, generate_botnet, FEATURES
from rag.generate       import finalize_explanation, stream_explanation
from rag.evaluate_explanations import (
    QUALITY_SCORE_NAMES,
    create_scorer,
    score_explanation,
)

MODEL_PATH = os.path.join(BASE_DIR, "cnn_lstm_component", "models", "best_model.pth")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW     = 10
LABELS     = {0: "Normal", 1: "DDoS", 2: "Botnet"}
COLORS     = {"Normal": "#3fb950", "DDoS": "#f85149", "Botnet": "#a371f7"}
BG_COLORS  = {"Normal": "rgba(63,185,80,0.08)",
              "DDoS":   "rgba(248,81,73,0.12)",
              "Botnet": "rgba(163,113,247,0.12)"}
ICONS      = {"Normal": "🟢", "DDoS": "🔴", "Botnet": "🟣"}

PRESET_SCENARIOS = {
    "Normal → DDoS → Normal": [
        ("normal", "Normal"),
        ("syn_flood", "DDoS"),
        ("normal", "Normal recovery"),
    ],
    "Normal → Botnet": [
        ("normal", "Normal"),
        ("botnet", "Botnet"),
    ],
}
SCENARIO_OPTIONS = (*PRESET_SCENARIOS, "Custom")
CUSTOM_PHASE_TYPES = {
    "Normal": "normal",
    "DDoS": "syn_flood",
    "Botnet": "botnet",
}
EXPLANATION_MODELS = ("llama3.2:3b", "gemma2:2b", "phi3:mini")

THREAT_COLORS = {"safe": "#3fb950", "elevated": "#d29922", "critical": "#f85149"}


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & STYLING
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="IoT Anomaly Shield", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_retriever():
    from rag.retrieve import retrieve
    return retrieve


@st.cache_resource(show_spinner="Loading local explanation quality models...")
def load_quality_scorer():
    try:
        return create_scorer(), None
    except Exception as error:
        print(
            f"Explanation quality scorer could not be loaded: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        return None, f"{type(error).__name__}: {error}"


QUALITY_SCORE_LABELS = {
    "relevance": "Relevance",
    "accuracy": "Accuracy",
    "actionability": "Actionability",
    "consistency": "Consistency",
    "clarity": "Clarity",
}


def render_quality_panel(scores):
    with st.container(border=True):
        st.caption("Explanation quality · automated proxy scores")
        tiles = []
        for score_name in QUALITY_SCORE_NAMES:
            value = scores.get(score_name)
            rendered_value = "n/a" if value is None else f"{value:.0%}"
            tiles.append(
                f'<div class="quality-tile">'
                f'<div class="quality-label">{QUALITY_SCORE_LABELS[score_name]}</div>'
                f'<div class="quality-value">{rendered_value}</div>'
                "</div>"
            )
        st.markdown(
            f'<div class="quality-grid">{"".join(tiles)}</div>',
            unsafe_allow_html=True,
        )


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-0: #010409;
    --bg-1: #0d1117;
    --bg-2: #161b22;
    --bg-3: #21262d;
    --border: #30363d;
    --border-light: rgba(240,246,252,0.1);
    --text-0: #f0f6fc;
    --text-1: #e6edf3;
    --text-2: #8b949e;
    --text-3: #6e7681;
    --green: #3fb950;
    --red: #f85149;
    --purple: #a371f7;
    --blue: #58a6ff;
    --orange: #d29922;
}

* { font-family: 'Inter', -apple-system, sans-serif !important; }
[data-testid="stIconMaterial"] {
    font-family: "Material Symbols Rounded" !important;
}
code, .mono { font-family: 'JetBrains Mono', monospace !important; }

[data-testid="stAppViewContainer"] { background: var(--bg-1); }
[data-testid="stMainBlockContainer"] {
    max-width: 1680px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: var(--bg-2);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-1) !important; }
[data-testid="stButton"] button {
    border-radius: 8px;
    font-weight: 600;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stNumberInput"] input {
    border-radius: 8px;
}

/* ── Status Header ──────────────────────────────────────────────── */
.status-header {
    background: linear-gradient(135deg, var(--bg-2) 0%, #1c2333 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 20px;
    position: relative;
    overflow: hidden;
}
.status-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.status-header.safe::before { background: var(--green); }
.status-header.attack::before { background: var(--red); }
.status-header.botnet::before { background: var(--purple); }

/* ── Pulse Indicator ────────────────────────────────────────────── */
.pulse-dot {
    width: 14px; height: 14px;
    border-radius: 50%;
    position: relative;
}
.pulse-dot::after {
    content: '';
    position: absolute;
    top: -4px; left: -4px;
    width: 22px; height: 22px;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}
.pulse-green { background: var(--green); }
.pulse-green::after { border: 2px solid var(--green); animation-name: pulse-g; }
.pulse-red { background: var(--red); }
.pulse-red::after { border: 2px solid var(--red); animation-name: pulse-r; }
.pulse-purple { background: var(--purple); }
.pulse-purple::after { border: 2px solid var(--purple); animation-name: pulse-p; }
@keyframes pulse-g { 0%,100%{opacity:0;transform:scale(0.8)} 50%{opacity:0.5;transform:scale(1.4)} }
@keyframes pulse-r { 0%,100%{opacity:0;transform:scale(0.8)} 50%{opacity:0.7;transform:scale(1.6)} }
@keyframes pulse-p { 0%,100%{opacity:0;transform:scale(0.8)} 50%{opacity:0.6;transform:scale(1.5)} }

/* ── Metric Cards ───────────────────────────────────────────────── */
.m-card {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 14px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    transition: border-color 0.3s, transform 0.2s;
}
.m-card:hover { border-color: var(--blue); transform: translateY(-2px); }
.m-icon { font-size: 20px; margin-bottom: 4px; }
.m-label { color: var(--text-2); font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
.m-value { color: var(--text-0); font-size: 24px; font-weight: 700; margin: 4px 0 2px; }
.m-bar { background: var(--bg-3); border-radius: 3px; height: 4px; margin: 6px 0 4px; overflow: hidden; }
.m-fill { height: 4px; border-radius: 3px; transition: width 0.5s ease; }
.m-pct { font-size: 11px; font-weight: 500; }

/* ── Probability Strip ──────────────────────────────────────────── */
.prob-strip {
    display: flex;
    height: 10px;
    border-radius: 5px;
    overflow: hidden;
    background: var(--bg-3);
    margin: 0 0 6px;
}
.prob-strip > div { transition: width 0.5s ease; min-width: 2px; }
.prob-labels { display: flex; justify-content: space-between; }
.prob-labels span { font-size: 11px; font-weight: 500; }

/* ── Flow Log ───────────────────────────────────────────────────── */
.flow-log {
    background: var(--bg-0);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px;
    height: 300px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px;
    line-height: 1.7;
}
.flow-log::-webkit-scrollbar { width: 6px; }
.flow-log::-webkit-scrollbar-track { background: transparent; }
.flow-log::-webkit-scrollbar-thumb { background: var(--bg-3); border-radius: 3px; }

/* ── Alert Banner ───────────────────────────────────────────────── */
.threat-banner {
    border-radius: 12px;
    padding: 16px 24px;
    margin: 12px 0;
    display: flex;
    align-items: center;
    gap: 14px;
    animation: threat-glow 2s ease-in-out infinite alternate;
}
.threat-banner.ddos {
    background: linear-gradient(135deg, #2d0f0f 0%, #1a0808 100%);
    border: 1px solid rgba(248,81,73,0.4);
}
.threat-banner.botnet {
    background: linear-gradient(135deg, #1f0d33 0%, #120822 100%);
    border: 1px solid rgba(163,113,247,0.4);
}
@keyframes threat-glow {
    from { box-shadow: 0 0 8px rgba(248,81,73,0.15); }
    to   { box-shadow: 0 0 24px rgba(248,81,73,0.35); }
}

/* ── LLM Cards ──────────────────────────────────────────────────── */
div[data-testid="stExpander"] {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
}
.llm-response {
    background: var(--bg-0);
    border-radius: 8px;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12.5px;
    color: var(--text-1);
    white-space: pre-wrap;
    line-height: 1.9;
    border-left: 3px solid var(--blue);
}
.llm-pending {
    background: var(--bg-0);
    border-radius: 8px;
    padding: 24px 16px;
    text-align: center;
}

/* ── Blink cursor for streaming ──────────────────────────────── */
@keyframes blink { 50% { opacity: 0; } }
.cursor { color: var(--blue); font-weight: 700; animation: blink 1s step-end infinite; }

/* ── Alert Table ────────────────────────────────────────────────── */
.alert-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    background: var(--bg-2);
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
}
.alert-table th {
    padding: 12px 16px;
    text-align: left;
    color: var(--text-2);
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    background: var(--bg-3);
    border-bottom: 1px solid var(--border);
}
.alert-table td {
    padding: 10px 16px;
    font-size: 13px;
    border-bottom: 1px solid var(--border-light);
}
.alert-table tr:last-child td { border-bottom: none; }

/* ── Section Headers ────────────────────────────────────────────── */
.section-hdr {
    color: var(--text-1);
    font-size: 16px;
    font-weight: 600;
    margin: 20px 0 12px;
    display: flex; align-items: center; gap: 8px;
}
.quality-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(88px, 1fr));
    gap: 8px;
}
.quality-tile {
    min-width: 0;
    padding: 8px 10px;
    background: var(--bg-0);
    border: 1px solid var(--border);
    border-radius: 8px;
}
.quality-label {
    color: var(--text-2);
    font-size: 11px;
    line-height: 1.25;
    overflow-wrap: anywhere;
}
.quality-value {
    color: var(--text-0);
    font-size: 20px;
    font-weight: 700;
    line-height: 1.3;
    white-space: nowrap;
}
.chart-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: var(--text-2);
    font-size: 12px;
    font-weight: 600;
    margin: 0 0 8px;
}
.live-pill {
    color: var(--green);
    font-size: 10px;
    letter-spacing: 0.7px;
    border: 1px solid rgba(63,185,80,0.35);
    border-radius: 999px;
    padding: 3px 8px;
    background: rgba(63,185,80,0.08);
}

/* ── Done Banner ────────────────────────────────────────────────── */
.done-banner {
    background: linear-gradient(135deg, #0d2818 0%, #0d1117 100%);
    border: 1px solid rgba(63,185,80,0.3);
    border-radius: 12px;
    padding: 20px 28px;
    text-align: center;
    margin: 16px 0;
}

/* ── Sidebar phase item ─────────────────────────────────────────── */
.phase-item {
    display: flex; align-items: center; gap: 8px;
    padding: 5px 10px; margin: 2px 0;
    border-radius: 6px; font-size: 13px;
    transition: background 0.2s;
}
.phase-item.active { background: rgba(88,166,255,0.1); font-weight: 600; color: white !important; }
.phase-item.done { color: var(--text-3) !important; text-decoration: line-through; }
.phase-item.pending { color: var(--text-2) !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = CNNLSTM(n_features=ckpt.get("n_features", 38),
                    n_classes=ckpt.get("n_classes", 3)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, window):
    x     = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    cls   = int(probs.argmax())
    return cls, float(probs[cls]), probs


def get_flow(t):
    if t == "normal":    return generate_normal(1)[0]
    if t == "syn_flood": return generate_ddos(1, attack_type="syn_flood")[0]
    if t == "udp_flood": return generate_ddos(1, attack_type="udp_flood")[0]
    if t == "botnet":    return generate_botnet(1)[0]
    return generate_normal(1)[0]


def split_flow_count(total, phase_count):
    base, remainder = divmod(total, phase_count)
    return [base + (index < remainder) for index in range(phase_count)]


def render_history_chart(status_slot, chart_slot, history, generated, target, running, buffered):
    if running:
        progress = f"FLOW {generated}/{target}"
        if not history:
            progress += f" · MODEL WINDOW {buffered}/{WINDOW}"
        status_slot.markdown(
            f'<div class="chart-toolbar"><span>Prediction confidence</span>'
            f'<span class="live-pill">LIVE · {progress}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        status_slot.markdown(
            '<div class="chart-toolbar"><span>Prediction confidence</span></div>',
            unsafe_allow_html=True,
        )

    if not history:
        message = (
            f"Collecting traffic window · {buffered}/{WINDOW} flows"
            if running else
            'Press <b style="color:#58a6ff">▶ Start</b> to begin the simulation'
        )
        chart_slot.markdown(
            f'<div style="background:#010409;border:1px dashed #30363d;'
            f'border-radius:10px;padding:60px 24px;text-align:center;'
            f'color:#8b949e;font-size:13px">{message}</div>',
            unsafe_allow_html=True,
        )
        return

    flows = [point["flow"] for point in history]
    labels = [point["label"] for point in history]
    confs = [point["conf"] * 100 for point in history]
    clrs = [COLORS.get(label, "#888") for label in labels]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=flows, y=confs, fill="tozeroy",
        fillcolor="rgba(88,166,255,0.04)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=flows, y=confs, mode="lines+markers",
        line=dict(color="#58a6ff", width=2.5, shape="spline"),
        marker=dict(color=clrs, size=7, line=dict(width=1.5, color="#0d1117")),
        customdata=labels,
        hovertemplate="<b>Flow #%{x}</b><br>%{customdata}<br>%{y:.1f}%<extra></extra>",
        showlegend=False,
    ))

    in_attack, attack_start, attack_label = False, None, None
    for index, (flow, label) in enumerate(zip(flows, labels)):
        if label != "Normal" and not in_attack:
            in_attack, attack_start, attack_label = True, flow, label
        elif label == "Normal" and in_attack:
            fig.add_vrect(
                x0=attack_start, x1=flows[index - 1],
                fillcolor=BG_COLORS.get(attack_label, "rgba(200,0,0,0.1)"),
                layer="below", line_width=0,
            )
            in_attack = False
    if in_attack and attack_start is not None:
        fig.add_vrect(
            x0=attack_start, x1=flows[-1],
            fillcolor=BG_COLORS.get(attack_label, "rgba(200,0,0,0.1)"),
            layer="below", line_width=0,
        )

    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#010409",
        font=dict(color="#8b949e", size=11),
        xaxis=dict(title="Generated flow #", gridcolor="#21262d", color="#8b949e",
                   showgrid=True, zeroline=False),
        yaxis=dict(title="Confidence %", range=[0, 108], gridcolor="#21262d",
                   color="#8b949e", showgrid=True, zeroline=False),
        margin=dict(l=50, r=12, t=12, b=40),
        hovermode="x unified",
    )
    chart_slot.plotly_chart(
        fig, width="stretch", key="live_detection_chart",
    )


def _drain_explanation_queue(pending, lock):
    while True:
        with lock:
            if not pending:
                return
            job = pending.pop(0)

        card = job["card"]
        card["pending"] = False
        card["streaming"] = True
        try:
            for model_name in job["models"]:
                card["active_model"] = model_name
                card["responses"][model_name] = ""
                if len(job["models"]) == 1:
                    card["response"] = ""

                tokens = []
                for token in stream_explanation(
                    job["label"],
                    job["features"],
                    use_rag=job["use_rag"],
                    model=model_name,
                    retrieved_chunks=job["retrieved_chunks"],
                ):
                    tokens.append(token)
                    card["responses"][model_name] += token
                    if len(job["models"]) == 1:
                        card["response"] += token

                complete_text = finalize_explanation(
                    "".join(tokens),
                    job["label"],
                    job["features"],
                    model=model_name,
                    retrieved_chunks=job["retrieved_chunks"],
                )
                card["responses"][model_name] = complete_text
                if len(job["models"]) == 1:
                    card["response"] = complete_text
                if job["label"] != "Normal":
                    try:
                        if job["quality_scorer"] is None:
                            raise RuntimeError(job["quality_scorer_error"])
                        card["quality_scores"][model_name] = score_explanation(
                            complete_text,
                            job["label"],
                            job["features"],
                            job["retrieved_chunks"],
                            scorer=job["quality_scorer"],
                        )
                    except Exception as error:
                        print(
                            f"Quality scoring failed for flow #{job['flow_id']} "
                            f"with {model_name}: {type(error).__name__}: {error}",
                            file=sys.stderr,
                            flush=True,
                        )
                        traceback.print_exc(file=sys.stderr)
                        card["quality_scores"][model_name] = {
                            name: None for name in QUALITY_SCORE_NAMES
                        }
                    job["flow_record"]["explanations"][
                        job["explanation_index"]
                    ].setdefault("quality_scores", {})[model_name] = (
                        card["quality_scores"][model_name]
                    )
        except Exception as error:
            card["error"] = f"{type(error).__name__}: {error}"
        finally:
            card["streaming"] = False
            card["active_model"] = None


def _ensure_explanation_worker():
    ss = st.session_state
    worker = ss.explanation_worker
    if worker is not None and worker.is_alive():
        return

    with ss.explanation_lock:
        if not ss.pending_explanations:
            return
        worker = threading.Thread(
            target=_drain_explanation_queue,
            args=(ss.pending_explanations, ss.explanation_lock),
            daemon=True,
        )
        ss.explanation_worker = worker
        worker.start()


def queue_explanation(flow_record, use_rag, selected_model, compare_all):
    ss = st.session_state
    retrieved_chunks = []
    normal_detection = flow_record["label"] == "Normal"
    if use_rag and not normal_detection:
        retrieved_chunks = load_retriever()(
            f"IoT {flow_record['label']} network anomaly characteristics and mitigations",
            k=3,
            attack_type=flow_record["label"],
        )

    models = list(EXPLANATION_MODELS) if compare_all else [selected_model]
    card = {
        "label": flow_record["label"],
        "confidence": round(flow_record["confidence"] * 100, 1),
        "flow": flow_record["flow_id"],
        "time": flow_record["time"],
        "phase": flow_record["phase"],
        "response": None,
        "responses": {model_name: "" for model_name in models},
        "models": models,
        "compare_all": compare_all,
        "pending": True,
        "streaming": False,
        "active_model": None,
        "p_normal": round(float(flow_record["probs"][0]) * 100, 1),
        "p_ddos": round(float(flow_record["probs"][1]) * 100, 1),
        "p_botnet": round(float(flow_record["probs"][2]) * 100, 1),
        "use_rag": use_rag,
        "quality_scores": {},
        "sources": [
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "score": chunk["score"],
            }
            for chunk in retrieved_chunks
        ],
    }
    explanation_index = len(flow_record["explanations"])
    ss.llm_cards.append(card)
    flow_record["explanations"].append(card)
    with ss.explanation_lock:
        ss.pending_explanations.append({
            "card": card,
            "label": flow_record["label"],
            "features": flow_record["features"],
            "use_rag": use_rag and not normal_detection and bool(retrieved_chunks),
            "retrieved_chunks": retrieved_chunks,
            "models": models,
            "quality_scorer": quality_scorer,
            "quality_scorer_error": quality_scorer_error,
            "flow_id": flow_record["flow_id"],
            "flow_record": flow_record,
            "explanation_index": explanation_index,
        })
    _ensure_explanation_worker()


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
def init():
    defaults = {
        "running": False, "done": False,
        "history": deque(maxlen=120),
        "log_lines": deque(maxlen=40),
        "stats": {"Normal": 0, "DDoS": 0, "Botnet": 0},
        "alerts": [],
        "llm_cards": [],
        "buffer": [], "flow_count": 0,
        "flows_generated": 0, "target_flows": 30,
        "flow_records": [], "scenario_plan": [],
        "scenario_idx": 0,
        "current_phase": "Idle", "current_label": "Normal",
        "current_conf": 0.0, "current_probs": [1.0, 0.0, 0.0],
        "last_predicted_label": "Normal",
        "pending_explanations": [], "explanation_lock": threading.Lock(),
        "explanation_worker": None,
        "simulation_run_id": 0,
        "health": 100.0,
        "sim_start": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()
model = load_model()


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🛡️ IoT Anomaly Shield")
    st.caption("CNN-LSTM · Ollama LLaMA 3.2:3b")
    st.markdown("---")

    scenario_choice = st.selectbox(
        "Scenario",
        SCENARIO_OPTIONS,
        disabled=st.session_state.running,
    )
    total_flows = int(st.number_input(
        "Number of flows",
        min_value=20,
        value=30,
        step=1,
        disabled=st.session_state.running,
    ))

    scenario_valid = True
    if scenario_choice == "Custom":
        custom_phase_count = int(st.number_input(
            "Custom phases",
            min_value=2,
            max_value=5,
            value=3,
            step=1,
            disabled=st.session_state.running,
        ))
        default_counts = split_flow_count(total_flows, custom_phase_count)
        default_types = ("Normal", "DDoS", "Normal", "Botnet", "Normal")
        selected_plan = []
        for phase_index in range(custom_phase_count):
            phase_col, count_col = st.columns([2, 1])
            phase_type = phase_col.selectbox(
                f"Phase {phase_index + 1}",
                tuple(CUSTOM_PHASE_TYPES),
                index=tuple(CUSTOM_PHASE_TYPES).index(default_types[phase_index]),
                key=f"custom_phase_type_{phase_index}",
                disabled=st.session_state.running,
            )
            phase_flows = int(count_col.number_input(
                "Flows",
                min_value=1,
                value=default_counts[phase_index],
                step=1,
                key=f"custom_phase_flows_{phase_index}",
                disabled=st.session_state.running,
            ))
            selected_plan.append((
                CUSTOM_PHASE_TYPES[phase_type],
                phase_flows,
                f"Phase {phase_index + 1} — {phase_type}",
            ))
        scenario_valid = sum(phase[1] for phase in selected_plan) == total_flows
        if not scenario_valid:
            st.warning("Custom phase flow counts must add up to the total number of flows.")
    else:
        phases = PRESET_SCENARIOS[scenario_choice]
        phase_counts = split_flow_count(total_flows, len(phases))
        selected_plan = [
            (traffic_type, phase_counts[index], description)
            for index, (traffic_type, description) in enumerate(phases)
        ]

    # ── Scenario Timeline ──
    st.markdown("**Scenario Timeline**")
    for i, (traffic_type, phase_flows, desc) in enumerate(selected_plan):
        icon = "🟢" if traffic_type == "normal" else ("🔴" if traffic_type in ("syn_flood", "udp_flood") else "🟣")
        idx  = st.session_state.scenario_idx
        if st.session_state.done:
            cls_name = "done"
        elif st.session_state.running:
            if i < idx:
                cls_name = "done"
            elif i == idx and st.session_state.running:
                cls_name = "active"
            else:
                cls_name = "pending"
        else:
            cls_name = "pending"
        st.markdown(
            f'<div class="phase-item {cls_name}">{icon} {desc} <span style="margin-left:auto;font-size:11px;color:#6e7681">{phase_flows} flows</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    flow_speed = st.slider("Flow interval (s)", 0.1, 0.8, 0.3, 0.05)
    enable_llm = st.toggle("🤖 LLM Analysis", value=True)
    use_rag = st.toggle(
        "📚 LLM-only / LLM+RAG",
        value=True,
        help="On: use retrieved references. Off: explain from traffic features only.",
    )
    ollama_model = st.selectbox(
        "Ollama model",
        EXPLANATION_MODELS,
    )
    compare_all_models = st.checkbox("Compare all models", value=False)
    show_quality_scores = st.toggle("Show quality scores", value=True)
    with st.expander("Score definitions"):
        st.caption("Relevance — semantic match to the detected class and input features.")
        st.caption("Accuracy — supported claims, contradictions, and MITRE ID validity.")
        st.caption("Actionability — concrete steps, required evidence, and mitigations.")
        st.caption("Consistency — correct class diagnosis and quoted feature values.")
        st.caption("Clarity — readability, formatting, and repeated phrasing.")
    st.markdown("---")

    c1, c2 = st.columns(2)
    explanation_busy = any(
        card.get("pending") or card.get("streaming")
        for card in st.session_state.llm_cards
    )
    start_btn = c1.button("▶ Start", use_container_width=True, type="primary",
                           disabled=st.session_state.running or not scenario_valid or explanation_busy)
    stop_btn  = c2.button("⏹ Stop",  use_container_width=True,
                           disabled=not st.session_state.running)

    if start_btn:
        ss = st.session_state
        ss.history    = deque(maxlen=120)
        ss.log_lines  = deque(maxlen=40)
        ss.alerts     = []
        ss.llm_cards  = []
        ss.buffer     = []
        ss.stats      = {"Normal": 0, "DDoS": 0, "Botnet": 0}
        ss.flow_records = []
        ss.scenario_plan = selected_plan
        ss.target_flows = total_flows
        ss.flows_generated = 0
        ss.flow_count = 0
        ss.scenario_idx = 0
        ss.pending_explanations = []
        ss.sim_start    = time.time()
        ss.simulation_run_id += 1
        ss.current_phase = selected_plan[0][2]
        ss.current_label = "Normal"
        ss.current_conf  = 0.0
        ss.current_probs = [1.0, 0.0, 0.0]
        ss.last_predicted_label = "Normal"
        ss.health  = 100.0
        ss.running = True
        ss.done    = False
        st.rerun()

    if stop_btn:
        st.session_state.running = False
        st.rerun()

    # Phase progress
    if st.session_state.running:
        generated = st.session_state.flows_generated
        target = st.session_state.target_flows
        st.progress(min(generated / target, 1.0))
        st.caption(
            f"Flow {generated}/{target} · "
            f"{max(0, generated - WINDOW + 1)} predictions"
        )

    st.markdown("---")
    st.markdown(f"""
<div style="font-size:12px;color:#6e7681;line-height:1.8">
    <b style="color:#8b949e">Device:</b> {DEVICE.upper()}<br>
    <b style="color:#8b949e">Accuracy:</b> 99.92%<br>
    <b style="color:#8b949e">Features:</b> 38 · Window: 10<br>
    <b style="color:#8b949e">Model:</b> CNN-LSTM (218K params)
</div>
""", unsafe_allow_html=True)


quality_scorer, quality_scorer_error = load_quality_scorer()


# ═══════════════════════════════════════════════════════════════════════════════
#  STATUS HEADER
# ═══════════════════════════════════════════════════════════════════════════════
_lbl   = st.session_state.current_label
_color = COLORS.get(_lbl, "#58a6ff")
_icon  = ICONS.get(_lbl, "⚪")
_conf  = st.session_state.current_conf
_probs = st.session_state.current_probs

_status_cls  = "safe" if _lbl == "Normal" else ("botnet" if _lbl == "Botnet" else "attack")
_pulse_cls   = "pulse-green" if _lbl == "Normal" else ("pulse-purple" if _lbl == "Botnet" else "pulse-red")
_status_text = "SECURE" if _lbl == "Normal" else f"THREAT DETECTED — {_lbl.upper()}"

_elapsed_str = ""
if st.session_state.running:
    _elapsed = time.time() - st.session_state.sim_start
    _elapsed_str = f"<span style='color:#6e7681;font-size:12px;margin-left:12px'>{_elapsed:.0f}s elapsed</span>"

st.markdown(f"""
<div class="status-header {_status_cls}">
    <div class="pulse-dot {_pulse_cls}"></div>
    <div style="flex:1">
        <div style="color:#6e7681;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">
            Network Status {_elapsed_str}
        </div>
        <div style="color:{_color};font-size:24px;font-weight:700">{_status_text}</div>
        <div style="color:#8b949e;font-size:13px;margin-top:2px">{st.session_state.current_phase}</div>
    </div>
    <div style="text-align:right">
        <div style="color:#6e7681;font-size:11px;text-transform:uppercase;letter-spacing:1px">Confidence</div>
        <div style="color:var(--text-0);font-size:36px;font-weight:700;line-height:1.1">{_conf*100:.1f}<span style="font-size:18px;color:#8b949e">%</span></div>
        <div style="color:#6e7681;font-size:12px;margin-top:2px">Flow #{st.session_state.flows_generated}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PROBABILITY STRIP
# ═══════════════════════════════════════════════════════════════════════════════
_pn = _probs[0] * 100
_pd = _probs[1] * 100
_pb = _probs[2] * 100
st.markdown(f"""
<div style="margin:-8px 0 16px">
    <div class="prob-strip">
        <div style="width:{max(_pn,1):.1f}%;background:var(--green)"></div>
        <div style="width:{max(_pd,1):.1f}%;background:var(--red)"></div>
        <div style="width:{max(_pb,1):.1f}%;background:var(--purple)"></div>
    </div>
    <div class="prob-labels">
        <span style="color:var(--green)">🟢 Normal {_pn:.1f}%</span>
        <span style="color:var(--red)">🔴 DDoS {_pd:.1f}%</span>
        <span style="color:var(--purple)">🟣 Botnet {_pb:.1f}%</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════════════════
stats  = st.session_state.stats
total  = max(sum(stats.values()), 1)
health = st.session_state.health

_h_color = "#3fb950" if health > 70 else ("#d29922" if health > 30 else "#f85149")
_h_label = "Healthy" if health > 70 else ("Degraded" if health > 30 else "Critical")

c1, c2, c3, c4, c5, c6 = st.columns(6)

def mcard(col, icon, label, val, pct, color):
    col.markdown(f"""
<div class="m-card">
    <div class="m-icon">{icon}</div>
    <div class="m-label">{label}</div>
    <div class="m-value">{val}</div>
    <div class="m-bar"><div class="m-fill" style="width:{min(pct,100):.0f}%;background:{color}"></div></div>
    <div class="m-pct" style="color:{color}">{pct:.1f}%</div>
</div>""", unsafe_allow_html=True)

mcard(c1, "📊", "Total Flows",   total if total > 1 else 0, 100, "#58a6ff")
mcard(c2, "🟢", "Normal",  stats["Normal"],  stats["Normal"]/total*100,  "#3fb950")
mcard(c3, "🔴", "DDoS",    stats["DDoS"],    stats["DDoS"]/total*100,    "#f85149")
mcard(c4, "🟣", "Botnet",  stats["Botnet"],  stats["Botnet"]/total*100,  "#a371f7")
mcard(c5, "🚨", "Alerts",  len(st.session_state.alerts),
      len(st.session_state.alerts)/max(total,1)*100, "#d29922")
mcard(c6, "💚", "Health",  f"{health:.0f}", health, _h_color)


# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE MONITOR — Chart + Flow Log
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">📈 Live Detection Monitor</div>',
            unsafe_allow_html=True)

chart_col, log_col = st.columns([3, 2])

with chart_col:
    chart_status = st.empty()
    chart_display = st.empty()
    if not st.session_state.running:
        render_history_chart(
            chart_status,
            chart_display,
            list(st.session_state.history),
            st.session_state.flows_generated,
            st.session_state.target_flows,
            False,
            len(st.session_state.buffer),
        )

with log_col:
    st.markdown('<div class="section-hdr" style="margin-top:0">🖥️ Flow Log</div>',
                unsafe_allow_html=True)
    log_lines = list(st.session_state.log_lines)
    if log_lines:
        lines_html = "<br>".join(reversed(log_lines))
        st.markdown(f'<div class="flow-log">{lines_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="flow-log"><span style="color:#6e7681">'
            '$ awaiting network flows...</span></div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  ACTIVE THREAT BANNER
# ═══════════════════════════════════════════════════════════════════════════════
if _lbl != "Normal" and st.session_state.running:
    _t_cls  = "ddos" if _lbl == "DDoS" else "botnet"
    st.markdown(f"""
<div class="threat-banner {_t_cls}">
    <span style="font-size:20px">{_icon}</span>
    <div style="flex:1">
        <div style="color:{_color};font-weight:700;font-size:15px">🚨 ACTIVE THREAT: {_lbl.upper()}</div>
        <div style="color:#8b949e;font-size:12px;margin-top:2px">
            Confidence: {_conf*100:.1f}% · Flow #{st.session_state.flows_generated} · {st.session_state.current_phase}
        </div>
    </div>
    <div style="text-align:right;font-family:'JetBrains Mono',monospace;font-size:12px;color:#8b949e">
        N:{_pn:.0f}% · D:{_pd:.0f}% · B:{_pb:.0f}%
    </div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">🤖 AI Security Analysis</div>',
            unsafe_allow_html=True)

llm_cards = st.session_state.llm_cards
if not llm_cards:
    st.markdown("""
<div style="background:var(--bg-2);border:1px dashed var(--border);border-radius:12px;
            padding:36px;text-align:center;color:#6e7681;font-size:13px">
    🤖 AI analysis will stream here in real-time when an attack is detected
</div>""", unsafe_allow_html=True)
else:
    for card in reversed(llm_cards):
        _c_lbl    = card["label"]
        _c_color  = COLORS.get(_c_lbl, "#58a6ff")
        _c_icon   = ICONS.get(_c_lbl, "⚠️")
        _latest   = (card is llm_cards[-1])
        _pending  = card.get("pending", card.get("response") is None)
        _stream   = card.get("streaming", False)
        _text     = card.get("response") or ""
        _models   = card.get("models", [ollama_model])

        if card.get("error"):
            _status = "⚠️ error"
        elif _pending:
            _status = "⏳ connecting to LLM..."
        elif _stream:
            _status = "⚡ streaming"
        else:
            _status = f"✓ {card['confidence']:.0f}% conf"

        with st.expander(
            f"{_c_icon} {_c_lbl} · {_status} · Flow #{card['flow']} · {card['time']}",
            expanded=_latest,
        ):
            col_text, col_prob = st.columns([5, 2])
            with col_text:
                if card.get("compare_all"):
                    model_columns = st.columns(3)
                    for model_column, model_name in zip(model_columns, _models):
                        with model_column:
                            st.caption(model_name)
                            model_text = card["responses"].get(model_name, "")
                            if _pending and not model_text:
                                st.caption("Queued...")
                            else:
                                model_cursor = (
                                    '<span class="cursor">▌</span>'
                                    if _stream and card.get("active_model") == model_name
                                    else ""
                                )
                                safe_model_text = (
                                    model_text.replace("&", "&amp;")
                                    .replace("<", "&lt;")
                                    .replace(">", "&gt;")
                                )
                                st.markdown(
                                    f'<div class="llm-response" style="border-left-color:{_c_color}">'
                                    f'{safe_model_text}{model_cursor}</div>',
                                    unsafe_allow_html=True,
                                )
                            if (
                                show_quality_scores
                                and _c_lbl != "Normal"
                            ):
                                model_scores = card.get(
                                    "quality_scores", {}
                                ).get(model_name)
                                if model_scores:
                                    render_quality_panel(model_scores)
                elif _pending:
                    st.markdown(f"""
<div class="llm-pending" style="border-left:3px solid {_c_color}">
    <div style="color:#58a6ff;font-size:13px;margin-bottom:4px">⏳ Connecting to {_models[0]}...</div>
    <div style="color:#6e7681;font-size:12px">Simulation continues — response streams live</div>
</div>""", unsafe_allow_html=True)
                else:
                    _cursor    = '<span class="cursor">▌</span>' if _stream else ""
                    _safe_text = _text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    st.markdown(
                        f'<div class="llm-response" style="border-left-color:{_c_color}">'
                        f'{_safe_text}{_cursor}</div>',
                        unsafe_allow_html=True,
                    )
                    if show_quality_scores and _c_lbl != "Normal":
                        model_scores = card.get("quality_scores", {}).get(_models[0])
                        if model_scores:
                            render_quality_panel(model_scores)
                if card.get("error"):
                    st.error(card["error"])
            with col_prob:
                _pn_c = card.get("p_normal", 0)
                _pd_c = card.get("p_ddos", 0)
                _pb_c = card.get("p_botnet", 0)
                st.markdown(f"""
<div style="background:var(--bg-0);border-radius:8px;padding:14px">
    <div style="color:#6e7681;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px">
        Model Output
    </div>
    <div style="margin:8px 0">
        <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px">
            <span style="color:#3fb950">🟢 Normal</span><span style="color:#8b949e">{_pn_c:.1f}%</span>
        </div>
        <div style="background:var(--bg-3);border-radius:4px;height:6px;overflow:hidden">
            <div style="background:#3fb950;width:{_pn_c:.0f}%;height:6px;border-radius:4px"></div>
        </div>
    </div>
    <div style="margin:8px 0">
        <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px">
            <span style="color:#f85149">🔴 DDoS</span><span style="color:#8b949e">{_pd_c:.1f}%</span>
        </div>
        <div style="background:var(--bg-3);border-radius:4px;height:6px;overflow:hidden">
            <div style="background:#f85149;width:{_pd_c:.0f}%;height:6px;border-radius:4px"></div>
        </div>
    </div>
    <div style="margin:8px 0">
        <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px">
            <span style="color:#a371f7">🟣 Botnet</span><span style="color:#8b949e">{_pb_c:.1f}%</span>
        </div>
        <div style="background:var(--bg-3);border-radius:4px;height:6px;overflow:hidden">
            <div style="background:#a371f7;width:{_pb_c:.0f}%;height:6px;border-radius:4px"></div>
        </div>
    </div>
    <hr style="border-color:var(--border);margin:12px 0 8px">
    <div style="color:#6e7681;font-size:11px">{card['phase']}</div>
</div>""", unsafe_allow_html=True)
        with st.expander("Retrieved sources"):
            if not card.get("use_rag", False):
                st.caption("RAG was disabled for this explanation.")
            elif not card.get("sources"):
                st.caption("No reference chunks were retrieved.")
            else:
                for source in card["sources"]:
                    page = source["page"] if source["page"] is not None else "N/A"
                    st.text(
                        f"{source['source']} | Page: {page} | "
                        f"Similarity score: {source['score']:.3f}"
                    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ATTACK EVENT LOG
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">📋 Attack Event Log</div>',
            unsafe_allow_html=True)

alerts = st.session_state.alerts
if not alerts:
    st.markdown("""
<div style="background:var(--bg-2);border:1px dashed var(--border);border-radius:12px;
            padding:24px;text-align:center;color:#6e7681;font-size:13px">
    No attacks detected yet
</div>""", unsafe_allow_html=True)
else:
    rows = ""
    for a in reversed(alerts[-25:]):
        _a_clr = COLORS.get(a["label"], "#888")
        _a_ico = ICONS.get(a["label"], "")
        rows += f"""<tr>
            <td style="color:#8b949e;font-family:'JetBrains Mono',monospace;font-size:12px">{a['time']}</td>
            <td style="color:{_a_clr};font-weight:600">{_a_ico} {a['label']}</td>
            <td style="color:var(--text-0);font-weight:600">{a['confidence']:.1f}%</td>
            <td style="color:#8b949e">#{a['flow']}</td>
            <td style="color:#8b949e">{a['phase']}</td>
        </tr>"""
    st.markdown(f"""
<table class="alert-table">
    <thead><tr>
        <th>Time</th><th>Type</th><th>Confidence</th><th>Flow</th><th>Phase</th>
    </tr></thead>
    <tbody>{rows}</tbody>
</table>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE SIMULATION FLOWS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">🔬 Live Simulation</div>',
            unsafe_allow_html=True)
if not st.session_state.flow_records:
    st.caption("Predictions appear after the 10-flow window is full.")
else:
    for flow_record in reversed(st.session_state.flow_records[-20:]):
        flow_label = flow_record["label"]
        flow_icon = ICONS.get(flow_label, "⚠️")
        with st.expander(
            f"{flow_icon} Flow #{flow_record['flow_id']} · {flow_label} · "
            f"{flow_record['confidence'] * 100:.1f}% · {flow_record['time']}",
            expanded=False,
        ):
            st.markdown(
                f"**Phase:** {flow_record['phase']}  \n"
                f"**Prediction window:** {WINDOW} flows"
            )
            st.button(
                "Explain this flow",
                key=(
                    f"explain_{st.session_state.simulation_run_id}_"
                    f"{flow_record['flow_id']}_{len(flow_record['explanations'])}"
                ),
                on_click=queue_explanation,
                args=(
                    flow_record,
                    use_rag,
                    ollama_model,
                    compare_all_models,
                ),
            )
            with st.expander("Observed feature values (latest timestep)"):
                st.json(flow_record["features"][-1])
            if flow_record["explanations"]:
                st.caption(
                    f"{len(flow_record['explanations'])} explanation(s) attached to this flow."
                )
                for explanation_index, explanation in enumerate(flow_record["explanations"], 1):
                    with st.expander(f"Explanation {explanation_index}"):
                        if explanation.get("compare_all"):
                            explanation_columns = st.columns(3)
                            for explanation_column, model_name in zip(
                                explanation_columns,
                                explanation["models"],
                            ):
                                with explanation_column:
                                    st.caption(model_name)
                                    response = explanation["responses"].get(model_name, "")
                                    if explanation.get("streaming") and explanation.get("active_model") == model_name:
                                        response += " ▌"
                                    st.text(response or "Waiting for response...")
                                    if (
                                        show_quality_scores
                                        and flow_label != "Normal"
                                    ):
                                        model_scores = explanation.get(
                                            "quality_scores", {}
                                        ).get(model_name)
                                        if model_scores:
                                            render_quality_panel(model_scores)
                        else:
                            st.text(
                                explanation.get("response")
                                or "Waiting for response..."
                            )
                            if show_quality_scores and flow_label != "Normal":
                                model_scores = explanation.get(
                                    "quality_scores", {}
                                ).get(explanation["models"][0])
                                if model_scores:
                                    render_quality_panel(model_scores)


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPLETION BANNER
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.done:
    _total_a = len(st.session_state.alerts)
    _dur     = time.time() - st.session_state.sim_start if st.session_state.sim_start else 0
    st.markdown(f"""
<div class="done-banner">
    <div style="font-size:28px;margin-bottom:8px">✅</div>
    <div style="color:#3fb950;font-size:18px;font-weight:700;margin-bottom:4px">Simulation Complete</div>
    <div style="color:#8b949e;font-size:13px">
        {st.session_state.flows_generated} flows simulated ·
        {st.session_state.flow_count} predictions · {_total_a} threats detected ·
        {_dur:.0f}s duration · Network health: {st.session_state.health:.0f}%
    </div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULATION STEP — MUST BE LAST (st.rerun() halts execution)
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.running:
    ss = st.session_state
    flow_id = ss.flows_generated + 1
    phase_end = 0
    for phase_index, (traffic_type, phase_flows, phase_desc) in enumerate(ss.scenario_plan):
        phase_end += phase_flows
        if flow_id <= phase_end:
            ss.scenario_idx = phase_index
            break
    ss.current_phase = phase_desc

    flow = get_flow(traffic_type)
    ss.buffer.append(flow)
    if len(ss.buffer) > WINDOW:
        ss.buffer.pop(0)
    ss.flows_generated = flow_id

    if len(ss.buffer) == WINDOW:
        ss.flow_count += 1
        prediction_id = ss.flow_count
        cls, conf, probs = predict(model, np.array(ss.buffer))
        lbl = LABELS[cls]
        ts = datetime.now().strftime("%H:%M:%S")

        ss.current_label = lbl
        ss.current_conf = conf
        ss.current_probs = probs.tolist()
        ss.stats[lbl] += 1
        ss.history.append({"flow": flow_id, "label": lbl, "conf": conf})

        if cls != 0:
            ss.health = max(0.0, ss.health - 1.5)
        else:
            ss.health = min(100.0, ss.health + 0.4)

        features = [
            {name: float(value) for name, value in zip(FEATURES, row)}
            for row in np.asarray(ss.buffer)
        ]
        flow_record = {
            "flow_id": flow_id,
            "prediction_id": prediction_id,
            "label": lbl,
            "confidence": conf,
            "probs": probs.tolist(),
            "time": ts,
            "phase": phase_desc,
            "features": features,
            "explanations": [],
        }
        ss.flow_records.append(flow_record)

        clr = COLORS[lbl]
        ss.log_lines.append(
            f'<span style="color:#6e7681">[{ts}]</span> '
            f'<span style="color:{clr};font-weight:600">#{flow_id:>4} {lbl:8}</span> '
            f'<span style="color:var(--text-1)">{conf*100:.1f}%</span> '
            f'<span style="color:#6e7681"> {phase_desc}</span>'
        )

        if cls != 0 and conf >= 0.85:
            ss.alerts.append({
                "time": ts, "label": lbl,
                "confidence": round(conf * 100, 1),
                "flow": flow_id, "phase": phase_desc,
                "p_normal": round(float(probs[0]) * 100, 1),
                "p_ddos": round(float(probs[1]) * 100, 1),
                "p_botnet": round(float(probs[2]) * 100, 1),
            })

        state_changed_to_attack = (
            lbl in ("DDoS", "Botnet") and lbl != ss.last_predicted_label
        )
        ss.last_predicted_label = lbl
        if enable_llm and state_changed_to_attack:
            queue_explanation(
                flow_record,
                use_rag,
                ollama_model,
                compare_all_models,
            )

    render_history_chart(
        chart_status,
        chart_display,
        list(ss.history),
        ss.flows_generated,
        ss.target_flows,
        True,
        len(ss.buffer),
    )

    if ss.flows_generated >= ss.target_flows:
        ss.running = False
        ss.done = True
    else:
        time.sleep(flow_speed)
    st.rerun()
elif any(
    card.get("pending") or card.get("streaming")
    for card in st.session_state.llm_cards
):
    time.sleep(0.1)
    st.rerun()