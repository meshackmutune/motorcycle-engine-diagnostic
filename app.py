import streamlit as st
import onnxruntime as ort
import numpy as np
import librosa
import matplotlib.pyplot as plt
from datetime import datetime
from zoneinfo import ZoneInfo
import io

LOCAL_TZ = ZoneInfo("Africa/Nairobi")

def now_local() -> datetime:
    return datetime.now(LOCAL_TZ)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Acoustic Diagnostics AI",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# THEME / DESIGN TOKENS
# ============================================================
COLORS = {
    "bg":        "#0a0e1a",
    "surface":   "#131a2c",
    "surface_2": "#1a2338",
    "border":    "#2a3350",
    "text":      "#f1f4fb",
    "text_dim":  "#8b96b3",
    "accent":    "#7c5cff",
    "accent_2":  "#22d3ee",
    "success":   "#2dd4a7",
    "warning":   "#fbbf24",
    "danger":    "#fb5b8f",
}

CLASSES = ["Valve Ticking", "Chain Slap", "Exhaust Leak", "Healthy Idle"]

REPAIRS = {
    "Valve Ticking": {
        "severity": "warning",
        "summary": "Loose valve clearance detected in the acoustic signature.",
        "action": "Inspect tappets and replace valve shims to restore clearance tolerances to OEM spec.",
    },
    "Chain Slap": {
        "severity": "warning",
        "summary": "Drive chain slack or insufficient lubrication detected.",
        "action": "Adjust drive chain tension to spec and apply high-viscosity chain lubricant.",
    },
    "Exhaust Leak": {
        "severity": "danger",
        "summary": "Exhaust manifold leak signature detected.",
        "action": "Check manifold bolt torque and inspect/replace the exhaust gasket.",
    },
    "Healthy Idle": {
        "severity": "success",
        "summary": "Engine is operating within normal acoustic parameters.",
        "action": "No mechanical intervention required. Continue routine maintenance schedule.",
    },
}

CONFIDENCE_THRESHOLD = 0.85
SAMPLE_RATE = 22050
WINDOW_SECONDS = 3
TARGET_LENGTH = SAMPLE_RATE * WINDOW_SECONDS

CLASS_COLORS = {
    "Valve Ticking": "#fbbf24",
    "Chain Slap": "#fb923c",
    "Exhaust Leak": "#fb5b8f",
    "Healthy Idle": "#2dd4a7",
}

# ============================================================
# GLOBAL STYLES
# ============================================================
st.markdown(f"""
<style>
    .stApp {{
        background: radial-gradient(circle at 15% 0%, rgba(124, 92, 255, 0.10), transparent 45%),
                    radial-gradient(circle at 85% 15%, rgba(34, 211, 238, 0.08), transparent 40%),
                    {COLORS['bg']};
        color: {COLORS['text']};
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{
        background: rgba(11, 18, 32, 0.85);
        backdrop-filter: blur(6px);
    }}

    /* Typography */
    h1, h2, h3 {{
        letter-spacing: -0.02em;
    }}
    .app-title {{
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0;
        background: linear-gradient(90deg, {COLORS['text']} 0%, {COLORS['accent_2']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: inline-block;
    }}
    .app-subtitle {{
        color: {COLORS['text_dim']};
        font-size: 0.95rem;
        margin-top: 2px;
        margin-bottom: 1.5rem;
    }}
    .section-label {{
        text-transform: uppercase;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: {COLORS['accent_2']};
        margin-bottom: 6px;
    }}
    .section-label::before {{
        content: "";
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: linear-gradient(135deg, {COLORS['accent']}, {COLORS['accent_2']});
        margin-right: 6px;
    }}

    /* Cards */
    .card {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 16px;
        transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.25s ease;
    }}
    .card:hover {{
        border-color: {COLORS['accent']};
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.15);
        transform: translateY(-2px);
    }}
    .card-tight {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: border-color 0.25s ease, transform 0.25s ease;
    }}
    .card-tight:hover {{
        border-color: {COLORS['accent_2']};
        transform: translateX(3px);
    }}

    /* KPI tiles */
    .kpi {{
        background-color: {COLORS['surface_2']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        padding: 16px 18px;
        text-align: left;
        transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.2s ease;
    }}
    .kpi:hover {{
        border-color: {COLORS['accent']};
        box-shadow: 0 0 0 1px {COLORS['accent']}, 0 8px 18px rgba(59, 130, 246, 0.18);
        transform: translateY(-3px);
    }}
    .kpi-label {{
        color: {COLORS['text_dim']};
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .kpi-value {{
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 4px;
        color: {COLORS['text']};
    }}

    /* Status badges */
    .badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .badge:hover {{
        transform: scale(1.06);
    }}
    .badge-success {{ background: rgba(34,197,94,0.15); color: {COLORS['success']}; border: 1px solid rgba(34,197,94,0.35); }}
    .badge-success:hover {{ box-shadow: 0 0 10px rgba(34,197,94,0.5); }}
    .badge-warning {{ background: rgba(245,158,11,0.15); color: {COLORS['warning']}; border: 1px solid rgba(245,158,11,0.35); }}
    .badge-warning:hover {{ box-shadow: 0 0 10px rgba(245,158,11,0.5); }}
    .badge-danger  {{ background: rgba(239,68,68,0.15);  color: {COLORS['danger']};  border: 1px solid rgba(239,68,68,0.35); }}
    .badge-danger:hover {{ box-shadow: 0 0 10px rgba(239,68,68,0.5); }}

    /* Diagnostic result panel */
    .result-panel {{
        border-radius: 12px;
        padding: 20px 22px;
        margin-top: 10px;
        margin-bottom: 18px;
        border-left: 4px solid;
    }}
    .result-success {{ background: rgba(34,197,94,0.08); border-color: {COLORS['success']}; }}
    .result-warning {{ background: rgba(245,158,11,0.08); border-color: {COLORS['warning']}; }}
    .result-danger  {{ background: rgba(239,68,68,0.08);  border-color: {COLORS['danger']}; }}

    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, {COLORS['accent']} 0%, #5b3df0 45%, {COLORS['accent_2']} 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 700;
        width: 100%;
        transition: all 0.25s ease;
        letter-spacing: 0.01em;
    }}
    .stButton>button:hover {{
        background: linear-gradient(135deg, #6a4bff 0%, #4c31d6 45%, #38e4fb 100%);
        box-shadow: 0 0 18px rgba(124, 92, 255, 0.5);
        transform: translateY(-1px);
    }}
    .stDownloadButton>button {{
        background: transparent;
        color: {COLORS['accent_2']};
        border: 1px solid {COLORS['accent']};
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }}

    /* Progress bars (fallback) */
    .stProgress > div > div > div > div {{
        background-color: {COLORS['accent']};
    }}

    /* Custom color-coded confidence bars */
    .cbar-row {{
        margin-bottom: 12px;
    }}
    .cbar-head {{
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        margin-bottom: 5px;
    }}
    .cbar-name {{
        font-weight: 600;
        color: {COLORS['text']};
    }}
    .cbar-pct {{
        font-weight: 700;
    }}
    .cbar-track {{
        width: 100%;
        height: 10px;
        background: {COLORS['surface_2']};
        border: 1px solid {COLORS['border']};
        border-radius: 999px;
        overflow: hidden;
    }}
    .cbar-fill {{
        height: 100%;
        border-radius: 999px;
        transition: width 0.6s ease, filter 0.2s ease;
    }}
    .cbar-track:hover .cbar-fill {{
        filter: brightness(1.25);
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab"] {{
        color: {COLORS['text_dim']};
        font-weight: 600;
        transition: color 0.2s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {COLORS['accent_2']};
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLORS['accent_2']} !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: {COLORS['accent']} !important;
    }}

    /* Footer */
    .app-footer {{
        color: {COLORS['text_dim']};
        font-size: 0.75rem;
        text-align: center;
        padding-top: 28px;
        padding-bottom: 8px;
        border-top: 1px solid {COLORS['border']};
        margin-top: 24px;
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
def badge_class(severity: str) -> str:
    return {"success": "badge-success", "warning": "badge-warning", "danger": "badge-danger"}[severity]


def kpi_tile(label: str, value: str) -> str:
    return f"""
    <div class="kpi">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """


def confidence_bar(class_name: str, prob: float) -> str:
    color = CLASS_COLORS.get(class_name, COLORS["accent"])
    pct = prob * 100
    return f"""
    <div class="cbar-row">
        <div class="cbar-head">
            <span class="cbar-name">{class_name}</span>
            <span class="cbar-pct" style="color:{color};">{pct:.1f}%</span>
        </div>
        <div class="cbar-track">
            <div class="cbar-fill" style="width:{pct:.1f}%; background:{color};"></div>
        </div>
    </div>
    """


def build_report_text(diagnosis, confidence, timestamp) -> str:
    info = REPAIRS.get(diagnosis, REPAIRS["Healthy Idle"])
    lines = [
        "ACOUSTIC DIAGNOSTICS AI — INSPECTION REPORT",
        "=" * 46,
        f"Generated: {timestamp}",
        f"Primary Detection: {diagnosis}",
        f"Confidence Score: {confidence:.1%}",
        "",
        "Summary:",
        info["summary"],
        "",
        "Recommended Action:",
        info["action"],
        "",
        "-" * 46,
        "This report was generated automatically by an acoustic",
        "machine-learning model and should be verified by a",
        "qualified technician before service action is taken.",
    ]
    return "\n".join(lines)


@st.cache_resource(show_spinner=False)
def load_onnx_session():
    return ort.InferenceSession("motorcycle_sound_model.onnx")


def generate_spectrogram_plot(y, sr, mel_spec_db):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.2))
    fig.patch.set_facecolor(COLORS["surface"])

    librosa.display.waveshow(y, sr=sr, ax=ax1, color=COLORS["accent_2"])
    ax1.set_title("Time-Domain Waveform", color=COLORS["text_dim"], fontsize=10, fontweight="bold", loc="left")
    ax1.set_facecolor(COLORS["bg"])
    ax1.tick_params(colors=COLORS["text_dim"])
    for spine in ax1.spines.values():
        spine.set_color(COLORS["border"])

    img = librosa.display.specshow(
        mel_spec_db, x_axis='time', y_axis='mel', sr=sr, ax=ax2, cmap='magma'
    )
    ax2.set_title("Mel-Spectrogram (Frequency Features)", color=COLORS["text_dim"], fontsize=10, fontweight="bold", loc="left")
    ax2.set_facecolor(COLORS["bg"])
    ax2.tick_params(colors=COLORS["text_dim"])
    for spine in ax2.spines.values():
        spine.set_color(COLORS["border"])

    cbar = fig.colorbar(img, ax=ax2, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color=COLORS["text_dim"])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS["text_dim"])

    plt.tight_layout()
    return fig


# ============================================================
# SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# MODEL LOAD
# ============================================================
try:
    session = load_onnx_session()
    model_loaded = True
    model_error = None
except Exception as e:
    session = None
    model_loaded = False
    model_error = str(e)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🏍️ Acoustic Diagnostics AI")
    st.caption("Engine fault detection via deep learning spectral analysis")
    st.markdown("---")

    st.markdown("**System Status**")
    if model_loaded:
        st.markdown(
            f'<span class="badge badge-success">● MODEL READY</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="badge badge-danger">● MODEL ERROR</span>',
            unsafe_allow_html=True,
        )
        st.error(model_error, icon="⚠️")

    st.markdown("---")
    st.markdown("**Acquisition Parameters**")
    st.caption(f"Sample rate: {SAMPLE_RATE:,} Hz")
    st.caption(f"Analysis window: {WINDOW_SECONDS:.1f} s")
    st.caption("Feature input: 128 Mel bins")
    st.caption(f"Decision threshold: {CONFIDENCE_THRESHOLD:.0%}")

    st.markdown("---")
    st.markdown("**Session**")
    st.caption(f"Diagnostics run: {len(st.session_state.history)}")
    if st.session_state.history:
        if st.button("Clear session history", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.markdown("---")
    st.caption("Version 1.1.0 · Build prod")
    st.caption("© Acoustic Diagnostics AI")


# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="app-title">🏍️ Motorcycle Acoustic AI Diagnostic</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Automated fault identification from engine sound, powered by an ONNX spectral classification model.</div>',
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([1, 1.7], gap="large")

# ============================================================
# LEFT COLUMN — INPUT
# ============================================================
with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Step 1 — Acoustic Input</div>', unsafe_allow_html=True)
    st.markdown("Provide a clear recording of the engine idling (a 3-second clip works best).")

    tab_rec, tab_upload = st.tabs(["🎙️ Record Live", "📁 Upload File"])

    audio_source = None
    with tab_rec:
        recorded_audio = st.audio_input("Record engine idle")
        if recorded_audio:
            audio_source = recorded_audio

    with tab_upload:
        uploaded_audio = st.file_uploader(
            "Upload engine recording (.wav / .mp3)",
            type=["wav", "mp3"],
            help="Provide a clear recording of the engine idling.",
        )
        if uploaded_audio:
            audio_source = uploaded_audio

    if audio_source is not None:
        st.audio(audio_source)
        run_analysis = st.button("⚡ Run AI Diagnostic", disabled=not model_loaded)
        if not model_loaded:
            st.caption("⚠️ Diagnostic disabled until the model loads successfully.")
    else:
        run_analysis = False
        st.info("Record or upload a clip to begin.", icon="🎧")

    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Recent Diagnostics</div>', unsafe_allow_html=True)
        for entry in reversed(st.session_state.history[-5:]):
            st.markdown(
                f"""<div class="card-tight" style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div style="font-weight:600;">{entry['diagnosis']}</div>
                            <div style="color:{COLORS['text_dim']}; font-size:0.78rem;">{entry['timestamp']}</div>
                        </div>
                        <span class="badge {badge_class(entry['severity'])}">{entry['confidence']:.0%}</span>
                    </div>""",
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# RIGHT COLUMN — RESULTS
# ============================================================
with col_right:
    if audio_source is not None and run_analysis and model_loaded:
        try:
            with st.spinner("Processing Mel-spectrogram and running inference..."):
                y, sr = librosa.load(audio_source, sr=SAMPLE_RATE)

                if len(y) > TARGET_LENGTH:
                    y = y[:TARGET_LENGTH]
                else:
                    y = np.pad(y, (0, TARGET_LENGTH - len(y)))

                mel_spec = librosa.feature.melspectrogram(
                    y=y, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                input_tensor = mel_spec_db[np.newaxis, np.newaxis, :, :130].astype(np.float32)

                outputs = session.run(None, {"mel_spectrogram": input_tensor})
                raw_logits = outputs[0][0]

                exp_logits = np.exp(raw_logits - np.max(raw_logits))
                probabilities = exp_logits / exp_logits.sum()

                top_idx = int(np.argmax(probabilities))
                top_conf = float(probabilities[top_idx])
                top_class = CLASSES[top_idx]

                if top_class != "Healthy Idle" and top_conf < CONFIDENCE_THRESHOLD:
                    diagnosis = "Healthy Idle / Inconclusive"
                    severity = "warning"
                    summary = f"Signal weakly resembles {top_class} but did not clear the {CONFIDENCE_THRESHOLD:.0%} confidence threshold."
                    action = "Consider recording a longer or cleaner sample and re-running the diagnostic."
                else:
                    diagnosis = top_class
                    info = REPAIRS[top_class]
                    severity = info["severity"]
                    summary = info["summary"]
                    action = info["action"]

            timestamp = now_local().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.history.append({
                "diagnosis": diagnosis,
                "confidence": top_conf,
                "severity": severity,
                "timestamp": timestamp,
            })

            # --- Summary header ---
            st.markdown('<div class="section-label">Step 2 — Diagnostic Summary</div>', unsafe_allow_html=True)
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            with kpi_col1:
                st.markdown(kpi_tile("Primary Detection", diagnosis), unsafe_allow_html=True)
            with kpi_col2:
                st.markdown(kpi_tile("Confidence Score", f"{top_conf:.1%}"), unsafe_allow_html=True)
            with kpi_col3:
                st.markdown(kpi_tile("Analyzed At", timestamp.split(" ")[1]), unsafe_allow_html=True)

            # --- Result panel ---
            st.markdown(
                f"""<div class="result-panel result-{severity}">
                        <span class="badge {badge_class(severity)}">{diagnosis.upper()}</span>
                        <p style="margin-top:12px; margin-bottom:6px; font-weight:600;">{summary}</p>
                        <p style="margin:0; color:{COLORS['text_dim']};">{action}</p>
                    </div>""",
                unsafe_allow_html=True,
            )

            # --- Confidence distribution ---
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Confidence Distribution</div>', unsafe_allow_html=True)
            bars_html = "".join(
                confidence_bar(class_name, probabilities[i]) for i, class_name in enumerate(CLASSES)
            )
            st.markdown(bars_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # --- Spectrogram ---
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Signal Spectrum Analysis</div>', unsafe_allow_html=True)
            fig = generate_spectrogram_plot(y, sr, mel_spec_db)
            st.pyplot(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # --- Export ---
            report_text = build_report_text(diagnosis, top_conf, timestamp)
            st.download_button(
                "⬇️ Download Inspection Report (.txt)",
                data=report_text,
                file_name=f"acoustic_report_{timestamp.replace(' ', '_').replace(':', '-')}.txt",
                mime="text/plain",
            )

        except Exception as e:
            st.error(f"Analysis failed: {e}", icon="🚫")

    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Step 2 — Diagnostic Summary</div>', unsafe_allow_html=True)
        st.info(
            "👈 Record or upload an audio file on the left panel, then click **Run AI Diagnostic** "
            "to view the fault classification, confidence breakdown, and spectrogram.",
            icon="🎧",
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="app-footer">Acoustic Diagnostics AI is a decision-support tool. '
    'Results should be verified by a qualified technician before any repair action is taken.</div>',
    unsafe_allow_html=True,
)
