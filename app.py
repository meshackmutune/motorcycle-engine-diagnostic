import streamlit as st
import onnxruntime as ort
import numpy as np
import librosa
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Acoustic Diagnostics AI",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PROFESSIONAL DASHBOARD LOOK ---
st.markdown("""
<style>
    /* Dark Theme Core Styles */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Card Container */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    /* Primary Accent Styling */
    .stButton>button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
        box-shadow: 0 0 12px rgba(2, 132, 199, 0.5);
    }

    /* Hide standard header decoration */
    header[data-testid="stHeader"] {
        background: rgba(15, 23, 42, 0.8);
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_onnx_session():
    return ort.InferenceSession("motorcycle_sound_model.onnx")

try:
    session = load_onnx_session()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# --- MAPS & CONSTANTS ---
CLASSES = ["Valve Ticking", "Chain Slap", "Exhaust Leak", "Healthy Idle"]
REPAIRS = {
    "Valve Ticking": "**Action Required:** Loose Valve Clearance.\n\nInspect tappets and replace valve shims to correct clearance tolerances.",
    "Chain Slap": "**Action Required:** Loose or Dry Drive Chain.\n\nAdjust drive chain tension to spec and apply high-viscosity lube.",
    "Exhaust Leak": "**Action Required:** Exhaust Manifold Leak.\n\nCheck manifold bolt torque specs or replace worn exhaust gasket.",
    "Healthy Idle": "**Status Normal:** Engine Operating within normal acoustic specs.\n\nNo immediate mechanical intervention required."
}

# --- VISUALIZATION GENERATOR ---
def generate_spectrogram_plot(y, sr, mel_spec_db):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    fig.patch.set_facecolor('#1e293b')
    
    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='#38bdf8')
    ax1.set_title("Time-Domain Waveform Signal", color="#94a3b8", fontsize=10, fontweight="bold")
    ax1.set_facecolor('#0f172a')
    ax1.tick_params(colors='#94a3b8')
    ax1.spines['bottom'].set_color('#334155')
    ax1.spines['top'].set_color('#334155')
    ax1.spines['left'].set_color('#334155')
    ax1.spines['right'].set_color('#334155')

    # Spectrogram
    img = librosa.display.specshow(
        mel_spec_db, x_axis='time', y_axis='mel', sr=sr, ax=ax2, cmap='magma'
    )
    ax2.set_title("Acoustic Mel-Spectrogram (Frequency Features)", color="#94a3b8", fontsize=10, fontweight="bold")
    ax2.set_facecolor('#0f172a')
    ax2.tick_params(colors='#94a3b8')
    ax2.spines['bottom'].set_color('#334155')
    ax2.spines['top'].set_color('#334155')
    ax2.spines['left'].set_color('#334155')
    ax2.spines['right'].set_color('#334155')
    
    cbar = fig.colorbar(img, ax=ax2, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='#94a3b8')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#94a3b8')

    plt.tight_layout()
    return fig

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.title("Control Panel")
    st.markdown("---")
    
    st.markdown("### System Status")
    if model_loaded:
        st.success("ONNX Runtime: Ready")
    else:
        st.error(f"Model Error: {model_error}")
        
    st.markdown("### Audio Config")
    st.caption("**Sample Rate:** 22,050 Hz")
    st.caption("**Window Size:** 3.0 Seconds")
    st.caption("**Feature Input:** 128 Mel Bins")
    
    st.markdown("---")
    st.markdown("**Version:** 1.0.4-prod")

# --- MAIN DASHBOARD INTERFACE ---
st.title("🏍️ Motorcycle Acoustic AI Diagnostic")
st.markdown("Automated fault identification via deep learning spectral analysis.")

# Layout Columns
col_left, col_right = st.columns([1, 1.8])

with col_left:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("1. Acoustic Input")
    
    audio_file = st.file_uploader(
        "Upload Engine Recording (.wav / .mp3)", 
        type=["wav", "mp3"],
        help="Provide a clear 3-second recording of the engine idling."
    )
    
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")
        run_analysis = st.button("⚡ Run AI Diagnostic")
    else:
        run_analysis = False
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    if audio_file is not None and run_analysis:
        with st.spinner("Processing Mel-Spectrogram matrix & inferring fault states..."):
            # Load audio
            y, sr = librosa.load(audio_file, sr=22050)
            
            # Normalize length to 3s
            target_length = 22050 * 3
            if len(y) > target_length:
                y = y[:target_length]
            else:
                y = np.pad(y, (0, target_length - len(y)))

            # Feature extraction
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=22050, n_fft=2048, hop_length=512, n_mels=128
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            input_tensor = mel_spec_db[np.newaxis, np.newaxis, :, :130].astype(np.float32)

            # Model inference
            outputs = session.run(None, {"mel_spectrogram": input_tensor})
            raw_logits = outputs[0][0]
            
            exp_logits = np.exp(raw_logits - np.max(raw_logits))
            probabilities = exp_logits / exp_logits.sum()

            top_idx = np.argmax(probabilities)
            top_conf = probabilities[top_idx]
            top_class = CLASSES[top_idx]

            # Threshold Guardrail
            if top_class != "Healthy Idle" and top_conf < 0.85:
                diagnosis = "Healthy Idle / Inconclusive"
                action_text = (
                    f"**Low Confidence Flag ({top_conf:.1%}):** Signal indicates slight sound variance "
                    f"resembling {CLASSES[top_idx]}, but failed to pass the 85% certainty threshold."
                )
            else:
                diagnosis = top_class
                action_text = REPAIRS[top_class]

        # Diagnostic KPIs
        st.subheader("2. Diagnostic Summary")
        kpi_col1, kpi_col2 = st.columns(2)
        
        with kpi_col1:
            st.metric("Primary Detection", diagnosis)
        with kpi_col2:
            st.metric("Confidence Score", f"{top_conf:.1%}")

        # Action Panel
        if diagnosis == "Healthy Idle":
            st.success(action_text)
        else:
            st.warning(action_text)

        # Probability Breakdown
        st.markdown("### Confidence Distribution")
        for i, class_name in enumerate(CLASSES):
            st.progress(float(probabilities[i]), text=f"{class_name}: {probabilities[i]:.1%}")

        # Visualizations
        st.markdown("### Signal Spectrum Analysis")
        fig = generate_spectrogram_plot(y, sr, mel_spec_db)
        st.pyplot(fig)
        
    else:
        st.info("👈 Upload an audio file on the left panel and click **Run AI Diagnostic** to display results.")
