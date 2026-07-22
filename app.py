import streamlit as st
import onnxruntime as ort
import numpy as np
import librosa
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Motorcycle Engine AI Diagnostic",
    page_icon="🏍️",
    layout="centered"
)

# --- LOAD ONNX MODEL ---
@st.cache_resource
def load_onnx_session():
    # Ensure motorcycle_sound_model.onnx is in the same folder as app.py
    return ort.InferenceSession("motorcycle_sound_model.onnx")

session = load_onnx_session()

# --- CONSTANTS & MAPS ---
CLASSES = ["Valve Ticking", "Chain Slap", "Exhaust Leak", "Healthy Idle"]

REPAIRS = {
    "Valve Ticking": "Loose Valve Clearance\nFIX: Inspect tappets and replace valve shims.",
    "Chain Slap": "Loose or Dry Chain\nFIX: Adjust drive chain tension or apply lube.",
    "Exhaust Leak": "Exhaust Manifold Leak\nFIX: Check manifold bolt torque or replace gasket.",
    "Healthy Idle": "Engine running normally!\nFIX: No immediate repairs required."
}

# --- VISUALIZATION FUNCTION ---
def create_plots(y, sr, mel_spec_db):
    """Generates Matplotlib figure containing Waveform and Spectrogram"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.patch.set_facecolor('#0e1117')
    
    # 1. Audio Waveform
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='#38bdf8')
    ax1.set_title("Audio Waveform (Time Domain)", color="white", fontsize=12, fontweight="bold")
    ax1.set_facecolor('#1e293b')
    ax1.tick_params(colors='white')

    # 2. Mel-Spectrogram
    img = librosa.display.specshow(
        mel_spec_db, x_axis='time', y_axis='mel', sr=sr, ax=ax2, cmap='magma'
    )
    ax2.set_title("Mel-Spectrogram Features (Fed to ONNX Model)", color="white", fontsize=12, fontweight="bold")
    ax2.set_facecolor('#1e293b')
    ax2.tick_params(colors='white')
    
    cbar = fig.colorbar(img, ax=ax2, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()
    return fig

# --- STREAMLIT UI LAYOUT ---
st.title("🏍️ Motorcycle Engine Sound Diagnostic AI")
st.write("Upload a 3-second audio recording of your motorcycle engine idle to detect potential mechanical faults.")

# Audio Uploader
audio_file = st.file_uploader("Upload Engine Sound (.wav or .mp3)", type=["wav", "mp3"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    
    if st.button("Analyze Engine Sound", type="primary"):
        with st.spinner("Processing audio with librosa & running ONNX model..."):
            
            # 1. Load audio signal at 22.05 kHz
            y, sr = librosa.load(audio_file, sr=22050)
            
            # 2. Pad or truncate audio to exactly 3 seconds (66,150 samples)
            target_length = 22050 * 3
            if len(y) > target_length:
                y = y[:target_length]
            else:
                y = np.pad(y, (0, target_length - len(y)))

            # 3. Compute Mel-Spectrogram features
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=22050, n_fft=2048, hop_length=512, n_mels=128
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 4. Format Input Tensor to match model shape [1, 1, 128, 130]
            input_tensor = mel_spec_db[np.newaxis, np.newaxis, :, :130].astype(np.float32)

            # 5. Execute ONNX Inference
            outputs = session.run(None, {"mel_spectrogram": input_tensor})
            raw_logits = outputs[0][0]
            
            # 6. Calculate Softmax probabilities
            exp_logits = np.exp(raw_logits - np.max(raw_logits))
            probabilities = exp_logits / exp_logits.sum()

            top_index = np.argmax(probabilities)
            top_confidence = probabilities[top_index]
            top_prediction = CLASSES[top_index]

            # --- 85% CONFIDENCE THRESHOLD GUARD ---
            if top_prediction != "Healthy Idle" and top_confidence < 0.85:
                final_diagnosis = "Healthy Idle / Inconclusive"
                repair_msg = (
                    f"Note: Model suspected {CLASSES[top_index]} ({top_confidence:.1%}), "
                    "but confidence was below the 85% threshold.\n\n"
                    "Engine sounds generally healthy or recording contains high background noise."
                )
            else:
                final_diagnosis = top_prediction
                repair_msg = REPAIRS[top_prediction]

        # --- RESULTS DISPLAY ---
        st.subheader(f"Diagnosis: {final_diagnosis}")
        st.info(f"**Recommended Repair Action:**\n\n{repair_msg}")

        # Confidence Score Progress Bars
        st.write("### Prediction Confidence Scores")
        for i, class_name in enumerate(CLASSES):
            st.progress(float(probabilities[i]), text=f"{class_name}: {probabilities[i]:.1%}")

        # Waveform & Spectrogram Plots
        st.write("### Audio Diagnostics Visualizations")
        fig = create_plots(y, sr, mel_spec_db)
        st.pyplot(fig)
