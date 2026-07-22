import gradio as gr
import onnxruntime as ort
import numpy as np
import librosa
import matplotlib.pyplot as plt

# 1. Load ONNX model session
session = ort.InferenceSession("motorcycle_sound_model.onnx")

CLASSES = ["Valve Ticking", "Chain Slap", "Exhaust Leak", "Healthy Idle"]

REPAIRS = {
    "Valve Ticking": "Loose Valve Clearance\nFIX: Inspect tappets and replace valve shims.",
    "Chain Slap": "Loose or Dry Chain\nFIX: Adjust drive chain tension or apply lube.",
    "Exhaust Leak": "Exhaust Manifold Leak\nFIX: Check manifold bolt torque or replace gasket.",
    "Healthy Idle": "Engine running normally!\nFIX: No immediate repairs required."
}

def create_plots(y, sr, mel_spec_db):
    """Generates Matplotlib figure containing Waveform and Spectrogram"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.patch.set_facecolor('#111827')  # Dark background matching UI
    
    # 1. Plot Time-Domain Audio Waveform
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='#38bdf8')
    ax1.set_title("Audio Waveform (Time Domain)", color="white", fontsize=12, fontweight="bold")
    ax1.set_facecolor('#1e293b')
    ax1.tick_params(colors='white')
    ax1.xaxis.label.set_color('white')
    ax1.yaxis.label.set_color('white')

    # 2. Plot Frequency Mel-Spectrogram
    img = librosa.display.specshow(
        mel_spec_db, x_axis='time', y_axis='mel', sr=sr, ax=ax2, cmap='magma'
    )
    ax2.set_title("Mel-Spectrogram (Frequency Features Fed to ONNX Model)", color="white", fontsize=12, fontweight="bold")
    ax2.set_facecolor('#1e293b')
    ax2.tick_params(colors='white')
    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    
    cbar = fig.colorbar(img, ax=ax2, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()
    return fig

def predict_engine_sound(audio_path):
    if audio_path is None:
        return None, "Please upload or record an audio clip.", None

    # Load audio at 22.05 kHz
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Pad or truncate to 3 seconds
    target_length = 22050 * 3
    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = np.pad(y, (0, target_length - len(y)))

    # Compute Mel-Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=22050, n_fft=2048, hop_length=512, n_mels=128
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Format tensor for ONNX
    input_tensor = mel_spec_db[np.newaxis, np.newaxis, :, :130].astype(np.float32)

    # Run ONNX inference
    outputs = session.run(None, {"mel_spectrogram": input_tensor})
    raw_logits = outputs[0][0]
    
    # Compute Softmax probabilities
    exp_logits = np.exp(raw_logits - np.max(raw_logits))
    probabilities = exp_logits / exp_logits.sum()

    confidence_scores = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
    
    # --- CONFIDENCE THRESHOLD GUARD ---
    top_index = np.argmax(probabilities)
    top_confidence = probabilities[top_index]
    top_prediction = CLASSES[top_index]

    # Require >= 85% confidence to diagnose a mechanical fault
    if top_prediction != "Healthy Idle" and top_confidence < 0.85:
        top_prediction = "Healthy Idle"
        repair_recommendation = (
            f"Note: Model suspected {CLASSES[top_index]} ({top_confidence:.1%}), but confidence was below the 85% threshold.\n"
            "Engine sounds generally healthy or recording contains high background noise.\n"
            "FIX: No immediate repairs required. Record closer to the engine if issue persists."
        )
    else:
        repair_recommendation = REPAIRS[top_prediction]
    # ----------------------------------

    # Generate visual figure
    visual_plot = create_plots(y, sr, mel_spec_db)

    return confidence_scores, repair_recommendation, visual_plot

# Build Layout
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown(
        """
        # 🏍️ Motorcycle Engine Sound Diagnostic AI
        Record or upload a 3-second audio recording of your motorcycle engine idle to analyze potential mechanical faults.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="Record or Upload Engine Sound")
            submit_btn = gr.Button("Analyze Engine Sound", variant="primary")
            
        with gr.Column(scale=1):
            diagnostic_label = gr.Label(label="Diagnostic Probabilities")
            repair_box = gr.Textbox(label="Recommended Repair Action", interactive=False)
            
    with gr.Row():
        visual_plot_output = gr.Plot(label="Audio Waveform & Frequency Spectrogram")

    submit_btn.click(
        fn=predict_engine_sound,
        inputs=[audio_input],
        outputs=[diagnostic_label, repair_box, visual_plot_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
