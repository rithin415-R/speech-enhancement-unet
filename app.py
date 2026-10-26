import torch
import numpy as np
import soundfile as sf
import librosa
import gradio as gr
import matplotlib.pyplot as plt
import whisper
from src.model import SpeechEnhancer

# 1. Load your Speech Enhancement AI
print("Loading Enhancement AI Model...")
device = torch.device("cpu")
model = SpeechEnhancer().to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()
print("Enhancement Model Loaded!")

# 2. Load OpenAI's Whisper AI
print("Loading Whisper ASR Model...")
whisper_model = whisper.load_model("base")
print("Whisper Model Loaded!")

# 3. Function to calculate Estimated SNR
def calculate_snr(clean_signal, noisy_signal):
    # The noise is whatever is left when you subtract the clean signal from the noisy signal
    noise = noisy_signal - clean_signal
    
    # Calculate power (volume) of the clean signal and the noise
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Add a tiny number to noise_power to prevent dividing by zero
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    return snr

# 4. Main processing function
def enhance_and_transcribe(audio_file):
    if audio_file is None:
        return None, None, "Please upload audio first.", "", "", ""
        
    # --- Step A: Load audio ---
    print("Loading audio file...")
    noisy_audio, sr = librosa.load(audio_file, sr=16000)
    
    # --- Step B: Transcribe the NOISY audio ---
    print("Transcribing noisy audio...")
    noisy_result = whisper_model.transcribe(noisy_audio)
    noisy_text = noisy_result["text"]
    
    # --- Step C: Enhance the audio ---
    print("Enhancing audio...")
    noisy_tensor = torch.from_numpy(noisy_audio).float().unsqueeze(0).unsqueeze(0).to(device)
    
    chunk_size = 32000
    enhanced_audio = np.zeros_like(noisy_audio)
    
    with torch.no_grad():
        for i in range(0, len(noisy_audio), chunk_size):
            chunk = noisy_tensor[:, :, i:i+chunk_size]
            if chunk.shape[2] < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.shape[2]))
            enhanced_chunk = model(chunk)
            enhanced_audio[i:i+chunk_size] = enhanced_chunk.squeeze().cpu().numpy()[:len(noisy_audio)-i]
            
    # --- Step D: Transcribe the CLEAN audio ---
    print("Transcribing enhanced audio...")
    clean_result = whisper_model.transcribe(enhanced_audio)
    clean_text = clean_result["text"]
    
    # --- Step E: Calculate SNR Metric ---
    print("Calculating Metrics...")
    snr_score = calculate_snr(enhanced_audio, noisy_audio)
    metric_text = f"✅ Estimated Noise Removed: {snr_score:.2f} dB"
    
    # --- Step F: Create Spectrogram ---
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].set_title("Noisy Audio (Input)")
    ax[0].specgram(noisy_audio, Fs=sr, cmap="inferno")
    ax[0].set_ylabel("Frequency")
    ax[1].set_title("Enhanced Audio (AI Output)")
    ax[1].specgram(enhanced_audio, Fs=sr, cmap="inferno")
    ax[1].set_ylabel("Frequency")
    ax[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plot_path = "spectrogram_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    
    status = "✅ Processing Complete!"
    
    # Return: Audio, Spectrogram, Noisy Text, Clean Text, Metric, Status
    return (sr, enhanced_audio), plot_path, noisy_text, clean_text, metric_text, status

# 5. Build the Web Interface using Gradio
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎙️ Speech Enhancement & Recognition AI")
    gr.Markdown("Upload noisy audio. The AI will clean it, transcribe it, and score the improvement!")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Input Noisy Audio")
            process_btn = gr.Button("Enhance & Transcribe", variant="primary")
            status_text = gr.Textbox(label="Status")
            
        with gr.Column():
            audio_output = gr.Audio(label="Output Clean Audio")
            spectrogram_output = gr.Image(label="Before & After Spectrogram")
            
    # --- NEW: Metrics Dashboard ---
    gr.Markdown("### 📊 Objective Metrics Dashboard")
    metric_output = gr.Textbox(label="Signal-to-Noise Ratio (SNR) Improvement", interactive=False)
            
    gr.Markdown("### 📝 Speech Recognition Comparison (Word Error Rate Proof)")
    with gr.Row():
        noisy_transcript = gr.Textbox(label="Transcript from NOISY Audio (ASR)", interactive=False)
        clean_transcript = gr.Textbox(label="Transcript from ENHANCED Audio (ASR)", interactive=False)
    
    # Connect the button to the function
    process_btn.click(
        fn=enhance_and_transcribe, 
        inputs=audio_input, 
        outputs=[audio_output, spectrogram_output, noisy_transcript, clean_transcript, metric_output, status_text]
    )

# 6. Run the app
if __name__ == "__main__":
    app.launch()