import os
import urllib.request
import tarfile
import librosa
import soundfile as sf
import numpy as np
import glob
import zipfile

def download_and_prepare_data():
    print("Downloading real human speech (LibriSpeech)...")
    speech_url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    speech_filename = "dev-clean.tar.gz"
    
    if not os.path.exists(speech_filename):
        urllib.request.urlretrieve(speech_url, speech_filename)
    if not os.path.exists("data/LibriSpeech"):
        with tarfile.open(speech_filename, "r:gz") as tar:
            tar.extractall("data/")

    print("Downloading real environmental noises (ESC-50)...")
    noise_url = "https://github.com/karoldvl/ESC-50/archive/refs/heads/master.zip"
    noise_filename = "esc50.zip"
    
    if not os.path.exists(noise_filename):
        urllib.request.urlretrieve(noise_url, noise_filename)
    if not os.path.exists("data/ESC-50-master"):
        with zipfile.ZipFile(noise_filename, 'r') as zip_ref:
            zip_ref.extractall("data/")
            
    clean_dir = os.path.join("data", "clean")
    noisy_dir = os.path.join("data", "noisy")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)
    
    # Find 500 speech files and 500 real noise files (dogs, rain, cars, etc.)
    speech_files = glob.glob("data/LibriSpeech/dev-clean/**/*.flac", recursive=True)[:500]
    noise_files = glob.glob("data/ESC-50-master/audio/*.wav")[:500]
    
    print(f"Found {len(speech_files)} speech files and {len(noise_files)} real noise files. Mixing...")
    
    for i, speech_path in enumerate(speech_files):
        # 1. Load clean speech
        clean_audio, sr = librosa.load(speech_path, sr=16000)
        
        # 2. Load a REAL random noise (dog barking, typing, street, etc.)
        noise_path = np.random.choice(noise_files)
        noise_audio, _ = librosa.load(noise_path, sr=16000)
        
        # 3. Loop the noise if it's shorter than the speech
        if len(noise_audio) < len(clean_audio):
            repeat_times = int(np.ceil(len(clean_audio) / len(noise_audio))) + 1
            noise_audio = np.tile(noise_audio, repeat_times)
        noise_audio = noise_audio[:len(clean_audio)]
        
        # 4. Mix them at a realistic volume (Random SNR between 0dB and 10dB)
        clean_power = np.mean(clean_audio**2)
        noise_power = np.mean(noise_audio**2)
        target_snr_db = np.random.uniform(0, 10) # Very noisy!
        scaling_factor = np.sqrt(clean_power / (noise_power + 1e-10)) * (10 ** (-target_snr_db / 20))
        noisy_audio = clean_audio + (noise_audio * scaling_factor)
        
        # 5. Save
        clean_path = os.path.join(clean_dir, f"sample_{i}.wav")
        noisy_path = os.path.join(noisy_dir, f"sample_{i}.wav")
        sf.write(clean_path, clean_audio, 16000)
        sf.write(noisy_path, noisy_audio, 16000)
        
    print("✅ Real-World Dataset Ready! 500 real noisy samples created.")

if __name__ == "__main__":
    download_and_prepare_data()