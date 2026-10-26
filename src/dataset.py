import os
import torch
from torch.utils.data import Dataset
import random
import soundfile as sf
import numpy as np

class SpeechEnhancementDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, segment_length=32000, sample_rate=16000):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        
        self.file_list = [f for f in os.listdir(clean_dir) if f.endswith('.wav')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.file_list[idx])
        noisy_file = os.path.join(self.noisy_dir, self.file_list[idx])
        
        # 1. Load audio using soundfile (returns numpy arrays)
        clean_data, sr1 = sf.read(clean_file)
        noisy_data, sr2 = sf.read(noisy_file)
        
        # 2. Convert to mono if stereo
        if len(clean_data.shape) > 1:
            clean_data = clean_data[:, 0]
        if len(noisy_data.shape) > 1:
            noisy_data = noisy_data[:, 0]
            
        # 3. Convert numpy arrays to PyTorch tensors and add channel dimension: [1, samples]
        clean_audio = torch.from_numpy(clean_data).float().unsqueeze(0)
        noisy_audio = torch.from_numpy(noisy_data).float().unsqueeze(0)

        # 4. Cut a random 2-second chunk (32000 samples)
        length = min(clean_audio.shape[1], noisy_audio.shape[1])
        
        if length < self.segment_length:
            padding = self.segment_length - length
            clean_audio = torch.nn.functional.pad(clean_audio, (0, padding))
            noisy_audio = torch.nn.functional.pad(noisy_audio, (0, padding))
        else:
            start = random.randint(0, length - self.segment_length)
            clean_audio = clean_audio[:, start:start+self.segment_length]
            noisy_audio = noisy_audio[:, start:start+self.segment_length]
            
        return noisy_audio, clean_audio

# --- Test block ---
if __name__ == "__main__":
    print("Testing Dataset Loader...")
    
    clean_dir = os.path.join("data", "clean")
    noisy_dir = os.path.join("data", "noisy")
    
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)
    if not os.path.exists(noisy_dir):
        os.makedirs(noisy_dir)
    
    dummy_clean = np.random.randn(48000).astype(np.float32)
    dummy_noisy = np.random.randn(48000).astype(np.float32)
    
    clean_path = os.path.join(clean_dir, "test1.wav")
    noisy_path = os.path.join(noisy_dir, "test1.wav")
    
    if not os.path.exists(clean_path):
        sf.write(clean_path, dummy_clean, 16000)
    if not os.path.exists(noisy_path):
        sf.write(noisy_path, dummy_noisy, 16000)
    
    dataset = SpeechEnhancementDataset(clean_dir, noisy_dir)
    noisy, clean = dataset[0]
    
    print(f"Loaded noisy audio shape: {noisy.shape}")
    print(f"Loaded clean audio shape: {clean.shape}")
    print("Dataset script is working perfectly!")