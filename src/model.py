import torch
import torch.nn as nn

class SpeechEnhancer(nn.Module):
    def __init__(self):
        super(SpeechEnhancer, self).__init__()
        
        # ENCODER: Extracts features and shrinks audio representation
        self.enc1 = nn.Conv1d(1, 32, kernel_size=8, stride=4, padding=2)
        self.enc2 = nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2)
        self.enc3 = nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2)
        
        # BOTTLENECK: Deepest layer representing the compressed audio
        self.bottleneck = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        
        # DECODER: Rebuilds the audio back to original size
        self.dec3 = nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=2)
        self.dec2 = nn.ConvTranspose1d(128, 32, kernel_size=8, stride=4, padding=2) 
        self.dec1 = nn.ConvTranspose1d(64, 1, kernel_size=8, stride=4, padding=2)   
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder pass
        e1 = self.relu(self.enc1(x))   # Size: 8000
        e2 = self.relu(self.enc2(e1))  # Size: 2000
        e3 = self.relu(self.enc3(e2))  # Size: 500
        
        # Bottleneck
        b = self.relu(self.bottleneck(e3)) # Size: 500
        
        # Decoder pass WITH U-Net Skip Connections
        d3 = self.relu(self.dec3(b))       # Size grows back to 2000
        d3 = torch.cat((d3, e2), dim=1)   # Glue d3 with e2 -> 128 channels
        
        d2 = self.relu(self.dec2(d3))     # Size grows back to 8000
        d2 = torch.cat((d2, e1), dim=1)   # Glue d2 with e1 -> 64 channels
        
        d1 = self.dec1(d2)                # Size grows back to 32000 (Clean audio)
        
        return d1

# Quick test to ensure it runs
if __name__ == "__main__":
    print("Initializing Real-World AI Model...")
    model = SpeechEnhancer()
    dummy_audio = torch.randn(1, 1, 32000) 
    output = model(dummy_audio)
    print(f"Input shape: {dummy_audio.shape}")
    print(f"Output shape: {output.shape}")
    print("Model Architecture Built Successfully!")