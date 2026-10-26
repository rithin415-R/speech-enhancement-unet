import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import SpeechEnhancementDataset
from src.model import SpeechEnhancer
import os

def train_model():
    print("Starting Training Process...")
    
    # 1. Setup device (Use GPU if you have one, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Initialize Dataset and DataLoader
    dataset = SpeechEnhancementDataset("data/clean", "data/noisy")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 3. Initialize Model, Loss Function, and Optimizer
    model = SpeechEnhancer().to(device)
    criterion = nn.L1Loss() # L1 Loss (Mean Absolute Error) as mentioned in the paper
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training Loop
    epochs = 10 # Number of times to read the data
    os.makedirs("models", exist_ok=True)
    
    for epoch in range(epochs):
        model.train() # Set model to training mode
        total_loss = 0
        
        for batch_idx, (noisy_audio, clean_audio) in enumerate(dataloader):
            noisy_audio = noisy_audio.to(device)
            clean_audio = clean_audio.to(device)
            
            # Forward pass: AI guesses the clean audio
            predictions = model(noisy_audio)
            
            # Calculate loss: How far off is the guess from the real clean audio?
            loss = criterion(predictions, clean_audio)
            
            # Backward pass: AI updates its brain to do better next time
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
    # 5. Save the trained model
    torch.save(model.state_dict(), "models/best_model.pth")
    print("Training Complete! Model saved to 'models/best_model.pth'")

if __name__ == "__main__":
    train_model()