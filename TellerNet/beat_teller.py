import torch
from torch.utils.data import DataLoader
from data.dataset import RekordboxAudioDataset
from models.teller_net import TellerNet
from models.train import train_model, evaluate_model
from utils.data_validate import validate_dataset

if __name__ == "__main__":
    # Define your audio directory
    audio_dir = "TellerNet/data/beatbank_train_audios"
    
    # Validate dataset first
    if not validate_dataset(audio_dir):
        print("Please add tempo labels to your audio files and try again.")
        exit(1)
    
    # Create dataset
    dataset = RekordboxAudioDataset(audio_dir=audio_dir)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create and train model
    model = TellerNet()
    model, train_losses, val_losses, epoch_errors = train_model(
        model, train_loader, val_loader, 5)
    
    # Evaluate the model
    results = evaluate_model(model, val_loader, torch.device)
    