import torch
from torch.utils.data import DataLoader
from TellerNet.data.dataset import RekordboxAudioDataset
from TellerNet.models.teller_net import TellerNet
from TellerNet.models.train import train_model

audio_dir = "TellerNet/data/beatbank_train_audios"
print(f"Using audio directory: {audio_dir}")

dataset = RekordboxAudioDataset(audio_dir)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train model
model = TellerNet()
train_model(model, train_loader, num_epochs=5)

# Save just the model weights - this will be a small file
torch.save(model.state_dict(), 'tempo_model_weights.pth')