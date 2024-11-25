import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime



def train_model(model, train_loader, val_loader, num_epochs=50):
    # Create directories for saving models if they don't exist
    save_dir = os.path.join('models', 'saved')
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Lists to store metrics
    train_losses = []
    val_losses = []
    epoch_errors = []  # Store average BPM error per epoch
    best_val_loss = float('inf')
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_bpm_errors = []
        
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate BPM error
            bpm_errors = torch.abs(outputs - targets).cpu().detach().numpy()
            train_bpm_errors.extend(bpm_errors)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_bpm_errors = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Calculate BPM error
                bpm_errors = torch.abs(outputs - targets).cpu().numpy()
                val_bpm_errors.extend(bpm_errors)
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epoch_errors.append(np.mean(val_bpm_errors))
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Average BPM Error: {np.mean(val_bpm_errors):.1f}')
        print(f'Max BPM Error: {np.max(val_bpm_errors):.1f}')
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(save_dir, f'teller_net_{timestamp}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'bpm_error': np.mean(val_bpm_errors)
            }, model_path)
    
    # Plot training results
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # BPM Error plot
    plt.subplot(1, 2, 2)
    plt.plot(epoch_errors, label='Average BPM Error')
    plt.xlabel('Epoch')
    plt.ylabel('BPM Error')
    plt.legend()
    plt.title('Average BPM Error per Epoch')
    
    plt.tight_layout()
    plt.savefig(f'training_results_{timestamp}.png')
    plt.close()
    
    return model, train_losses, val_losses, epoch_errors

def evaluate_model(model, test_loader, device):
    """Evaluate the model's performance in detail"""
    model.eval()
    all_predictions = []
    all_targets = []
    errors = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            
            predictions = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            errors.extend(np.abs(predictions - targets))
    
    errors = np.array(errors)
    
    results = {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'std_error': np.std(errors),
        'max_error': np.max(errors),
        'accuracy_within_3bpm': np.mean(errors <= 3.0),
        'accuracy_within_5bpm': np.mean(errors <= 5.0),
        'accuracy_within_10bpm': np.mean(errors <= 10.0)
    }
    
    return results