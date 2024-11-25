# audio_analyzer.py
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from TellerNet.utils.data_validate import validate_dataset
from TellerNet.data.dataset import RekordboxAudioDataset
from models.teller_net import TempoNet
from models.train import train_model

def analyze_file(file_path):
    """
    Analyze a single audio file using both ML model and traditional methods
    """
    try:
        # Load and initialize model
        model = TempoNet()
        model.load_state_dict(torch.load('best_tempo_model.pth'))
        model.eval()
        
        # Load audio
        y, sr = librosa.load(file_path)
        
        # Get librosa's basic tempo detection
        tempo_librosa, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Extract features for ML model
        features = RekordboxAudioDataset._extract_features(None, y, sr)
        features = torch.FloatTensor(features).unsqueeze(0)
        
        # Get ML model prediction
        with torch.no_grad():
            tempo_ml = model(features).item()
        
        # Combine predictions with weighted average
        # Give more weight to ML prediction if it's close to librosa's
        diff = abs(tempo_ml - tempo_librosa)
        if diff < 10:  # If predictions are close
            final_tempo = 0.7 * tempo_ml + 0.3 * tempo_librosa
        else:  # If predictions differ significantly
            final_tempo = 0.5 * tempo_ml + 0.5 * tempo_librosa
        
        result = {
            "tempo": round(final_tempo, 1),
            "confidence": round(1.0 / (1.0 + diff), 2),
            "detail": {
                "ml_tempo": round(tempo_ml, 1),
                "librosa_tempo": round(float(tempo_librosa), 1)
            }
        }
        
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({"error": str(e)})