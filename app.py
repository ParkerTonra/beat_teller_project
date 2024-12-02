from flask import Flask, render_template, request, jsonify
import os
import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
from tinytag import TinyTag

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

class TellerNet(nn.Module):
    def __init__(self):
        super(TellerNet, self).__init__()
        
        self.freq_convs = nn.Sequential(
            nn.Conv2d(34, 64, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 64, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            
            nn.Linear(32, 1)
        )
        
        self.tempo_scaling = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 34, 1, -1)
        
        conv_out = self.freq_convs(x)
        attention_weights = self.attention(conv_out)
        attended_features = conv_out * attention_weights
        
        lstm_input = attended_features.permute(0, 2, 3, 1)
        lstm_input = lstm_input.contiguous().view(batch_size, -1, 64)
        
        lstm_out, _ = self.lstm(lstm_input)
        pooled = torch.mean(lstm_out, dim=1)
        output = self.regression_head(pooled)
        output = output * self.tempo_scaling
        
        return output

# Load the model at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TellerNet()

# Load the complete checkpoint
checkpoint = torch.load('static/teller_net_20241201_173629_best.pth', map_location=device)

# Extract just the model state dict from the checkpoint
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

def process_audio(audio_path):
    """Process audio file using the same preprocessing as training"""
    try:
        # Load audio with same parameters as training
        tag = TinyTag.get(audio_path)
        offset = tag.duration / 3 if tag.duration else 0
        y, sr = librosa.load(audio_path, offset=offset, duration=30)
        
        # Get initial tempo estimate from librosa
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        librosa_tempo = float(tempo.item())
        
        # Normalize tempo to 78-155 range
        while librosa_tempo < 78:
            librosa_tempo *= 2
        while librosa_tempo > 155:
            librosa_tempo /= 2
            
        # Extract features
        features = extract_features(y, sr)
        
        # Convert to tensor and add batch dimension
        features = torch.FloatTensor(features)
        
        # Add librosa tempo information
        librosa_tempo_norm = (librosa_tempo - 78) / 77
        tempo_channel = torch.tensor([[librosa_tempo_norm] * features.shape[1]])
        
        # Calculate tempo confidence
        confidence_input = torch.tensor(-min(
            abs(librosa_tempo - 120),
            abs(librosa_tempo - 128),
            abs(librosa_tempo - 140)
        ) / 20.0)
        tempo_confidence = torch.exp(confidence_input)
        tempo_confidence = tempo_confidence.view(1, -1).expand(1, features.shape[1])
        
        # Concatenate all features
        features = torch.cat([
            tempo_channel,
            tempo_confidence,
            features
        ], dim=0)
        
        # Add batch dimension
        features = features.unsqueeze(0)
        
        return features
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        raise

def extract_features(y, sr):
    """Extract audio features matching training preprocessing"""
    # Fixed parameters
    hop_length = 512
    n_fft = 1024
    
    # Ensure audio length is consistent
    target_length = 30 * sr
    if len(y) > target_length:
        y = y[:target_length]
    elif len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    
    # Core rhythm features
    onset_env = librosa.onset.onset_strength(
        y=y, 
        sr=sr,
        hop_length=hop_length
    )
    onset_env = librosa.util.normalize(onset_env)
    
    # Beat tracking with two different configurations
    tempo1, beats1 = librosa.beat.beat_track(
        y=y, sr=sr, 
        hop_length=hop_length, 
        start_bpm=120, 
        tightness=100
    )
    
    tempo2, beats2 = librosa.beat.beat_track(
        y=y, sr=sr, 
        hop_length=hop_length, 
        start_bpm=140, 
        tightness=50
    )
    
    # Convert beats to onset envelopes
    beat_env1 = np.zeros(len(onset_env))
    beat_env2 = np.zeros(len(onset_env))
    beat_frames1 = beats1[beats1 < len(beat_env1)]
    beat_frames2 = beats2[beats2 < len(beat_env2)]
    beat_env1[beat_frames1] = 1.0
    beat_env2[beat_frames2] = 1.0
    
    # Compute reduced tempogram
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        win_length=384
    )
    
    # Reduce to fewer bands
    n_bands = 29
    tempogram_reduced = np.zeros((n_bands, tempogram.shape[1]))
    band_size = tempogram.shape[0] // n_bands
    for i in range(n_bands):
        start_idx = i * band_size
        end_idx = (i + 1) * band_size
        tempogram_reduced[i] = np.mean(tempogram[start_idx:end_idx], axis=0)
    
    tempo1_norm = (tempo1 - 78) / 77
    tempo2_norm = (tempo2 - 78) / 77
    
    # Stack features
    features = np.vstack([
        onset_env.reshape(1, -1),      # 1 band
        beat_env1.reshape(1, -1),      # 1 band
        beat_env2.reshape(1, -1),      # 1 band
        np.full((1, onset_env.shape[0]), tempo1_norm),  # Additional tempo feature
        np.full((1, onset_env.shape[0]), tempo2_norm),  # Additional tempo feature
        tempogram_reduced              # 29 bands
    ])
    
    # Normalize
    features = (features - features.mean()) / (features.std() + 1e-8)
    
    return features

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
            
        if file:
            # Save the uploaded file temporarily
            temp_path = os.path.join('static', 'temp', file.filename)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            file.save(temp_path)
            
            try:
                # Process the audio file
                features = process_audio(temp_path)
                features = features.to(device)
                
                # Make prediction
                with torch.no_grad():
                    prediction = model(features)
                
                # Denormalize the prediction back to BPM
                tempo = (prediction.item() * 77) + 78
                
                # Clean up
                os.remove(temp_path)
                
                return jsonify({'tempo': round(tempo, 1)})
                
            except Exception as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                print(f"Error processing file: {str(e)}")  # Add logging
                return jsonify({'error': str(e)})
                
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)