from flask import Flask, render_template, request, jsonify
import os
import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
from tinytag import TinyTag
from TellerNet.models.teller_net import TellerNet

# [Previous TellerNet class definition remains the same]

class WebInference:
    def __init__(self, models, target_length=1292):
        self.models = models
        self.target_length = target_length
        
    def normalize_tempo(self, librosa_tempo, true_tempo=120):
        while librosa_tempo < 78:
            librosa_tempo *= 2
        while librosa_tempo > 155:
            librosa_tempo /= 2
        return librosa_tempo

    def extract_features(self, audio_path):
        try:
            # Load audio with same parameters as training
            tag = TinyTag.get(audio_path)
            offset = tag.duration / 3 if tag.duration else 0
            y, sr = librosa.load(audio_path, offset=offset, duration=30)
            
            # Use librosa to estimate initial tempo
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo_librosa, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            
            # Normalize the estimated tempo
            tempo_librosa = self.normalize_tempo(tempo_librosa)
            
            # Normalize librosa estimated tempo to [0,1]
            tempo_librosa_norm = (tempo_librosa - 78) / 77
            
            # Compute MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs = librosa.util.fix_length(mfccs, size=self.target_length)
            
            # Compute spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_centroid = librosa.util.fix_length(spectral_centroid, size=self.target_length)
            
            # Normalize features
            mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-8)
            spectral_centroid = (spectral_centroid - np.mean(spectral_centroid)) / (np.std(spectral_centroid) + 1e-8)
            
            # Stack features
            tempo_feature = np.full((1, self.target_length), tempo_librosa_norm)
            features = np.vstack([tempo_feature, mfccs, spectral_centroid])
            
            return torch.FloatTensor(features)
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None

    def predict(self, features):
        features = features.to(next(self.models[0].parameters()).device)
        features = features.permute(0, 1, 2)  # Adjust dimensions for model
        
        all_predictions = []
        with torch.no_grad():
            for model in self.models:
                prediction = model(features)
                pred_bpm = prediction.item() * 77 + 78
                all_predictions.append(pred_bpm)
        
        # Get median prediction
        median_bpm = np.median(all_predictions)
        return median_bpm

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Load models at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_paths = [
    'pths/model_fold1_20241203_140536.pth',
    'pths/model_fold2_20241203_140553.pth',
    'pths/model_fold3_20241203_140611.pth',
    'pths/model_fold4_20241203_140633.pth',
    'pths/model_fold5_20241203_140716.pth',
]

ensemble_models = []
for path in model_paths:
    model = TellerNet()
    model.to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.eval()
    ensemble_models.append(model)

# Create inference instance
inference = WebInference(ensemble_models)

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
                # Extract features
                features = inference.extract_features(temp_path)
                if features is None:
                    raise ValueError("Failed to extract features from audio file")
                
                # Add batch dimension
                features = features.unsqueeze(0)
                
                # Get prediction
                tempo = inference.predict(features)
                
                # Clean up
                os.remove(temp_path)
                
                return jsonify({'tempo': round(tempo, 1)})
                
            except Exception as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                print(f"Error processing file: {str(e)}")
                return jsonify({'error': str(e)})
                
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)