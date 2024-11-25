# dataset.py
import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
import re


class RekordboxAudioDataset(Dataset):
    def __init__(self, audio_dir, transform=None):
        self.audio_dir = audio_dir
        self.transform = transform
        
        # Load all audio files and extract tempos from filenames
        self.audio_files = []
        self.true_tempos = []
        self.librosa_tempos = []
        self.onset_strengths = []
        
        tempo_pattern = re.compile(r'^(\d+\.?\d*)\s+(.+)$')
        
        for filename in os.listdir(audio_dir):
            if filename.endswith(('.mp3', '.wav', '.aiff', '.m4a')):
                match = tempo_pattern.match(filename)
                if match:
                    try:
                        # Get true tempo from filename
                        true_tempo = float(match.group(1))
                        audio_path = os.path.join(audio_dir, filename)
                        
                        # Load audio file
                        duration = librosa.get_duration(path=audio_path)
                        offset = duration / 3 if duration else 0
                        y, sr = librosa.load(audio_path, offset=offset, duration=30)
                        
                        # Get librosa tempo prediction
                        librosa_tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                        
                        # Calculate onset strength
                        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                        normalized_onset = librosa.util.normalize(onset_env)
                        
                        # Extract features to check shape
                        features = self._extract_features(y, sr)
                        if features.shape[0] != 256:  # Updated expected shape
                            print(f"Skipping {filename}: Unexpected feature shape {features.shape}")
                            continue
                            
                        # Store everything
                        self.audio_files.append(filename)
                        self.true_tempos.append(true_tempo)
                        self.librosa_tempos.append(librosa_tempo)
                        self.onset_strengths.append(normalized_onset)
                        
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
                        continue
        
        print(f"Successfully loaded {len(self.audio_files)} files")