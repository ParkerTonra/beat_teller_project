from tinytag import TinyTag
import librosa
import os
import re
import numpy as np
from torch.utils.data import Dataset
import torch
import random
from torch.utils.data import Subset


class RekordboxAudioDataset(Dataset):
    def __init__(self, audio_dir, transform=None, target_length=1292    ):
        super().__init__()
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_length = target_length
        self.audio_files = []
        self.true_tempos = []
        self.features = []
        
        self._load_audio_files()
        
    def _pad_or_truncate_features(self, features):
        if features.shape[1] < self.target_length:
            padded = np.zeros((features.shape[0], self.target_length))
            padded[:, :features.shape[1]] = features
            return padded
        return features[:, :self.target_length]
        
    def __len__(self):
        return len(self.audio_files)
    
    def normalize_tempo(self, librosa_tempo, true_tempo):
        while librosa_tempo < 78:
            librosa_tempo *= 2
        while librosa_tempo > 155:
            librosa_tempo /= 2
            
        if librosa_tempo * 2 <= 155:
            if abs(librosa_tempo * 2 - true_tempo) < abs(librosa_tempo - true_tempo):
                librosa_tempo *= 2
        if librosa_tempo / 2 >= 78:
            if abs(librosa_tempo / 2 - true_tempo) < abs(librosa_tempo - true_tempo):
                librosa_tempo /= 2
                
        return librosa_tempo
    
    def _extract_features(self, y, sr, true_tempo):
        try:
            # Use librosa to estimate tempo
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo_librosa, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            
            # Normalize the estimated tempo
            tempo_librosa = self.normalize_tempo(tempo_librosa, true_tempo)
            
            # Normalize librosa estimated tempo to [0,1]
            tempo_librosa_norm = (tempo_librosa - 78) / 77
            
            # Compute MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs = librosa.util.fix_length(mfccs, size=self.target_length)
            
            # Compute spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_centroid = librosa.util.fix_length(spectral_centroid, size=self.target_length)
            
            mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-8)
    
            # Normalize spectral centroid
            spectral_centroid = (spectral_centroid - np.mean(spectral_centroid)) / (np.std(spectral_centroid) + 1e-8)
            
            # Stack features
            tempo_feature = np.full((1, self.target_length), tempo_librosa_norm)
            features = np.vstack([tempo_feature, mfccs, spectral_centroid])
            
            return features
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return np.zeros((15, self.target_length))
    
    def _load_audio_files(self):
        tempo_pattern = re.compile(r'^(\d+\.?\d*)\s+(.+)$')
        all_audio_files = [f for f in os.listdir(self.audio_dir) 
                        if f.endswith(('.mp3', '.wav', '.aiff', '.m4a'))]
        
        for filename in all_audio_files:
            match = tempo_pattern.match(filename)
            if match:
                try:
                    true_tempo = float(match.group(1))
                    audio_path = os.path.join(self.audio_dir, filename)
                    
                    tag = TinyTag.get(audio_path)
                    offset = tag.duration / 3 if tag.duration else 0
                    y, sr = librosa.load(audio_path, offset=offset, duration=30)
                    
                    features = self._extract_features(y, sr, true_tempo)
                    features = self._pad_or_truncate_features(features)
                    
                    self.audio_files.append(filename)
                    self.true_tempos.append(true_tempo)
                    self.features.append(features)
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue
            else:
                continue  # If filename doesn't match the pattern, skip it
    
    def __getitem__(self, idx):
        true_tempo = self.true_tempos[idx]
        features = self.features[idx]
        
        if self.transform:
            features = self.transform(features)
        
        features = torch.FloatTensor(features)
        # Normalize the true tempo for the target
        true_tempo_norm = (true_tempo - 78) / 77
        target = torch.tensor([true_tempo_norm], dtype=torch.float32)
        
        return features, target
class AugmentedRekordboxDataset(Dataset):
    def __init__(self, base_dataset, augmentations_per_sample=4):
        print('augmenting data....')
        # Handle both Dataset and Subset objects
        if isinstance(base_dataset, Subset):
            print("base dataset is a subset")
            self.base_dataset = base_dataset.dataset
            self.indices = base_dataset.indices
        else:
            print(base_dataset.feature_length)
            self.base_dataset = base_dataset
            self.indices = range(len(base_dataset))
            
        self.augmentations_per_sample = augmentations_per_sample
        self.feature_length = base_dataset.feature_length
        self.feature_shape = (15, self.feature_length)
        
    def __len__(self):
        return len(self.indices) * self.augmentations_per_sample

    def __getitem__(self, idx):
        original_idx = self.indices[idx // self.augmentations_per_sample]
        features, target = self.base_dataset[original_idx]
        original_tempo_norm = features[0, 0].item()  # Get the original tempo feature

        if idx % self.augmentations_per_sample == 0:
            return features, target

        features_np = features.numpy()
        aug_type = idx % self.augmentations_per_sample

        if aug_type == 1:
            features_np, new_tempo_norm, stretch_factor = self._time_stretch(features_np, original_tempo_norm)
            
            # Adjust the target tempo
            # Convert normalized target back to BPM
            target_bpm = target.item() * 77 + 78
            # Adjust tempo BPM according to stretch factor
            new_target_bpm = target_bpm / stretch_factor
            # Normalize the adjusted BPM back to [0, 1]
            new_target_norm = (new_target_bpm - 78) / 77
            target = torch.tensor([new_target_norm], dtype=torch.float32)

        elif aug_type == 2:
            features_np = self._mask_augment(features_np)

        elif aug_type == 3:
            features_np = self._add_noise(features_np)

        return torch.FloatTensor(features_np), target

    def _time_stretch(self, features, original_tempo_norm):
        stretch_factor = np.random.uniform(0.95, 1.05)
        dynamic_features = features[1:]  # Exclude tempo feature

        # Convert original tempo feature back to BPM
        tempo_bpm = original_tempo_norm * 77 + 78
        # Adjust tempo BPM according to stretch factor
        new_tempo_bpm = tempo_bpm / stretch_factor
        # Normalize the adjusted BPM back to [0, 1]
        new_tempo_norm = (new_tempo_bpm - 78) / 77
        features[0] = new_tempo_norm

        time_axis = self.feature_length
        stretched_length = int(time_axis * stretch_factor)

        stretched = np.zeros((features.shape[0], time_axis))
        stretched[0] = features[0]  # Updated tempo feature

        indices = np.linspace(0, time_axis - 1, stretched_length)
        for i in range(dynamic_features.shape[0]):
            # Time-stretch dynamic features
            temp = np.interp(indices, np.arange(time_axis), dynamic_features[i])
            stretched[i + 1] = np.interp(np.arange(time_axis), np.arange(stretched_length), temp)

        return stretched, new_tempo_norm, stretch_factor

    def _mask_augment(self, features):
        features = features.copy()
        mask_size = random.randint(32, min(128, self.feature_length))
        mask_start = random.randint(0, self.feature_length - mask_size)
        features[1:, mask_start:mask_start + mask_size] = 0  # Mask dynamic features
        return features

    def _add_noise(self, features):
        noise = np.random.normal(0, 0.01, (features.shape[0], self.feature_length))
        noise[0] = 0  # Don't add noise to tempo feature
        return features + noise