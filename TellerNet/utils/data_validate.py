import os, re
import numpy as np

def validate_dataset(audio_dir):
    tempo_pattern = re.compile(r'^(\d+\.?\d*)\s+(.+)$')
    tempos = []
    invalid_files = []
    
    for filename in os.listdir(audio_dir):
        if filename.endswith(('.mp3', '.wav', '.aiff', '.m4a')):
            match = tempo_pattern.match(filename)
            if match:
                tempo = float(match.group(1))
                tempos.append(tempo)
            else:
                invalid_files.append(filename)
    
    print("\nDataset Statistics:")
    print(f"Total audio files found: {len(tempos) + len(invalid_files)}")
    print(f"Valid labeled files: {len(tempos)}")
    print(f"Invalid files: {len(invalid_files)}")
    
    if tempos:
        print(f"\nTempo Distribution:")
        print(f"Min tempo: {min(tempos):.1f} BPM")
        print(f"Max tempo: {max(tempos):.1f} BPM")
        print(f"Mean tempo: {np.mean(tempos):.1f} BPM")
        print(f"Median tempo: {np.median(tempos):.1f} BPM")
    
    if invalid_files:
        print("\nFiles missing tempo labels:")
        for file in invalid_files[:5]:  # Show first 5 invalid files
            print(f"- {file}")
        if len(invalid_files) > 5:
            print(f"...and {len(invalid_files) - 5} more")
    
    return len(tempos) > 0