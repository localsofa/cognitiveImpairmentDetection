import librosa
import numpy as np
import pandas as pd

# https://docs.pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html
# data: https://github.com/numediart/EmoV-DB

# load data
import os
base_path = r"C:\Users\erwin\OneDrive\Desktop\uni\persönliche projekte\cognitiveImpairmentDetection\data"

def load_emov_db(base_path):
    data = []

    for label_name, label in [("normal", 0), ("delir", 1)]:
        folder = os.path.join(base_path, label_name)

        for file in os.listdir(folder):
            if file.endswith(".wav"):
                data.append({
                    "audio_path": os.path.join(folder, file),
                    "label": label
                })

    return data


# feature extraction
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    # pitch
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # energy
    rms = np.mean(librosa.feature.rms(y=y))

    # zero crossing rate (speech irregularity)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    tempo = librosa.feature.tempo(y=y, sr=sr)

    # add speech rate; pauses (length of silences; amount); articulation rate 

    # return all features
    return {
        "mfcc_" + str(i): mfccs_mean[i] for i in range(len(mfccs_mean))
    } | {
        "spectral_centroid": spectral_centroid,
        "rms": rms,
        "zcr": zcr,
        "tempro": tempo[0]
    }

# build dataset
dataset = load_emov_db(base_path)

rows = []

for sample in dataset:
    feats = extract_audio_features(sample["audio_path"])
    feats["label"] = sample["label"]
    rows.append(feats)

df = pd.DataFrame(rows)
print(df.head())