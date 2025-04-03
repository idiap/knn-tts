# SPDX-FileCopyrightText: 2024 Idiap Research Institute
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: MIT

import os

import requests


def find_wav_paths(root_dir):
    wav_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_paths.append(os.path.join(subdir, file))
    return wav_paths

def get_vocoder_checkpoint_path(checkpoints_dir):
    os.makedirs(checkpoints_dir, exist_ok=True)  # Ensure directory exists

    checkpoint_path = os.path.join(checkpoints_dir, "prematch_g_02500000.pt")
    url = "https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt"

    if not os.path.exists(checkpoint_path):
        print(f"Downloading checkpoint to {checkpoint_path}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(checkpoint_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        else:
            raise Exception(f"Failed to download checkpoint: {response.status_code}")

    return checkpoint_path
