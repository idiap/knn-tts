# SPDX-FileCopyrightText: 2024 Idiap Research Institute
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: MIT

import os
import sys

import torch

from knn_tts.ssl.models import WavLM
from knn_tts.utils import find_wav_paths


def load_model(model_name):
    if model_name == "wavlm":
        model = WavLM()
    else:
        raise NameError("Invalid model name")
    return model


def rename_embeddings_dir_in_path(output_path, model_name):
    directory, filename = os.path.split(output_path)
    dir_parts = directory.split(os.path.sep)
    if dir_parts[-1] == "wavs":
        dir_parts[-1] = model_name
    else:
        dir_parts[-1] = f"{dir_parts[-1]}-{model_name}"
    new_directory = os.path.sep.join(dir_parts)
    new_output_path = os.path.join(new_directory, filename)
    return new_output_path


def save_embeddings(embeddings, input_path, output_root, dataset_root, model_name):
    relative_path = os.path.relpath(input_path, dataset_root)
    output_path = os.path.join(output_root, relative_path.replace(".wav", ".pt"))
    output_path = rename_embeddings_dir_in_path(output_path, model_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"Saved {output_path}")


def main(model_name, dataset_path, output_path):
    model = load_model(model_name)
    wav_paths = find_wav_paths(dataset_path)
    for wav_path in wav_paths:
        with torch.no_grad():
            embeddings = model.extract_framewise_features(wav_path).cpu()
        save_embeddings(embeddings, wav_path, output_path, dataset_path, model_name)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_dataset_embeddings.py [model_name] [dataset_path] [output_path]")
        sys.exit(1)
    MODEL_NAME = sys.argv[1]
    DATASET_PATH = sys.argv[2]
    OUTPUT_PATH = sys.argv[3]
    main(MODEL_NAME, DATASET_PATH, OUTPUT_PATH)
