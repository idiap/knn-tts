# SPDX-FileCopyrightText: 2024 Idiap Research Institute
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: MIT

import json

import torch

from knn_tts.vocoder.hifigan.models import Generator as HiFiGAN
from knn_tts.vocoder.hifigan.utils import AttrDict


class HiFiGANWavLM:
    def __init__(
        self,
        checkpoint_path,
        config_path="knn_tts/vocoder/hifigan/config_v1_wavlm.json",
        device=None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        with open(config_path, encoding="utf-8") as f:
            data = f.read()
        json_config = json.loads(data)
        self.h = AttrDict(json_config)

        self.generator = HiFiGAN(self.h).to(self.device)

        state_dict_g = torch.load(checkpoint_path, weights_only=False, map_location=device)
        self.generator.load_state_dict(state_dict_g["generator"])
        self.generator.eval()
        self.generator.remove_weight_norm()
        num_parameters = sum(p.numel() for p in self.generator.parameters())
        print(f"[HiFiGAN] Generator loaded with {num_parameters:,d} parameters.")

    def __call__(self, x):
        with torch.no_grad():
            wav = self.generator(x.to(self.device))
            wav = wav.squeeze(1)
            wav = wav.cpu().squeeze()
        return wav
