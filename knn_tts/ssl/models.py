# SPDX-FileCopyrightText: 2024 Idiap Research Institute
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: MIT

import torch
import torchaudio

from knn_tts.ssl.WavLM.WavLM import WavLMConfig, WavLMModel


class WavLM:
    def __init__(self, device=None):
        """
        Load the WavLM-Large model (see: https://github.com/microsoft/unilm/tree/master/wavlm).
        Torch hub checkpoint from https://github.com/bshall/knn-vc
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        checkpoint_url = "https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt"
        checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, map_location=self.device, progress=True)

        cfg = WavLMConfig(checkpoint["cfg"])
        self.model = WavLMModel(cfg)
        self.model.load_state_dict(checkpoint["model"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.model.sample_rate = 16000
        self.SPEAKER_INFORMATION_LAYER = 6
        num_parameters = sum(p.numel() for p in self.model.parameters())
        print(f"WavLM-Large loaded with {num_parameters:,d} parameters.")

    @torch.no_grad()
    def extract_framewise_features(self, wav_path, output_layer=None):
        audio, orig_sr = torchaudio.load(wav_path)
        audio = audio.to(self.device)
        if orig_sr != self.model.sample_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=self.model.sample_rate)

        if output_layer is None:
            output_layer = self.SPEAKER_INFORMATION_LAYER

        embeddings = self.model.extract_features(audio, output_layer=output_layer, ret_layer_results=False)[0]
        embeddings = embeddings.squeeze(0)
        return embeddings  # [frame_len, 1024]
