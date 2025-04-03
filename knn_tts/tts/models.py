# SPDX-FileCopyrightText: 2024 Idiap Research Institute
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: MIT

import torch

from knn_tts.tts.GlowTTS.glow_tts_ssl import load_glow_tts_ssl, ssl_synthesis


class GlowTTS:
    def __init__(
        self,
        tts_model_base_path,
        tts_model_checkpoint,
        vocoder_checkpoint_path=None,
        device=None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = load_glow_tts_ssl(
            tts_model_base_path,
            tts_model_checkpoint,
            vocoder_checkpoint_path,
            self.device,
        )

    def __call__(self, text, *args, **kwargs):
        with torch.no_grad():
            ssl_feats_out = ssl_synthesis(self.model, text, None, return_ssl_feats=True)
        return ssl_feats_out
