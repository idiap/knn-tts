# SPDX-FileCopyrightText: 2024 Idiap Research Institute
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: MIT

import torch
import torchaudio

from knn_tts.text_cleaners import clean_input_text
from knn_tts.tts.models import GlowTTS
from knn_tts.vc.knn import knn_vc, load_target_style_feats
from knn_tts.vocoder.models import HiFiGANWavLM


class Synthesizer:
    def __init__(
        self,
        tts_model_base_path,
        tts_model_checkpoint,
        vocoder_checkpoint_path,
        model_name="glowtts",
    ):
        self.model_name = model_name
        self.model = GlowTTS(tts_model_base_path, tts_model_checkpoint)
        self.vocoder = HiFiGANWavLM(checkpoint_path=vocoder_checkpoint_path, device=self.model.device)
        self.target_style_feats_path = None
        self.target_style_feats = None

    def __call__(
        self,
        text_input,
        target_style_feats_path,
        knnvc_topk=4,
        weighted_average=False,
        interpolation_rate=1.0,
        save_path=None,
        timesteps=10,
        max_target_num_files=1000,
    ):
        with torch.no_grad():
            # Text-to-SSL
            text_input = clean_input_text(text_input)
            tts_feats = self.model(text_input, timesteps)  # timesteps are used for GradTTS only

            # kNN-VC
            if interpolation_rate != 0.0:

                if target_style_feats_path != self.target_style_feats_path:
                    self.target_style_feats_path = target_style_feats_path
                    self.target_style_feats = load_target_style_feats(target_style_feats_path, max_target_num_files)

                selected_feats = knn_vc(
                    tts_feats,
                    self.target_style_feats,
                    topk=knnvc_topk,
                    weighted_average=weighted_average,
                    device=self.model.device,
                )

                converted_feats = interpolation_rate * selected_feats + (1.0 - interpolation_rate) * tts_feats
            else:
                converted_feats = tts_feats

            # Vocoder
            wav = self.vocoder(converted_feats.unsqueeze(0)).unsqueeze(0)

        if save_path is not None:
            torchaudio.save(save_path, wav, 16000)
        return wav
