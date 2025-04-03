# SPDX-FileCopyrightText: 2024 Idiap Research Institute
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: MIT

import os

import torch
from trainer import Trainer, TrainerArgs
from TTS.config import BaseAudioConfig
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from knn_tts.tts.GlowTTS.glow_tts_ssl import GlowTTSSSL
from knn_tts.utils import get_vocoder_checkpoint_path

SAMPLE_RATE = 16000
RUN_NAME = "glow_tts_ssl_ljspeech"
LANGUAGE = "en"
SSL_MODEL = "wavlm"
SSL_DIM = 1024
SEGMENT_SIZE = 25  # ms
HOP_LEN = 20  # ms
CHECKPOINTS_DIR = "checkpoints"
HIFIGAN_CHECKPOINT_PATH = get_vocoder_checkpoint_path(CHECKPOINTS_DIR)
DATASET_PATH = "knn-tts/datasets/ljspeech_ssl"
OUTPUT_PATH = f"knn-tts/outputs/glow_tts_ssl/{SSL_MODEL}/ljspeech"


def knn_tts_formatter(root_path, meta_files=None, ignored_speakers=None):  # pylint: disable=unused-argument
    root_path = os.path.normpath(root_path)
    csv_file = os.path.join(root_path, "metadata.csv")
    wavs_path = os.path.join(root_path, "wavs")
    ssl_feats_path = os.path.join(root_path, SSL_MODEL)

    items = []

    print(f" | > {csv_file}")

    with open(csv_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_name = cols[0] + ".wav"
            ssl_feats_name = cols[0] + ".pt"
            wav_file = os.path.join(wavs_path, wav_name)
            ssl_feats_file = os.path.join(ssl_feats_path, ssl_feats_name)
            if os.path.isfile(wav_file):
                text = cols[-1].strip()
                items.append(
                    {
                        "text": text,
                        "audio_file": wav_file,
                        "ssl_feats_file": ssl_feats_file,
                        "root_path": root_path,
                        "speaker_name": None,
                    }
                )
            else:
                print(f"> File {wav_file} does not exist!")

    return items


dataset_config = BaseDatasetConfig(formatter="", meta_file_train=None, path=DATASET_PATH, language=LANGUAGE)

audio_config = BaseAudioConfig(
    sample_rate=SAMPLE_RATE,
    win_length=int(SEGMENT_SIZE * 1e-3 * SAMPLE_RATE),
    hop_length=int(HOP_LEN * 1e-3 * SAMPLE_RATE),
)

config = GlowTTSConfig(
    run_name=RUN_NAME,
    audio=audio_config,
    out_channels=SSL_DIM,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=2,
    num_eval_loader_workers=2,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(OUTPUT_PATH, "phoneme_cache"),
    print_step=25,
    print_eval=True,
    mixed_precision=torch.cuda.is_available(),
    test_sentences=[
        "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
        "It sounds plausible enough tonight, but wait until tomorrow. Wait for the common sense of the morning.",
    ],
    output_path=OUTPUT_PATH,
    datasets=dataset_config,
)

ap = AudioProcessor.init_from_config(config)

tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    formatter=knn_tts_formatter,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

model = GlowTTSSSL(
    config,
    ap,
    tokenizer,
    speaker_manager=None,
    hifigan_checkpoint_path=HIFIGAN_CHECKPOINT_PATH,
)

trainer = Trainer(
    TrainerArgs(),
    config,
    OUTPUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()
