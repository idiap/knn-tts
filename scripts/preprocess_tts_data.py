# SPDX-FileCopyrightText: 2024 Idiap Research Institute
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: MIT

import os
import sys

import numpy as np
import webrtcvad
from pydub import AudioSegment

from knn_tts.utils import find_wav_paths


def save_audio(audio, input_path, output_root, dataset_root):
    relative_path = os.path.relpath(input_path, dataset_root)
    output_path = os.path.join(output_root, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio.export(output_path, format="wav")


def get_speech_segments_webrtcvad(audio_array, sample_rate, frame_duration, vad_mode):
    vad = webrtcvad.Vad(vad_mode)

    # Convert the frame duration to samples
    frame_duration_samples = int(sample_rate * frame_duration / 1000)

    # Detect speech regions using VAD
    speech_segments = []
    start = -1
    for i in range(0, len(audio_array), frame_duration_samples):
        frame = audio_array[i : i + frame_duration_samples]

        if len(frame) < 160:
            is_speech = False
        else:
            frame = frame.tobytes()
            is_speech = vad.is_speech(frame, sample_rate)

        if is_speech and start == -1:
            start = i
        elif not is_speech and start != -1:
            end = i
            speech_segments.append((start, end))
            start = -1

    return speech_segments


def get_start_end_using_vad(audio, sample_rate):
    audio_array = np.array(audio.get_array_of_samples())

    speech_segments = get_speech_segments_webrtcvad(audio_array, sample_rate, VAD_FRAME_DURATION, VAD_MODE)
    if len(speech_segments) == 0:
        speech_segments = get_speech_segments_webrtcvad(audio_array, sample_rate, VAD_FRAME_DURATION, VAD_MODE - 1)

    start_sample = speech_segments[0][0]
    end_sample = speech_segments[-1][1]

    start_time = float(start_sample / VAD_SR)
    end_time = float(end_sample / VAD_SR)

    return start_time, end_time


def trim_silences(audio, target_sr):
    audio_copy = audio[:]

    audio_copy = audio_copy.set_frame_rate(VAD_SR)

    start_time, end_time = get_start_end_using_vad(audio_copy, VAD_SR)

    start_sample_orig_sr = int(start_time * target_sr)
    end_sample_orig_sr = int(end_time * target_sr)

    filtered_audio_array = np.array(audio.get_array_of_samples())
    filtered_audio_array = filtered_audio_array[start_sample_orig_sr:end_sample_orig_sr]

    filtered_audio = AudioSegment(
        filtered_audio_array.tobytes(),
        frame_rate=target_sr,
        sample_width=audio.sample_width,
        channels=audio.channels,
    )

    return filtered_audio


def match_target_amplitude(audio, target_dBFS):
    change_in_dBFS = target_dBFS - audio.dBFS
    return audio.apply_gain(change_in_dBFS)


def preprocess_wav(wav_path, target_sr, do_trim_silences=True):
    audio = AudioSegment.from_file(wav_path)

    # Convert audio to mono
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Resample audio
    audio = audio.set_frame_rate(target_sr)

    # Convert the audio to 16-bit PCM format
    audio = audio.set_sample_width(2)

    # Remove silences
    if do_trim_silences:
        audio = trim_silences(audio, target_sr)

    # Loudness normalization to -20dB
    audio = match_target_amplitude(audio, -20.0)

    return audio


def main(target_sr, dataset_path, output_path):
    wav_paths = find_wav_paths(dataset_path)
    for wav_path in wav_paths:
        processed_audio = preprocess_wav(wav_path, target_sr)
        save_audio(processed_audio, wav_path, output_path, dataset_path)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python preprocess_tts_data.py [target_sr] [dataset_path] [output_path]")
        sys.exit(1)
    TARGET_SR = int(sys.argv[1])
    DATASET_PATH = sys.argv[2]
    OUTPUT_PATH = sys.argv[3]

    # Parameters for VAD
    VAD_SR = 16000
    VAD_MODE = 2  # Aggressiveness level (0-3, where 3 is the most aggressive)
    VAD_FRAME_DURATION = 10  # Frame duration in milliseconds

    main(TARGET_SR, DATASET_PATH, OUTPUT_PATH)
