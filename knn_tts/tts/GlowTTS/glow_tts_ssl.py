# SPDX-FileCopyrightText: Coqui AI
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: MPL-2.0

import collections
import os

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.datasets.dataset import TTSDataset, noise_augment_audio
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.data import prepare_data, prepare_stop_target, prepare_tensor
from TTS.tts.utils.synthesis import (
    embedding_to_torch,
    id_to_torch,
    numpy_to_torch,
    run_model_torch,
    trim_silence,
)
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.audio import AudioProcessor

from knn_tts.vocoder.models import HiFiGANWavLM


def load_glow_tts_ssl(model_base_path, checkpoint, hifigan_checkpoint_path, device):
    checkpoint_path = os.path.join(model_base_path, checkpoint)
    config_path = os.path.join(model_base_path, "config.json")

    config = GlowTTSConfig()
    config.load_json(config_path)
    ap = AudioProcessor.init_from_config(config)
    tokenizer, _ = TTSTokenizer.init_from_config(config)

    model = GlowTTSSSL(
        config=config,
        ap=ap,
        tokenizer=tokenizer,
        speaker_manager=None,
        hifigan_checkpoint_path=hifigan_checkpoint_path,
    )
    model.load_checkpoint(config=config, checkpoint_path=checkpoint_path)
    model.to(device)
    model.eval()
    return model


def ssl_synthesis(
    model,
    text,
    vocoder,
    return_ssl_feats=False,
    speaker_id=None,
    do_trim_silence=False,
    d_vector=None,
    language_id=None,
):
    # device
    device = next(model.parameters()).device

    language_name = None
    if language_id is not None:
        language = [k for k, v in model.language_manager.name_to_id.items() if v == language_id]
        assert len(language) == 1, "language_id must be a valid language"
        language_name = language[0]

    # convert text to sequence of token IDs
    text_inputs = np.asarray(
        model.tokenizer.text_to_ids(text, language=language_name),
        dtype=np.int32,
    )

    # pass tensors to backend
    if speaker_id is not None:
        speaker_id = id_to_torch(speaker_id, device=device)

    if d_vector is not None:
        d_vector = embedding_to_torch(d_vector, device=device)

    if language_id is not None:
        language_id = id_to_torch(language_id, device=device)

    text_inputs = numpy_to_torch(text_inputs, torch.long, device=device)
    text_inputs = text_inputs.unsqueeze(0)
    # synthesize voice
    outputs = run_model_torch(
        model,
        text_inputs,
        speaker_id,
        style_mel=None,
        style_text=None,
        d_vector=d_vector,
        language_id=language_id,
    )
    model_outputs = outputs["model_outputs"]

    if return_ssl_feats:
        ssl_feats = model_outputs.squeeze()
        return ssl_feats

    model_outputs = model_outputs[0].data.cpu().numpy()
    alignments = outputs["alignments"]

    ssl_feats = model_outputs.squeeze()
    wav = vocoder(torch.from_numpy(ssl_feats).unsqueeze(0)).unsqueeze(0).cpu().numpy()

    if do_trim_silence:
        wav = trim_silence(wav, model.ap)

    return_dict = {
        "wav": wav,
        "alignments": alignments,
        "text_inputs": text_inputs,
        "outputs": outputs,
    }
    return return_dict


class GlowTTSSSLDataset(TTSDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_id = self.tokenizer.characters.pad_id

    def load_data(self, idx):
        item = self.samples[idx]

        raw_text = item["text"]

        wav = np.asarray(self.load_wav(item["audio_file"]), dtype=np.float32)

        # apply noise for augmentation
        if self.use_noise_augment:
            wav = noise_augment_audio(wav)

        # get token ids
        token_ids = self.get_token_ids(idx, item["text"])

        # get pre-computed attention maps
        attn = None
        if "alignment_file" in item:
            attn = self.get_attn_mask(item["alignment_file"])

        # after phonemization the text length may change
        # this is a shareful 🤭 hack to prevent longer phonemes
        # TODO: find a better fix # pylint: disable=fixme
        if len(token_ids) > self.max_text_len or len(wav) < self.min_audio_len:
            self.rescue_item_idx += 1
            return self.load_data(self.rescue_item_idx)

        # get f0 values
        f0 = None
        if self.compute_f0:
            f0 = self.get_f0(idx)["f0"]
        energy = None
        if self.compute_energy:
            energy = self.get_energy(idx)["energy"]

        ssl_feats = torch.load(item["ssl_feats_file"])

        if len(ssl_feats.size()) == 2:
            ssl_feats = ssl_feats.unsqueeze(0)

        sample = {
            "raw_text": raw_text,
            "token_ids": token_ids,
            "wav": wav,
            "pitch": f0,
            "energy": energy,
            "attn": attn,
            "item_idx": item["audio_file"],
            "speaker_name": item["speaker_name"],
            "language_name": item["language"],
            "wav_file_name": os.path.basename(item["audio_file"]),
            "audio_unique_name": item["audio_unique_name"],
            "ssl_feats": ssl_feats,
        }
        return sample

    # pylint: disable=too-many-branches, too-many-statements
    def collate_fn(self, batch):
        # Puts each data field into a tensor with outer dimension batch size
        if isinstance(batch[0], collections.abc.Mapping):
            B = len(batch)

            token_ids_lengths = np.array([len(d["token_ids"]) for d in batch])

            # sort items with text input length for RNN efficiency
            batch, token_ids_lengths, ids_sorted_decreasing = self._sort_batch(batch, token_ids_lengths)

            # convert list of dicts to dict of lists
            batch = {k: [dic[k] for dic in batch] for k in batch[0]}

            # get language ids from language names
            if self.language_id_mapping is not None:
                language_ids = [self.language_id_mapping[ln] for ln in batch["language_name"]]
            else:
                language_ids = None
            # get pre-computed d-vectors
            if self.d_vector_mapping is not None:
                embedding_keys = list(batch["audio_unique_name"])
                d_vectors = [self.d_vector_mapping[w]["embedding"] for w in embedding_keys]
            else:
                d_vectors = None

            # get numerical speaker ids from speaker names
            if self.speaker_id_mapping:
                speaker_ids = [self.speaker_id_mapping[sn] for sn in batch["speaker_name"]]
            else:
                speaker_ids = None
            # compute features
            mel = [self.ap.melspectrogram(w).astype("float32") for w in batch["wav"]]

            mel_lengths = [m.shape[1] for m in mel]

            # lengths adjusted by the reduction factor
            mel_lengths_adjusted = [m.shape[1] + (self.outputs_per_step - (m.shape[1] % self.outputs_per_step)) if m.shape[1] % self.outputs_per_step else m.shape[1] for m in mel]

            # PAD sequences with longest instance in the batch
            token_ids = prepare_data(batch["token_ids"]).astype(np.int32)

            # PAD features with longest instance
            mel = prepare_tensor(mel, self.outputs_per_step)

            # B x D x T --> B x T x D
            mel = mel.transpose(0, 2, 1)

            # convert things to pytorch
            token_ids_lengths = torch.LongTensor(token_ids_lengths)
            token_ids = torch.LongTensor(token_ids)
            mel = torch.FloatTensor(mel).contiguous()
            mel_lengths = torch.LongTensor(mel_lengths)

            # format ssl_feats
            ssl_feats_dim = batch["ssl_feats"][0].shape[2]
            ssl_feats_lens = [emb.shape[1] for emb in batch["ssl_feats"]]
            ssl_feats_lens = torch.LongTensor(ssl_feats_lens)
            ssl_feats_lens_max = torch.max(ssl_feats_lens)
            ssl_feats_rel_lens = ssl_feats_lens / ssl_feats_lens_max
            ssl_feats_padded = torch.FloatTensor(B, ssl_feats_lens_max, ssl_feats_dim)
            ssl_feats_padded = ssl_feats_padded.zero_() + self.pad_id

            for i, emb in enumerate(batch["ssl_feats"]):
                ssl_feats_padded[i, : emb.size(1), :] = torch.FloatTensor(emb)

            stop_targets = [np.array([0.0] * (ssl_len - 1) + [1.0]) for ssl_len in ssl_feats_lens]
            stop_targets = prepare_stop_target(stop_targets, self.outputs_per_step)
            stop_targets = torch.FloatTensor(stop_targets)

            # speaker vectors
            if d_vectors is not None:
                d_vectors = torch.FloatTensor(d_vectors)

            if speaker_ids is not None:
                speaker_ids = torch.LongTensor(speaker_ids)

            if language_ids is not None:
                language_ids = torch.LongTensor(language_ids)

            # compute linear spectrogram
            linear = None
            if self.compute_linear_spec:
                linear = [self.ap.spectrogram(w).astype("float32") for w in batch["wav"]]
                linear = prepare_tensor(linear, self.outputs_per_step)
                linear = linear.transpose(0, 2, 1)
                assert mel.shape[1] == linear.shape[1]
                linear = torch.FloatTensor(linear).contiguous()

            # format waveforms
            wav_padded = None
            if self.return_wav:
                wav_lengths = [w.shape[0] for w in batch["wav"]]
                # max_wav_len = max(mel_lengths_adjusted) * self.ap.hop_length
                ssl_feats_hop_length = self.ap.hop_length
                max_wav_len = max(ssl_feats_lens) * ssl_feats_hop_length
                wav_lengths = torch.LongTensor(wav_lengths)
                wav_padded = torch.zeros(len(batch["wav"]), 1, max_wav_len)
                for i, w in enumerate(batch["wav"]):
                    mel_length = mel_lengths_adjusted[i]
                    w = np.pad(w, (0, self.ap.hop_length * self.outputs_per_step), mode="edge")
                    w = w[: mel_length * self.ap.hop_length]
                    wav_padded[i, :, : w.shape[0]] = torch.from_numpy(w)
                wav_padded.transpose_(1, 2)

            # format F0
            if self.compute_f0:
                pitch = prepare_data(batch["pitch"])
                assert mel.shape[1] == pitch.shape[1], f"[!] {mel.shape} vs {pitch.shape}"
                pitch = torch.FloatTensor(pitch)[:, None, :].contiguous()  # B x 1 xT
            else:
                pitch = None
            # format energy
            if self.compute_energy:
                energy = prepare_data(batch["energy"])
                assert mel.shape[1] == energy.shape[1], f"[!] {mel.shape} vs {energy.shape}"
                energy = torch.FloatTensor(energy)[:, None, :].contiguous()  # B x 1 xT
            else:
                energy = None
            # format attention masks
            attns = None
            if batch["attn"][0] is not None:
                attns = [batch["attn"][idx].T for idx in ids_sorted_decreasing]
                for idx, attn in enumerate(attns):
                    pad2 = mel.shape[1] - attn.shape[1]
                    pad1 = token_ids.shape[1] - attn.shape[0]
                    assert pad1 >= 0 and pad2 >= 0, f"[!] Negative padding - {pad1} and {pad2}"
                    attn = np.pad(attn, [[0, pad1], [0, pad2]])
                    attns[idx] = attn
                attns = prepare_tensor(attns, self.outputs_per_step)
                attns = torch.FloatTensor(attns).unsqueeze(1)

            return {
                "token_id": token_ids,
                "token_id_lengths": token_ids_lengths,
                "speaker_names": batch["speaker_name"],
                "linear": linear,
                "mel": mel,
                "mel_lengths": mel_lengths,
                "stop_targets": stop_targets,
                "item_idxs": batch["item_idx"],
                "d_vectors": d_vectors,
                "speaker_ids": speaker_ids,
                "attns": attns,
                "waveform": wav_padded,
                "raw_text": batch["raw_text"],
                "pitch": pitch,
                "energy": energy,
                "language_ids": language_ids,
                "audio_unique_names": batch["audio_unique_name"],
                "ssl_feats": ssl_feats_padded,
                "ssl_feats_lens": ssl_feats_lens,
                "ssl_feats_rel_lens": ssl_feats_rel_lens,
            }
        return None


class GlowTTSSSL(GlowTTS):
    def __init__(
        self,
        config: GlowTTSConfig,
        ap=None,
        tokenizer=None,
        speaker_manager=None,
        hifigan_checkpoint_path: str = "checkpoints/hifigan-wavlm/prematch_g_02500000.pt",
    ):
        super().__init__(config, ap, tokenizer, speaker_manager)
        if hifigan_checkpoint_path is None:
            self.vocoder = None
        else:
            self.vocoder = HiFiGANWavLM(
                checkpoint_path=hifigan_checkpoint_path,
                device=str(next(self.parameters()).device),
            )

    def train_step(self, batch: dict, criterion: nn.Module):
        text_input = batch["text_input"]
        text_lengths = batch["text_lengths"]
        ssl_feats_input = batch["ssl_feats"]
        ssl_feats_lengths = batch["ssl_feats_lens"]
        d_vectors = batch["d_vectors"]
        speaker_ids = batch["speaker_ids"]

        if self.run_data_dep_init and self.training:
            # compute data-dependent initialization of activation norm layers
            self.unlock_act_norm_layers()
            with torch.no_grad():
                _ = self.forward(
                    text_input,
                    text_lengths,
                    ssl_feats_input,
                    ssl_feats_lengths,
                    aux_input={"d_vectors": d_vectors, "speaker_ids": speaker_ids},
                )
            outputs = None
            loss_dict = None
            self.lock_act_norm_layers()
        else:
            # normal training step
            outputs = self.forward(
                text_input,
                text_lengths,
                ssl_feats_input,
                ssl_feats_lengths,
                aux_input={"d_vectors": d_vectors, "speaker_ids": speaker_ids},
            )

            with autocast(enabled=False):  # avoid mixed_precision in criterion
                loss_dict = criterion(
                    outputs["z"].float(),
                    outputs["y_mean"].float(),
                    outputs["y_log_scale"].float(),
                    outputs["logdet"].float(),
                    ssl_feats_lengths,
                    outputs["durations_log"].float(),
                    outputs["total_durations_log"].float(),
                    text_lengths,
                )

        return outputs, loss_dict

    @torch.no_grad()
    def _create_logs(self, batch, outputs, ap):
        alignments = outputs["alignments"]
        text_input = batch["text_input"][:1] if batch["text_input"] is not None else None
        text_lengths = batch["text_lengths"]
        mel_input = batch["mel_input"]
        d_vectors = batch["d_vectors"][:1] if batch["d_vectors"] is not None else None
        speaker_ids = batch["speaker_ids"][:1] if batch["speaker_ids"] is not None else None

        gt_spec = mel_input[0].data.cpu().numpy()
        align_img = alignments[0].data.cpu().numpy()

        # model runs reverse flow to predict spectrograms
        with torch.no_grad():
            pred_outputs = self.inference(
                text_input,
                aux_input={
                    "x_lengths": text_lengths[:1],
                    "d_vectors": d_vectors,
                    "speaker_ids": speaker_ids,
                },
            )

            pred_ssl_feats = pred_outputs["model_outputs"]

            pred_audio = self.vocoder(pred_ssl_feats).unsqueeze(0).cpu().detach().numpy()

        pred_spec = self.ap.melspectrogram(pred_audio).astype("float32")
        pred_spec = pred_spec.squeeze(axis=1)
        pred_spec = pred_spec.T

        figures = {
            "prediction": plot_spectrogram(pred_spec, ap, output_fig=False),
            "ground_truth": plot_spectrogram(gt_spec, ap, output_fig=False),
            "alignment": plot_alignment(align_img, output_fig=False),
        }

        return figures, {"audio": pred_audio}

    def get_data_loader(
        self,
        config,
        assets,
        is_eval,
        samples,
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":
        if is_eval and not config.run_eval:
            loader = None
        else:
            # setup multi-speaker attributes
            if self.speaker_manager is not None:
                if hasattr(config, "model_args"):
                    speaker_id_mapping = self.speaker_manager.name_to_id if config.model_args.use_speaker_embedding else None
                    d_vector_mapping = self.speaker_manager.embeddings if config.model_args.use_d_vector_file else None
                    config.use_d_vector_file = config.model_args.use_d_vector_file
                else:
                    speaker_id_mapping = self.speaker_manager.name_to_id if config.use_speaker_embedding else None
                    d_vector_mapping = self.speaker_manager.embeddings if config.use_d_vector_file else None
            else:
                speaker_id_mapping = None
                d_vector_mapping = None

            # setup multi-lingual attributes
            if self.language_manager is not None:
                language_id_mapping = self.language_manager.name_to_id if self.args.use_language_embedding else None
            else:
                language_id_mapping = None

            # init dataloader
            dataset = GlowTTSSSLDataset(
                outputs_per_step=config.r if "r" in config else 1,
                compute_linear_spec=config.model.lower() == "tacotron" or config.compute_linear_spec,
                compute_f0=config.get("compute_f0", False),
                f0_cache_path=config.get("f0_cache_path", None),
                compute_energy=config.get("compute_energy", False),
                energy_cache_path=config.get("energy_cache_path", None),
                samples=samples,
                ap=self.ap,
                return_wav=config.return_wav if "return_wav" in config else False,
                batch_group_size=0 if is_eval else config.batch_group_size * config.batch_size,
                min_text_len=config.min_text_len,
                max_text_len=config.max_text_len,
                min_audio_len=config.min_audio_len,
                max_audio_len=config.max_audio_len,
                phoneme_cache_path=config.phoneme_cache_path,
                precompute_num_workers=config.precompute_num_workers,
                use_noise_augment=False if is_eval else config.use_noise_augment,
                speaker_id_mapping=speaker_id_mapping,
                d_vector_mapping=d_vector_mapping if config.use_d_vector_file else None,
                tokenizer=self.tokenizer,
                start_by_longest=config.start_by_longest,
                language_id_mapping=language_id_mapping,
            )

            # wait all the DDP process to be ready
            if num_gpus > 1:
                dist.barrier()

            # sort input sequences from short to long
            dataset.preprocess_samples()

            # get samplers
            sampler = self.get_sampler(config, dataset, num_gpus)

            loader = DataLoader(
                dataset,
                batch_size=config.eval_batch_size if is_eval else config.batch_size,
                shuffle=config.shuffle if sampler is None else False,  # if there is no other sampler
                collate_fn=dataset.collate_fn,
                drop_last=config.drop_last,  # setting this False might cause issues in AMP training.
                sampler=sampler,
                num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                pin_memory=False,
            )
        return loader

    def format_batch(self, batch):
        # setup input batch
        text_input = batch["token_id"]
        text_lengths = batch["token_id_lengths"]
        speaker_names = batch["speaker_names"]
        linear_input = batch["linear"]
        mel_input = batch["mel"]
        mel_lengths = batch["mel_lengths"]
        stop_targets = batch["stop_targets"]
        item_idx = batch["item_idxs"]
        d_vectors = batch["d_vectors"]
        speaker_ids = batch["speaker_ids"]
        attn_mask = batch["attns"]
        waveform = batch["waveform"]
        pitch = batch["pitch"]
        energy = batch["energy"]
        language_ids = batch["language_ids"]
        max_text_length = torch.max(text_lengths.float())
        max_spec_length = torch.max(mel_lengths.float())

        # compute durations from attention masks
        durations = None
        if attn_mask is not None:
            durations = torch.zeros(attn_mask.shape[0], attn_mask.shape[2])
            for idx, am in enumerate(attn_mask):
                # compute raw durations
                c_idxs = am[:, : text_lengths[idx], : mel_lengths[idx]].max(1)[1]
                # c_idxs, counts = torch.unique_consecutive(c_idxs, return_counts=True)
                c_idxs, counts = torch.unique(c_idxs, return_counts=True)
                dur = torch.ones([text_lengths[idx]]).to(counts.dtype)
                dur[c_idxs] = counts
                # smooth the durations and set any 0 duration to 1
                # by cutting off from the largest duration indeces.
                extra_frames = dur.sum() - mel_lengths[idx]
                largest_idxs = torch.argsort(-dur)[:extra_frames]
                dur[largest_idxs] -= 1
                assert dur.sum() == mel_lengths[idx], f" [!] total duration {dur.sum()} vs spectrogram length {mel_lengths[idx]}"
                durations[idx, : text_lengths[idx]] = dur

        # set stop targets wrt reduction factor
        stop_targets = stop_targets.view(text_input.shape[0], stop_targets.size(1) // self.config.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze(2)
        stop_target_lengths = torch.divide(mel_lengths, self.config.r).ceil_()

        return {
            "text_input": text_input,
            "text_lengths": text_lengths,
            "speaker_names": speaker_names,
            "mel_input": mel_input,
            "mel_lengths": mel_lengths,
            "linear_input": linear_input,
            "stop_targets": stop_targets,
            "stop_target_lengths": stop_target_lengths,
            "attn_mask": attn_mask,
            "durations": durations,
            "speaker_ids": speaker_ids,
            "d_vectors": d_vectors,
            "max_text_length": float(max_text_length),
            "max_spec_length": float(max_spec_length),
            "item_idx": item_idx,
            "waveform": waveform,
            "pitch": pitch,
            "energy": energy,
            "language_ids": language_ids,
            "audio_unique_names": batch["audio_unique_names"],
            "ssl_feats": batch["ssl_feats"],
            "ssl_feats_lens": batch["ssl_feats_lens"],
            "ssl_feats_rel_lens": batch["ssl_feats_rel_lens"],
        }

    @torch.no_grad()
    def test_run(self, assets):
        print(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        aux_inputs = self._get_test_aux_input()
        if len(test_sentences) == 0:
            print(" | [!] No test sentences provided.")
        else:
            for idx, sen in enumerate(test_sentences):
                outputs = ssl_synthesis(
                    self,
                    sen,
                    self.vocoder,
                    speaker_id=aux_inputs["speaker_id"],
                    d_vector=aux_inputs["d_vector"],
                    do_trim_silence=False,
                )

                test_audios[f"{idx}-audio"] = outputs["wav"]
                test_figures[f"{idx}-prediction"] = plot_spectrogram(outputs["outputs"]["model_outputs"], self.ap, output_fig=False)
                test_figures[f"{idx}-alignment"] = plot_alignment(outputs["alignments"], output_fig=False)
        return {"figures": test_figures, "audios": test_audios}

    def test_log(self, outputs, logger, assets, steps) -> None:  # pylint:disable=W0613
        logger.test_audios(steps, outputs["audios"], self.ap.sample_rate)
        logger.test_figures(steps, outputs["figures"])
