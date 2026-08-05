"""Tests for the audio front end.

The shape test is the load-bearing one: OPERA-GT's positional embedding has
exactly 1025 entries, so a spectrogram of the wrong size fails at the encoder with
a shape error. Catching it here, against the documented parameters, is cheaper.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from respnet.data.audio import (
    AudioConfig,
    LogMelSpectrogram,
    clip_waveform,
    mel_filterbank,
    _hz_to_mel,
    _mel_to_hz,
)


@pytest.fixture
def cfg() -> AudioConfig:
    return AudioConfig()


class TestAudioConfig:
    def test_derived_lengths_match_opera_parameters(self, cfg):
        # 64 ms window, 32 ms hop at 16 kHz.
        assert cfg.win_length == 1024
        assert cfg.hop_length == 512

    def test_n_fft_is_power_of_two_at_least_window_length(self, cfg):
        assert cfg.n_fft >= cfg.win_length
        assert cfg.n_fft & (cfg.n_fft - 1) == 0

    def test_patch_grid_matches_opera_gt_positional_embedding(self, cfg):
        """(n_mels/4) * (n_frames/4) must equal 1024 patches."""
        assert (cfg.n_mels // 4) * (cfg.n_frames // 4) == 1024

    def test_clip_seconds_consistent_with_frames_and_hop(self, cfg):
        assert cfg.clip_seconds == pytest.approx(cfg.n_frames * cfg.hop_ms / 1000.0)


class TestMelScale:
    def test_round_trip(self):
        freqs = np.array([0.0, 100.0, 1000.0, 8000.0])
        assert np.allclose(_mel_to_hz(_hz_to_mel(freqs)), freqs, atol=1e-3)

    def test_monotonic(self):
        f = np.linspace(0, 8000, 100)
        assert np.all(np.diff(_hz_to_mel(f)) > 0)

    def test_zero_maps_to_zero(self):
        assert _hz_to_mel(0.0) == pytest.approx(0.0)


class TestMelFilterbank:
    def test_shape(self, cfg):
        assert mel_filterbank(cfg).shape == (cfg.n_mels, cfg.n_fft // 2 + 1)

    def test_all_weights_non_negative(self, cfg):
        assert (mel_filterbank(cfg) >= 0).all()

    def test_every_filter_has_support(self, cfg):
        """A filter summing to zero contributes nothing and signals bad edge spacing."""
        assert (mel_filterbank(cfg).sum(dim=1) > 0).all()

    def test_filters_are_ordered_by_centre_frequency(self, cfg):
        fb = mel_filterbank(cfg)
        centres = fb.argmax(dim=1).numpy()
        assert np.all(np.diff(centres) >= 0)


class TestLogMelSpectrogram:
    def test_output_shape_is_exactly_the_configured_size(self, cfg):
        spec = LogMelSpectrogram(cfg)(torch.randn(cfg.sample_rate * 9))
        assert spec.shape == (1, cfg.n_mels, cfg.n_frames)

    def test_short_input_is_padded_to_full_width(self, cfg):
        spec = LogMelSpectrogram(cfg)(torch.randn(cfg.sample_rate // 2))
        assert spec.shape == (1, cfg.n_mels, cfg.n_frames)

    def test_batched_input(self, cfg):
        spec = LogMelSpectrogram(cfg)(torch.randn(4, cfg.sample_rate * 9))
        assert spec.shape == (4, cfg.n_mels, cfg.n_frames)

    def test_per_example_standardisation(self, cfg):
        spec = LogMelSpectrogram(cfg)(torch.randn(3, cfg.sample_rate * 9))
        assert spec.mean(dim=(-2, -1)).abs().max() < 1e-4
        assert (spec.std(dim=(-2, -1)) - 1.0).abs().max() < 1e-2

    def test_gain_invariance(self, cfg):
        """A 40 dB level change must not change the output.

        Recording level varies enormously across ICBHI's four devices. If this
        fails, the encoder is being shown gain rather than content.
        """
        mel = LogMelSpectrogram(cfg)
        wave = torch.randn(cfg.sample_rate * 9)
        quiet, loud = mel(wave * 0.01), mel(wave * 1.0)
        assert torch.allclose(quiet, loud, atol=1e-3)

    def test_silence_produces_no_nans(self, cfg):
        """The log and the std division both have divide-by-zero paths."""
        spec = LogMelSpectrogram(cfg)(torch.zeros(cfg.sample_rate * 9))
        assert torch.isfinite(spec).all()

    def test_deterministic(self, cfg):
        mel = LogMelSpectrogram(cfg)
        wave = torch.randn(cfg.sample_rate * 9)
        assert torch.equal(mel(wave), mel(wave))


class TestClipWaveform:
    def test_exact_length(self, cfg):
        need = int(cfg.clip_seconds * cfg.sample_rate)
        assert len(clip_waveform(np.random.randn(cfg.sample_rate * 20).astype(np.float32), cfg)) == need

    def test_short_input_is_tiled_not_truncated(self, cfg):
        need = int(cfg.clip_seconds * cfg.sample_rate)
        assert len(clip_waveform(np.random.randn(1000).astype(np.float32), cfg)) == need

    def test_offset_selects_a_later_window(self, cfg):
        wave = np.arange(cfg.sample_rate * 30, dtype=np.float32)
        assert clip_waveform(wave, cfg, offset=1000)[0] == 1000.0

    def test_offset_beyond_end_still_returns_full_length(self, cfg):
        need = int(cfg.clip_seconds * cfg.sample_rate)
        wave = np.random.randn(cfg.sample_rate * 2).astype(np.float32)
        assert len(clip_waveform(wave, cfg, offset=10**7)) == need
