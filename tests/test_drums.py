"""CPU smoke test of the drum stage (skipped when the weights cannot be loaded)."""
import numpy as np
import pytest

from dettato.drums import FPS, MIN_SOURCE_RATE, DrumTranscriber, features, peaks


@pytest.fixture(scope="module")
def drums():
    try:
        return DrumTranscriber.from_pretrained("cpu")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"drum weights not available: {e!r}")


def clicks(seconds=4.0, sr=44100):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(int(seconds * sr)) * 0.01).astype(np.float32)
    for t in np.arange(0.25, seconds, 0.5):
        i = int(t * sr)
        x[i:i + 400] += np.exp(-np.arange(400) / 60.0).astype(np.float32) * rng.standard_normal(400).astype(np.float32)
    return x


def test_features_shape():
    f = features(clicks(2.0))
    assert f.shape == (200, 280) and f.dtype == np.float16


def test_peaks():
    p = np.zeros((10, 2), np.float32)
    p[3, 0], p[4, 0], p[8, 1] = 0.9, 0.6, 0.7
    fr, ln = peaks(p, 0.5)
    assert fr.tolist() == [3, 8] and ln.tolist() == [0, 1]


def test_transcribe(drums):
    hits = drums.transcribe(clicks())
    assert isinstance(hits, list)
    for h in hits:
        assert set(h) == {"onset", "offset", "pitch", "is_drum", "velocity"}
        assert 27 <= h["pitch"] <= 87 and 0.05 <= h["velocity"] <= 1.0 and h["offset"] > h["onset"]
    assert [h["onset"] for h in hits] == sorted(h["onset"] for h in hits)
    assert drums.last_info["model"] in ("drum-kit", "orchestral") and 0.0 <= drums.last_info["router"] <= 1.0
    assert MIN_SOURCE_RATE == 40000 and FPS == 50


def test_other_rate_and_short_input(drums):
    assert isinstance(drums.transcribe(clicks(2.0, 48000), sr=48000), list)
    assert drums.transcribe(np.zeros(100, np.float32)) == []
