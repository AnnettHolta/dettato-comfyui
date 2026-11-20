import numpy as np

from conftest import needs, tone


@needs["transcriber"]
def test_transcriber_finds_two_notes():
    from dettato.transcriber import HEADS, Transcriber

    t = Transcriber.from_pretrained("cpu")
    x = tone(4.0, freqs=(220.0, 330.0))
    maps, buf = t.run(x, taps=("hidden",))
    assert set(maps) == set(HEADS)
    assert maps["onset"].shape == (200, 88)
    assert buf["hidden"].shape == (200, 88, t.channels)
    notes = t.notes(maps)
    assert set(notes) == {"onset", "offset", "pitch", "conf", "slot"}
    assert {57, 64} <= set(notes["pitch"].tolist())          # A3 and E4
    assert np.all(notes["offset"] > notes["onset"])


def test_decode_single_peak():
    from dettato.transcriber import decode

    T = 50
    on, fr, off = np.zeros((T, 88)), np.zeros((T, 88)), np.zeros((T, 88))
    on[10, 39] = 0.9
    fr[10:30, 39] = 0.9
    notes = decode(on, fr, off, 0.02)
    assert len(notes) == 1 and notes[0][2] == 60
    assert abs(notes[0][0] - 0.2) < 1e-9 and abs(notes[0][1] - 0.59) < 1e-9


@needs["transcriber"]
def test_sub_run_equals_a_separate_coarser_run():
    """One pass at hop 0.25 also returns the average a separate hop-0.5 run gives. The windows are the same and are added in the
    same order; only the batch a window ran in differs, which moves a CPU float32 result by ~1e-9 (a float16 tap by <= 1 step)."""
    import pytest
    import torch

    from dettato.transcriber import Transcriber

    t = Transcriber.from_pretrained("cpu")
    x = tone(21.0, freqs=(220.0, 330.0))              # 6 windows at hop 0.5, 11 at hop 0.25
    maps, buf, (smaps, sbuf) = t.run(x, hop_frac=0.25, taps=("hidden",), sub_hop_frac=0.5, sub_taps=("hidden", "stem"))
    maps_ref, buf_ref = t.run(x, hop_frac=0.25, taps=("hidden",))
    smaps_ref, sbuf_ref = t.run(x, hop_frac=0.5, taps=("hidden", "stem"))
    assert set(buf) == {"hidden"} and set(sbuf) == {"hidden", "stem"}
    for k in maps:                                    # the full average is the old run's, window for window
        assert np.array_equal(maps[k], maps_ref[k])
        assert np.allclose(smaps[k], smaps_ref[k], atol=1e-6)
    assert torch.equal(buf["hidden"], buf_ref["hidden"])
    for k in ("hidden", "stem"):
        assert (sbuf[k].float() - sbuf_ref[k].float()).abs().max() < 0.02
    with pytest.raises(ValueError):
        t.run(x, hop_frac=0.5, sub_hop_frac=0.75)     # not a multiple of the hop


def test_host_buffer_matches_a_device_buffer():
    """The streamed host-memory tap buffer (long recordings) is bit-identical to summing on the device and dividing at the end."""
    import torch

    from dettato.transcriber.core import _HostBuffer

    torch.manual_seed(0)
    T, C, Fw = 1000, 6, 400
    starts = list(range(-200, T, 100))                                   # windows in order of their start, as in run()
    w = torch.hann_window(Fw + 2, periodic=False)[1:-1]
    full, wsum = torch.zeros(T, 88, C, dtype=torch.float16), torch.zeros(T)
    hb = _HostBuffer(T, C, torch.device("cpu"), 2 * Fw)
    for g0 in starts:
        a, b = max(0, g0), min(T, g0 + Fw)
        wt = w[a - g0:b - g0]
        x = (torch.randn(b - a, 88, C) * wt[:, None, None]).half()
        full[a:b] += x
        hb.add(a, b, x, wsum.clamp(min=1e-6))
        wsum[a:b] += wt
    ws = wsum.clamp(min=1e-6)
    full /= ws[:, None, None].half()
    hb.flush(T, ws)
    assert torch.equal(hb.host, full)


@needs["transcriber"]
def test_host_resident_taps_equal_device_taps():
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from dettato.transcriber import Transcriber

    t = Transcriber.from_pretrained("cuda")
    x = tone(30.0, freqs=(220.0, 330.0))
    kw = dict(hop_frac=0.25, taps=("hidden",), sub_hop_frac=0.5, sub_taps=("hidden", "stem"))
    maps, buf, (smaps, sbuf) = t.run(x, **kw)
    maps_h, buf_h, (smaps_h, sbuf_h) = t.run(x, buf_device="cpu", **kw)
    assert buf_h["hidden"].device.type == "cpu"
    for k in maps:
        assert np.array_equal(maps[k], maps_h[k]) and np.array_equal(smaps[k], smaps_h[k])
    assert torch.equal(buf["hidden"].cpu(), buf_h["hidden"])
    assert torch.equal(sbuf["hidden"].cpu(), sbuf_h["hidden"]) and torch.equal(sbuf["stem"].cpu(), sbuf_h["stem"])
