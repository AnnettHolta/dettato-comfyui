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
