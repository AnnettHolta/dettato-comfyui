import numpy as np

from conftest import needs, tone


@needs["transcriber"]
@needs["evidence"]
def test_repairer_runs():
    from dettato.evidence import N_EV, EvidenceRepairer, transcribe_with_evidence
    from dettato.transcriber import Transcriber

    t = Transcriber.from_pretrained("cpu")
    rep = EvidenceRepairer.from_pretrained("cpu")
    x = tone(4.0)
    res, _, ev = transcribe_with_evidence(t, x)
    assert ev.shape == (len(res["onset"]), N_EV)
    notes, original, info = rep(t, x)
    assert set(notes) == {"onset", "offset", "pitch", "inserted", "score"}
    assert info["kept"] + info["dropped"] == info["transcribed"] == len(original["onset"])
    assert np.all(notes["offset"] > notes["onset"])


@needs["transcriber"]
@needs["evidence"]
def test_repairer_from_run_equals_its_own_pass():
    from dettato.evidence import EvidenceRepairer
    from dettato.transcriber import Transcriber

    t = Transcriber.from_pretrained("cpu")
    rep = EvidenceRepairer.from_pretrained("cpu")
    x = tone(4.0)
    ref, _, info_ref = rep(t, x)
    maps, buf = t.run(x, hop_frac=0.25, taps=("hidden",))
    notes, _, info = rep.from_run(t, maps, buf["hidden"])
    assert info == info_ref
    for k in ref:
        assert np.array_equal(notes[k], ref[k])
