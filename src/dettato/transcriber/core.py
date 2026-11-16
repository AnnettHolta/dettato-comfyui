"""Running the transcriber over a whole recording.

The network sees 8 s windows. A song is covered by overlapping windows (every 2 s, or every 4 s for the grouper's features); each
window's outputs are weighted by a Hann taper and averaged per 20 ms frame. Besides the six probability maps the same pass can
average two internal states for the later stages: ``hidden`` (the 368-d input of the output heads, per frame and key) and
``stem`` (the stem output, the least processed state).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .. import hub
from .decode import decode
from .frontend import N_KEYS, PITCH_LO, SR
from .model import LaneNet

HEADS = ("onset", "frame", "offset", "onset2", "frame2", "offset2")
WIN_S = 8.0


def _autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda"))


class Transcriber:
    """Audio (16 kHz mono) -> notes, two voices per key, each with a confidence.

    >>> t = Transcriber.from_pretrained("cuda")
    >>> maps, _ = t.run(audio)
    >>> notes = t.notes(maps)          # dict of arrays: onset, offset, pitch, conf, slot
    """

    def __init__(self, model: LaneNet, thresholds: dict, device="cpu"):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.th = dict(thresholds)
        self.hop = model.hop_out

    @classmethod
    def from_pretrained(cls, device="cpu"):
        cfg = hub.load_config("transcriber")
        model = LaneNet(**cfg["model"])
        model.load_state_dict(hub.load_state("transcriber"))
        return cls(model, cfg["thresholds"], device)

    @property
    def channels(self) -> int:
        return self.model.head.in_channels

    # ------------------------------------------------------------------ windowed pass
    def _layout(self, n, hop_frac):
        W, Fw = int(round(WIN_S * SR)), int(round(WIN_S / self.hop))
        T = int(math.ceil(n / SR / self.hop))
        step = max(1, int(round(W * hop_frac)))
        pad = W // 2
        starts = list(range(0, n + pad, step))
        g_pad = int(round(pad / SR / self.hop))
        return W, Fw, T, starts, pad, g_pad

    def frame_offset(self, s0, g_pad):
        """Song frame of window frame 0 for a window starting at padded sample s0."""
        return int(round(s0 / SR / self.hop)) - g_pad

    @torch.no_grad()
    def windows(self, audio, hop_frac=0.25, batch=2, taps=(), pad_batch=False):
        """Yield (window starts, {head: sigmoid probs (b, Fw, 88)}, {tap: (b, C, Fw, lanes)}) batch by batch.
        pad_batch: fill the last batch with silent windows (one static batch shape)."""
        n = len(audio)
        W, Fw, T, starts, pad, g_pad = self._layout(n, hop_frac)
        x_all = np.concatenate([np.zeros(pad, np.float32), audio.astype(np.float32), np.zeros(pad + W, np.float32)])
        for bi in range(0, len(starts), batch):
            ss = starts[bi:bi + batch]
            xs = np.stack([x_all[s0:s0 + W] for s0 in ss])
            if pad_batch and len(ss) < batch:
                xs = np.concatenate([xs, np.zeros((batch - len(ss), W), np.float32)])
            x = torch.from_numpy(xs).to(self.device)
            box = {} if taps else None
            with _autocast(self.device):
                res = self.model(x, box)
            probs = {k: torch.sigmoid(res[k].float()) for k in HEADS}
            yield ss, probs, {k: box[k] for k in taps}

    @torch.no_grad()
    def run(self, audio, hop_frac=0.25, batch=2, taps=(), pad_batch=False):
        """-> (maps {head: (T, 88) float32 numpy}, {tap: (T, 88, C) float16 tensor on the device}).

        Each map / tap is the Hann-weighted mean over the windows covering the frame."""
        n = len(audio)
        W, Fw, T, starts, pad, g_pad = self._layout(n, hop_frac)
        w = torch.from_numpy(np.hanning(Fw + 2)[1:-1].astype(np.float32)).to(self.device)
        P = {k: torch.zeros(T, N_KEYS, device=self.device) for k in HEADS}
        buf = {}
        wsum = torch.zeros(T, device=self.device)
        for ss, pr, tp in self.windows(audio, hop_frac, batch, taps, pad_batch):
            for k, s0 in enumerate(ss):
                g0 = self.frame_offset(s0, g_pad)
                a, b = max(0, g0), min(T, g0 + Fw)
                if b <= a:
                    continue
                wt = w[a - g0:b - g0]
                for hd in HEADS:
                    P[hd][a:b] += pr[hd][k, a - g0:b - g0] * wt[:, None]
                for name, v in tp.items():
                    if name not in buf:
                        buf[name] = torch.zeros(T, N_KEYS, v.shape[1], device=self.device, dtype=torch.float16)
                    buf[name][a:b] += (v[k, :, a - g0:b - g0, :N_KEYS].permute(1, 2, 0).float() * wt[:, None, None]).half()
                wsum[a:b] += wt
        ws = wsum.clamp(min=1e-6)
        maps = {hd: (P[hd] / ws[:, None]).cpu().numpy() for hd in HEADS}
        for name in buf:
            buf[name] /= ws[:, None, None].half()
        return maps, buf

    # ------------------------------------------------------------------ notes
    def notes(self, maps):
        """Decode both voices -> dict of arrays sorted by (onset, offset, pitch):
        onset / offset (s), pitch (MIDI), conf (the voice's onset probability at the onset frame), slot (1 / 2)."""
        th = self.th
        args = dict(off_th=th["off_th"], fr_th=th["fr_th"], fr_low_frames=th["fr_low_frames"], on_nms=th.get("on_nms", 0))
        rows = []
        for slot, keys, on_th in ((1, ("onset", "frame", "offset"), th["on_th"]), (2, ("onset2", "frame2", "offset2"), th.get("v2_th"))):
            if on_th is None:
                continue
            on = maps[keys[0]]
            for u0, u1, p in decode(maps[keys[0]], maps[keys[1]], maps[keys[2]], self.hop, on_th=on_th, **args):
                t = int(min(max(round(u0 / self.hop), 0), on.shape[0] - 1))
                rows.append((u0, u1, p, float(on[t, p - PITCH_LO]), slot))
        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        a = np.array([r[:3] for r in rows], np.float64).reshape(-1, 3)
        return dict(onset=a[:, 0], offset=a[:, 1], pitch=a[:, 2].astype(np.int64), conf=np.array([r[3] for r in rows], np.float32),
                    slot=np.array([r[4] for r in rows], np.int8))
