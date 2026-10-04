"""
concat.py — 合并拼接: ordered concatenation of audio segments into one file.

Takes an ordered list of *segments* — each a whole file or a ``[start, end)``
excerpt of one (several segments may come from the same file) — and splices
them into a single WAV:

  * **Hard splice** (``crossfade_s = 0``): segments are appended directly.
  * **Crossfade**: adjacent segments overlap by up to ``crossfade_s`` seconds
    with an equal-power (or linear) fade curve so the transition is smooth.

Properties
----------
* **Mixed formats.**  Segments may differ in sample rate and channel count;
  every segment is resampled (seamless streaming resampler) and channel-mapped
  to the output format on the fly.  The output rate/channel count default to
  the maxima of the inputs unless explicitly requested.
* **Any segment length.**  The fade at each join is clamped to half the length
  of the shorter adjacent segment, so a tiny segment (a few milliseconds)
  between two long ones cannot corrupt the splice — it simply gets a shorter
  fade.  Conversely the pipeline streams chunk-by-chunk, so arbitrarily long
  segments are processed with memory bounded by one chunk plus one fade buffer.
* **Predictable duration.**  Output length = sum of segment lengths minus the
  crossfade overlaps (±1 frame per resampled segment).

The result is written as 16-bit PCM WAV via :class:`audio_io.WavWriter`.
"""

from __future__ import annotations

import math
import os
from collections import deque
from typing import Dict, Generator, List, Optional, Sequence

from . import audio_io, dsp

MAX_SEGMENTS = 100
MAX_CROSSFADE_S = 60.0
CURVES = ("equal_power", "linear")


# --------------------------------------------------------------------------- #
# Segment resolution
# --------------------------------------------------------------------------- #

class _Segment:
    """A resolved segment: an open reader plus a clamped frame range."""

    def __init__(self, reader: audio_io.WavReader, start_f: int, end_f: int):
        self.reader = reader
        self.start_f = start_f
        self.end_f = end_f

    @property
    def frames(self) -> int:
        return self.end_f - self.start_f

    def est_out_frames(self, sr: int) -> int:
        """Estimated length (frames) after resampling to ``sr`` (±1 frame)."""
        return int(round(self.frames * sr / self.reader.sr))


def _resolve(segments: Sequence[Dict]) -> List[_Segment]:
    """Open every segment file and clamp its [start, end) range to the file."""
    out: List[_Segment] = []
    try:
        for spec in segments:
            path = spec.get("path")
            if not path or not os.path.isfile(path):
                raise ValueError(f"片段文件不存在: {path}")
            r = audio_io.WavReader(path)
            total = r.nframes
            try:
                start = float(spec.get("start", 0.0) or 0.0)
                end_raw = spec.get("end")
                end = None if end_raw is None else float(end_raw)
            except (TypeError, ValueError):
                r.close()
                raise ValueError("片段起止时间无效")
            start_f = max(0, min(int(round(start * r.sr)), total))
            end_f = total if end is None else max(start_f, min(int(round(end * r.sr)), total))
            if end_f - start_f < 1:
                r.close()
                raise ValueError("片段为空：起点/终点超出文件范围或长度为零")
            out.append(_Segment(r, start_f, end_f))
    except Exception:
        for s in out:
            s.reader.close()
        raise
    return out


# --------------------------------------------------------------------------- #
# Channel mapping
# --------------------------------------------------------------------------- #

def _convert_channels(raw: Sequence[Sequence[float]], dst_ch: int) -> List[List[float]]:
    """Map a de-interleaved chunk to ``dst_ch`` channels.

    mono→N duplicates, N→1 averages, otherwise channels are folded (downmix)
    or tiled (upmix) cyclically — simple, predictable and click-free."""
    src_ch = len(raw)
    if src_ch == dst_ch:
        return [list(c) for c in raw]
    if dst_ch == 1:
        return [audio_io.to_mono(raw)]
    if src_ch == 1:
        return [list(raw[0]) for _ in range(dst_ch)]
    out = []
    for i in range(dst_ch):
        if src_ch > dst_ch:
            group = raw[i::dst_ch]
            if len(group) == 1:
                out.append(list(group[0]))
            else:
                m = min(len(g) for g in group)
                inv = 1.0 / len(group)
                out.append([sum(g[k] for g in group) * inv for k in range(m)])
        else:
            out.append(list(raw[i % src_ch]))
    return out


# --------------------------------------------------------------------------- #
# Segment streaming
# --------------------------------------------------------------------------- #

def _segment_chunks(seg: _Segment, sr: int, ch: int, chunk: int
                    ) -> Generator[List[List[float]], None, None]:
    """Yield one segment as output-rate, ``ch``-channel chunks."""
    r = seg.reader
    r.seek(seg.start_f)
    need_rs = r.sr != sr
    resamplers = [dsp.StreamingResampler(r.sr, sr) for _ in range(ch)] if need_rs else None
    remaining = seg.frames
    while remaining > 0:
        raw = r.read_chunk(min(chunk, remaining))
        if raw is None:
            break
        n = len(raw[0])
        if n == 0:
            break
        remaining -= n
        conv = _convert_channels(raw, ch)
        if need_rs:
            out = []
            for c in range(ch):
                resamplers[c].push(conv[c])
                out.append(resamplers[c].pull(1 << 22))
            m = min(len(o) for o in out)
            if m > 0:
                yield [o[:m] for o in out]
        else:
            yield conv
    if need_rs:
        tail = [rs.flush(1 << 22) for rs in resamplers]
        m = min(len(t) for t in tail)
        if m > 0:
            yield [t[:m] for t in tail]


class _Feed:
    """Pull-based chunk stream with take()/unget() for the join regions."""

    def __init__(self, gen, ch: int):
        self.gen = gen
        self.ch = ch
        self._pushback: deque = deque()
        self._done = False

    def next_chunk(self) -> Optional[List[List[float]]]:
        while self._pushback:
            c = self._pushback.popleft()
            if c and len(c[0]) > 0:
                return c
        if self._done:
            return None
        for c in self.gen:
            if c and len(c[0]) > 0:
                return c
        self._done = True
        return None

    def unget(self, chunk: List[List[float]]) -> None:
        if chunk and len(chunk[0]) > 0:
            self._pushback.appendleft(chunk)

    def take(self, n: int) -> List[List[float]]:
        """Take up to ``n`` frames (fewer if the stream runs out)."""
        out = [[] for _ in range(self.ch)]
        need = n
        while need > 0:
            c = self.next_chunk()
            if c is None:
                break
            m = len(c[0])
            if m <= need:
                for i in range(self.ch):
                    out[i].extend(c[i])
                need -= m
            else:
                for i in range(self.ch):
                    out[i].extend(c[i][:need])
                self._pushback.appendleft([c[i][need:] for i in range(self.ch)])
                need = 0
        return out


# --------------------------------------------------------------------------- #
# Crossfade
# --------------------------------------------------------------------------- #

def _fade_gains(f: int, curve: str) -> List[tuple]:
    """Per-frame (out_gain, in_gain) pairs across an ``f``-frame overlap."""
    if f <= 0:
        return []
    denom = max(1, f - 1)
    gains = []
    for k in range(f):
        t = k / denom
        if curve == "linear":
            gains.append((1.0 - t, t))
        else:  # equal power: constant loudness through the transition
            a = t * math.pi / 2.0
            gains.append((math.cos(a), math.sin(a)))
    return gains


def _mix(tail: Sequence[Sequence[float]], head: Sequence[Sequence[float]],
         gains: Sequence[tuple]) -> List[List[float]]:
    """Blend the tail of the previous segment with the head of the next."""
    out = []
    for tc, hc in zip(tail, head):
        out.append([tc[k] * g0 + hc[k] * g1 for k, (g0, g1) in enumerate(gains)])
    return out


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def concatenate(segments: Sequence[Dict], out_path: str,
                target_sr: Optional[int] = None, target_ch: Optional[int] = None,
                crossfade_s: float = 0.0, curve: str = "equal_power",
                chunk: int = 1 << 15) -> Dict:
    """Splice ``segments`` (in order) into ``out_path``.

    Each segment is ``{"path": str, "start": sec?, "end": sec?}``; omitted
    bounds mean the whole file.  ``crossfade_s = 0`` gives a hard splice.
    Returns a metadata dict (sr/channels/frames/duration/joins).
    """
    if not segments:
        raise ValueError("没有可拼接的片段")
    if len(segments) > MAX_SEGMENTS:
        raise ValueError(f"片段数量超过上限 {MAX_SEGMENTS}")

    segs = _resolve(segments)
    try:
        sr = int(target_sr) if target_sr else max(s.reader.sr for s in segs)
        ch = int(target_ch) if target_ch else max(s.reader.channels for s in segs)
        if sr <= 0 or ch <= 0:
            raise ValueError("输出采样率/声道数无效")
        if curve not in CURVES:
            curve = "equal_power"
        fade_n = max(0, int(min(max(float(crossfade_s), 0.0), MAX_CROSSFADE_S) * sr))

        # Per-join fade lengths, clamped to half of each adjacent segment so a
        # very short segment can never be swallowed by its two fades.
        est = [s.est_out_frames(sr) for s in segs]
        joins = []
        for i in range(len(segs) - 1):
            f = min(fade_n, est[i] // 2, est[i + 1] // 2)
            joins.append(max(0, f))

        total = 0
        with audio_io.WavWriter(out_path, sr, ch, 2) as w:
            prev_tail: Optional[List[List[float]]] = None
            for idx, seg in enumerate(segs):
                feed = _Feed(_segment_chunks(seg, sr, ch, chunk), ch)

                # -- join with the previous segment ------------------------- #
                if idx > 0 and prev_tail is not None:
                    head = feed.take(joins[idx - 1])
                    tp = len(prev_tail[0])
                    hn = len(head[0])
                    f = min(tp, hn)
                    if tp > f:  # part of the tail that stays un-faded
                        w.write_chunk([t[:tp - f] for t in prev_tail])
                        total += tp - f
                    if f > 0:
                        gains = _fade_gains(f, curve)
                        w.write_chunk(_mix([t[tp - f:] for t in prev_tail],
                                           [h[:f] for h in head], gains))
                        total += f
                    if hn > f:  # head frames beyond the overlap
                        feed.unget([h[f:] for h in head])

                # -- stream the segment body, holding back its tail --------- #
                hold = joins[idx] if idx < len(joins) else 0
                buf: List[List[float]] = [[] for _ in range(ch)]
                while True:
                    c = feed.next_chunk()
                    if c is None:
                        break
                    for i in range(ch):
                        buf[i].extend(c[i])
                    excess = len(buf[0]) - hold
                    if excess > 0:
                        w.write_chunk([b[:excess] for b in buf])
                        total += excess
                        buf = [b[excess:] for b in buf]
                prev_tail = buf

            if prev_tail is not None and len(prev_tail[0]) > 0:
                w.write_chunk(prev_tail)
                total += len(prev_tail[0])

        return {
            "sr": sr,
            "channels": ch,
            "frames": total,
            "duration": total / sr if sr else 0.0,
            "segments": len(segs),
            "crossfade_s": fade_n / sr if sr else 0.0,
            "curve": curve,
            "joins": joins,
        }
    finally:
        for s in segs:
            s.reader.close()
