"""
concat.py — 合并拼接 (sequential concatenation) of audio files and excerpts.

Given an ordered list of *segments* — each either a whole library file or a
[start, end) excerpt of one (the same source file may appear several times) —
this module joins them head-to-tail into a single new WAV:

    out = seg0 | seg1 | ... | segN-1

Two junction styles are supported:

* **Hard splice** (``mode="hard"``): segments are placed back-to-back with no
  overlap, so the output length is exactly the sum of the segment lengths.
* **Crossfade** (``mode="crossfade"``): neighbouring segments overlap for a
  short fade region.  The outgoing segment fades out while the incoming one
  fades in, giving a click-free transition.  Output length is the segment sum
  minus the sum of the (clamped) overlap lengths.

  Two curves are available: ``"linear"`` (equal-amplitude) and
  ``"equal_power"`` (equal-energy — cos/sin, the natural choice for unrelated
  material).  A junction whose fade is longer than half of either adjacent
  segment is automatically shortened (``fade = min(fade, len_i // 2,
  len_{i+1} // 2)``), so a very short segment can never overlap itself or a
  neighbour twice.

Heterogeneous material is normalised transparently:

* different sample rates  -> every segment is resampled to one target rate
  (the highest source rate by default) using the seamless streaming
  resampler from :mod:`dsp`;
* different channel counts -> mono is duplicated to stereo, stereo is
  down-mixed to mono, and multi-channel material is folded to mono first,
  all at the target channel layout (max of the sources, capped at stereo).

Like :mod:`mixer`, the renderer streams every segment a fixed-size chunk at a
time: the only material held back is the pending tail of one segment (at most
one fade length), so arbitrarily long inputs stay bounded in memory.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Iterator, List, Optional, Sequence

from . import audio_io, dsp

CHUNK = 1 << 14          # output frames per processing chunk
MIN_SEGMENTS = 2
MAX_SEGMENTS = 64        # "十几段" is the typical case; allow headroom
MAX_FADE_S = 60.0


# --------------------------------------------------------------------------- #
# Channel-layout adaptation
# --------------------------------------------------------------------------- #

def _adapt_channels(data: Sequence[List[float]], src_ch: int,
                    dst_ch: int) -> List[List[float]]:
    """Convert a de-interleaved chunk from ``src_ch`` to ``dst_ch`` channels.

    * same layout                  -> untouched
    * anything -> mono             : average all channels
    * mono -> multi-channel        : duplicate
    * multi-channel (e.g. 5.1) -> stereo/mono : fold to mono, then spread
    """
    n = len(data[0])
    if src_ch == dst_ch:
        return [list(c) for c in data]
    if dst_ch == 1:
        inv = 1.0 / src_ch
        mono = [sum(data[c][i] for c in range(src_ch)) * inv for i in range(n)]
        return [mono]
    if src_ch == 1:
        mono = data[0]
        return [list(mono) for _ in range(dst_ch)]
    inv = 1.0 / src_ch
    mono = [sum(data[c][i] for c in range(src_ch)) * inv for i in range(n)]
    return [list(mono) for _ in range(dst_ch)]


# --------------------------------------------------------------------------- #
# Deterministic per-segment stream
# --------------------------------------------------------------------------- #

def _segment_stream(reader: audio_io.WavReader, start_frame: int, end_frame: int,
                    dst_sr: int, dst_ch: int) -> Iterator[List[List[float]]]:
    """Yield exactly ``n_out`` target-rate/channel frames for one excerpt.

    The reader is seeked to ``start_frame`` and at most ``end_frame`` source
    frames are consumed.  Resampling is pumped per chunk with
    :class:`dsp.StreamingResampler`; if the resampler trails by a sample or
    two at the source end (it never extrapolates past the final source
    sample), the last value is held — zero-order hold for 1-2 frames is
    inaudible and keeps every segment's length exactly deterministic, which
    the overlap arithmetic depends on.
    """
    rch = reader.channels
    src_sr = reader.sr
    n_src = end_frame - start_frame
    n_out = max(1, int(round(n_src * dst_sr / src_sr)))
    reader._w.setpos(start_frame)

    resamplers: Optional[List[dsp.StreamingResampler]] = None
    if src_sr != dst_sr:
        resamplers = [dsp.StreamingResampler(src_sr, dst_sr) for _ in range(rch)]

    eof = False
    src_read = 0
    produced = 0
    while produced < n_out:
        want = min(CHUNK, n_out - produced)
        outs = [[] for _ in range(rch)]

        if resamplers is None:
            while len(outs[0]) < want and not eof:
                ask = min(want - len(outs[0]), n_src - src_read)
                if ask <= 0:
                    eof = True
                    break
                raw = reader.read_chunk(ask)
                if raw is None:
                    eof = True
                    break
                take = min(len(raw[0]), ask)
                for c in range(rch):
                    outs[c].extend(raw[c][:take])
                src_read += take
        else:
            while len(outs[0]) < want:
                # Drain whatever the resampler buffers already hold.
                for c in range(rch):
                    outs[c].extend(resamplers[c].pull(want - len(outs[c])))
                if len(outs[0]) >= want:
                    break
                if src_read >= n_src:
                    eof = True
                    for c in range(rch):
                        outs[c].extend(resamplers[c].flush(1 << 20))
                    break
                # Ask for roughly the source equivalent of the output shortfall.
                ask = max(1, min(int(want * src_sr / dst_sr) + 64, n_src - src_read))
                raw = reader.read_chunk(ask)
                if raw is None:
                    eof = True
                    for c in range(rch):
                        outs[c].extend(resamplers[c].flush(1 << 20))
                    break
                take = len(raw[0])
                for c in range(rch):
                    resamplers[c].push(raw[c])
                src_read += take

        m = min((len(o) for o in outs), default=0)
        m = min(m, want)
        if m > 0:
            adapted = _adapt_channels([o[:m] for o in outs], rch, dst_ch)
            yield adapted
            produced += m
        else:
            adapted = None

        if produced < n_out and eof:
            # Source genuinely exhausted before the target length (resampler
            # tail or a short final read): finish with a held value, or
            # silence if the source produced nothing at all.
            missing = n_out - produced
            fill = [adapted[c][-1] for c in range(dst_ch)] if adapted else [0.0] * dst_ch
            yield [[fill[c]] * missing for c in range(dst_ch)]
            produced += missing


# --------------------------------------------------------------------------- #
# Crossfade curves
# --------------------------------------------------------------------------- #

def _fade_gains(curve: str, n: int) -> List[tuple]:
    """Per-frame (outgoing gain, incoming gain) pairs for a fade of length n."""
    gains = []
    for k in range(n):
        t = (k + 0.5) / n
        if curve == "linear":
            gains.append((1.0 - t, t))
        elif curve == "equal_power":
            ang = t * math.pi / 2.0
            gains.append((math.cos(ang), math.sin(ang)))
        else:
            raise ValueError(f"unknown fade curve {curve!r} (use 'linear' or 'equal_power')")
    return gains


def _gather_head(stream: Iterator[List[List[float]]], count: int) -> tuple:
    """Pull exactly ``count`` head frames from ``stream``.

    Returns ``(head, leftover)`` where ``leftover`` holds any already-consumed
    frames belonging to the segment's body (a chunk may straddle the
    head/body boundary and must not be lost).
    """
    out: Optional[List[List[float]]] = None
    ch = 0
    for chunk in stream:
        if out is None:
            ch = len(chunk)
            out = [[] for _ in range(ch)]
        have = len(out[0])
        if have + len(chunk[0]) <= count:
            for c in range(ch):
                out[c].extend(chunk[c])
        else:
            take = count - have
            for c in range(ch):
                out[c].extend(chunk[c][:take])
            leftover = [chunk[c][take:] for c in range(ch)]
            return out, leftover
        if len(out[0]) == count:
            return out, None
    # Stream ended early (should not happen — segment length is fixed).
    if out is None:
        raise ValueError("segment ended before crossfade region")
    return out, None


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def concatenate(segments: Sequence[Dict], out_path: str,
                mode: str = "crossfade", fade_s: float = 0.5,
                curve: str = "equal_power",
                target_sr: Optional[int] = None,
                target_channels: Optional[int] = None) -> Dict:
    """Join ``segments`` head-to-tail and write a PCM16 WAV to ``out_path``.

    Each segment is ``{"path", "start" (s, default 0), "end" (s, default
    EOF)}``.  Returns a metadata dict (lengths, effective per-junction
    fades, …).

    ``mode`` is ``"hard"`` (back-to-back, no overlap) or ``"crossfade"``
    with ``fade_s`` seconds of overlap and ``curve`` in
    ``{"linear", "equal_power"}``.
    """
    if not (MIN_SEGMENTS <= len(segments) <= MAX_SEGMENTS):
        raise ValueError(
            f"need between {MIN_SEGMENTS} and {MAX_SEGMENTS} segments, got {len(segments)}")
    if mode not in ("hard", "crossfade"):
        raise ValueError(f"unknown concat mode {mode!r} (use 'hard' or 'crossfade')")
    fade_s = max(0.0, min(float(fade_s), MAX_FADE_S))
    if mode == "hard":
        fade_s = 0.0
    elif curve not in ("linear", "equal_power"):
        raise ValueError(f"unknown fade curve {curve!r} (use 'linear' or 'equal_power')")

    # Open every source once and resolve the [start, end) ranges to frames.
    readers: List[audio_io.WavReader] = []
    resolved: List[Dict] = []
    try:
        for idx, seg in enumerate(segments):
            path = seg.get("path")
            if not path:
                raise ValueError(f"segment {idx}: missing path")
            r = audio_io.WavReader(path)
            readers.append(r)
            dur = r.duration
            start = float(seg.get("start") or 0.0)
            end = seg.get("end")
            end = float(end) if end is not None else dur
            start = max(0.0, min(start, dur))
            end = max(0.0, min(end, dur))
            sf = int(round(start * r.sr))
            ef = int(round(end * r.sr))
            if ef <= sf:
                raise ValueError(
                    f"segment {idx}: empty/invalid range [{start:.3f}, {end:.3f}] s")
            resolved.append({"reader": r, "start_frame": sf, "end_frame": ef})

        sr = int(target_sr) if target_sr else max(r.sr for r in readers)
        if sr <= 0:
            raise ValueError("invalid target sample rate")
        if target_channels:
            ch = int(target_channels)
            if ch not in (1, 2):
                raise ValueError("target channel count must be 1 or 2")
        else:
            ch = min(2, max(r.channels for r in readers))

        # Exact target-frame length of every segment.
        lengths = [
            max(1, int(round((s["end_frame"] - s["start_frame"]) * sr / s["reader"].sr)))
            for s in resolved
        ]

        # Effective overlap per junction: never more than half of either side.
        fade_cfg = int(round(fade_s * sr)) if mode == "crossfade" else 0
        xfades: List[int] = []
        for i in range(len(resolved) - 1):
            xfades.append(min(fade_cfg, lengths[i] // 2, lengths[i + 1] // 2))

        total_frames = sum(lengths) - sum(xfades)
        gains_tables = {n: _fade_gains(curve, n) for n in set(xfades) if n > 0}

        total_written = 0
        with audio_io.WavWriter(out_path, sr, ch, 2) as w:
            pending: Optional[List[List[float]]] = None  # previous segment's held tail

            for i, spec in enumerate(resolved):
                xin = xfades[i - 1] if i > 0 else 0
                xout = xfades[i] if i < len(xfades) else 0
                stream = _segment_stream(spec["reader"], spec["start_frame"],
                                         spec["end_frame"], sr, ch)

                # 1) head of this segment cross-fades on top of the pending tail.
                carry: Optional[List[List[float]]] = None
                if xin:
                    if pending is None or len(pending[0]) != xin:
                        raise ValueError("internal: pending tail length mismatch")
                    head, carry = _gather_head(stream, xin)
                    gains = gains_tables[xin]
                    blended = [[0.0] * xin for _ in range(ch)]
                    for c in range(ch):
                        pc, hc = pending[c], head[c]
                        bc = blended[c]
                        for k in range(xin):
                            ga, gb = gains[k]
                            v = pc[k] * ga + hc[k] * gb
                            bc[k] = dsp.clamp(v, -1.0, 1.0)
                    w.write_chunk(blended)
                    total_written += xin
                    pending = None

                # 2) body: stream straight through, holding back the last
                #    ``xout`` frames so they can overlap the next segment.
                held = [deque() for _ in range(ch)]
                emit = [[] for _ in range(ch)]

                def _feed(chunk_frames: Sequence[Sequence[float]]) -> None:
                    nonlocal total_written
                    if xout == 0:
                        w.write_chunk(chunk_frames)
                        total_written += len(chunk_frames[0])
                        return
                    nfr = len(chunk_frames[0])
                    for j in range(nfr):
                        if len(held[0]) < xout:
                            for c in range(ch):
                                held[c].append(chunk_frames[c][j])
                        else:
                            for c in range(ch):
                                emit[c].append(held[c].popleft())
                                held[c].append(chunk_frames[c][j])
                    if len(emit[0]) >= CHUNK:
                        w.write_chunk(emit)
                        total_written += len(emit[0])
                        for c in range(ch):
                            emit[c] = []

                if carry is not None and carry[0]:
                    _feed(carry)
                for chunk_frames in stream:
                    _feed(chunk_frames)
                if emit[0]:
                    w.write_chunk(emit)
                    total_written += len(emit[0])
                pending = [list(d) for d in held]

            # Trailing tail of the final segment (xfades end in 0 anyway).
            if pending and pending[0]:
                w.write_chunk(pending)
                total_written += len(pending[0])

        if total_written != total_frames:
            raise ValueError(
                f"internal: wrote {total_written} frames, expected {total_frames}")
    finally:
        for r in readers:
            r.close()

    return {
        "segments": len(segments),
        "sr": sr,
        "channels": ch,
        "frames": total_frames,
        "duration": total_frames / sr if sr else 0.0,
        "mode": mode,
        "curve": curve if mode == "crossfade" else None,
        "segment_frames": lengths,
        "xfades_frames": xfades,
        "xfades_s": [x / sr for x in xfades],
        "sum_duration": sum(lengths) / sr if sr else 0.0,
    }
