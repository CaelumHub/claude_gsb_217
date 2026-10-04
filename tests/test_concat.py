"""
tests/test_concat.py — tests for the merge/splice engine (backend/concat.py)
and the /api/concat endpoint.

Run from the repo root:

    python3 -m unittest discover -s tests -v

The backend tests are pure-stdlib.  The API tests are skipped automatically
when Flask is not installed.
"""

import math
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import audio_io  # noqa: E402
from backend.concat import concatenate  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def write_sine(path, sr, ch, dur, freq=440.0, amp=0.5):
    n = int(round(sr * dur))
    mono = [amp * math.sin(2.0 * math.pi * freq * i / sr) for i in range(n)]
    audio_io.save(path, audio_io.AudioData([list(mono) for _ in range(ch)], sr))


def write_const(path, sr, ch, dur, val):
    n = int(round(sr * dur))
    audio_io.save(path, audio_io.AudioData([[val] * n for _ in range(ch)], sr))


class ConcatBase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def p(self, name):
        return os.path.join(self.dir, name)


# --------------------------------------------------------------------------- #
# Backend: duration / format / content
# --------------------------------------------------------------------------- #

class TestConcat(ConcatBase):
    def test_hard_splice_mixed_sr_channels(self):
        """22.05k mono + 44.1k stereo + 48k stereo -> 48k stereo, sum of durations."""
        a = self.p("a.wav"); write_sine(a, 22050, 1, 1.0)
        b = self.p("b.wav"); write_sine(b, 44100, 2, 0.5)
        c = self.p("c.wav"); write_sine(c, 48000, 2, 2.0)
        out = self.p("out.wav")
        r = concatenate([{"path": a}, {"path": b}, {"path": c}], out)
        self.assertEqual(r["sr"], 48000)
        self.assertEqual(r["channels"], 2)
        # 1.0 + 0.5 + 2.0 = 3.5 s at 48 kHz, +/-1 frame per resampled segment.
        self.assertAlmostEqual(r["frames"], 168000, delta=4)
        self.assertAlmostEqual(r["duration"], 3.5, delta=0.001)
        with audio_io.WavReader(out) as rd:
            self.assertEqual(rd.sr, 48000)
            self.assertEqual(rd.channels, 2)
            self.assertEqual(rd.nframes, r["frames"])

    def test_crossfade_duration_overlap(self):
        """Output length = sum of segments - crossfade overlaps."""
        a = self.p("a.wav"); write_sine(a, 44100, 1, 1.0)
        b = self.p("b.wav"); write_sine(b, 44100, 1, 1.0)
        c = self.p("c.wav"); write_sine(c, 44100, 1, 1.0)
        out = self.p("out.wav")
        fade = 0.2
        r = concatenate([{"path": a}, {"path": b}, {"path": c}], out,
                        crossfade_s=fade, curve="linear")
        fade_n = int(fade * 44100)
        self.assertEqual(r["joins"], [fade_n, fade_n])
        self.assertEqual(r["frames"], 3 * 44100 - 2 * fade_n)
        self.assertAlmostEqual(r["duration"], 3.0 - 2 * fade, places=6)

    def test_hard_splice_content_and_order(self):
        """DC levels verify exact sample order and that order is honoured."""
        a = self.p("a.wav"); write_const(a, 44100, 1, 0.2, 0.5)
        b = self.p("b.wav"); write_const(b, 44100, 1, 0.2, -0.5)
        n = int(44100 * 0.2)

        out = self.p("out.wav")
        concatenate([{"path": a}, {"path": b}], out)
        d = audio_io.load(out)
        self.assertEqual(d.frames, 2 * n)
        self.assertAlmostEqual(d.samples[0][0], 0.5, places=3)
        self.assertAlmostEqual(d.samples[0][n - 1], 0.5, places=3)
        self.assertAlmostEqual(d.samples[0][n], -0.5, places=3)
        self.assertAlmostEqual(d.samples[0][-1], -0.5, places=3)

        out2 = self.p("out2.wav")
        concatenate([{"path": b}, {"path": a}], out2)
        d2 = audio_io.load(out2)
        self.assertAlmostEqual(d2.samples[0][0], -0.5, places=3)
        self.assertAlmostEqual(d2.samples[0][n], 0.5, places=3)

    def test_crossfade_content_linear(self):
        """Midpoint of a linear crossfade between +0.5 and -0.5 is ~0."""
        a = self.p("a.wav"); write_const(a, 44100, 1, 0.2, 0.5)
        b = self.p("b.wav"); write_const(b, 44100, 1, 0.2, -0.5)
        out = self.p("out.wav")
        n = int(44100 * 0.2)
        fade_n = int(0.1 * 44100)
        r = concatenate([{"path": a}, {"path": b}], out,
                        crossfade_s=0.1, curve="linear")
        self.assertEqual(r["frames"], 2 * n - fade_n)
        d = audio_io.load(out)
        # The fade region starts at frame n - fade_n; check its midpoint.
        mid = n - fade_n + fade_n // 2
        self.assertAlmostEqual(d.samples[0][mid], 0.0, delta=0.02)
        # Just before / after the fade the signal is essentially untouched.
        self.assertAlmostEqual(d.samples[0][n - fade_n - 10], 0.5, places=2)
        self.assertAlmostEqual(d.samples[0][n + 10], -0.5, places=2)

    def test_crossfade_equal_power_runs(self):
        a = self.p("a.wav"); write_sine(a, 44100, 2, 0.5, 440)
        b = self.p("b.wav"); write_sine(b, 44100, 2, 0.5, 550)
        out = self.p("out.wav")
        r = concatenate([{"path": a}, {"path": b}], out,
                        crossfade_s=0.25, curve="equal_power")
        self.assertEqual(r["frames"], 2 * 22050 - int(0.25 * 44100))
        d = audio_io.load(out)
        peak = max(abs(x) for x in d.samples[0])
        self.assertLessEqual(peak, 1.0)

    def test_tiny_segment_between_long_ones(self):
        """A 30 ms segment between two 2 s segments with a large fade request."""
        long1 = self.p("l1.wav"); write_sine(long1, 44100, 1, 2.0, 440)
        tiny = self.p("t.wav"); write_sine(tiny, 44100, 1, 0.03, 880)
        long2 = self.p("l2.wav"); write_sine(long2, 48000, 2, 2.0, 550)
        out = self.p("out.wav")
        r = concatenate([{"path": long1}, {"path": tiny}, {"path": long2}], out,
                        target_sr=48000, crossfade_s=0.5)
        tiny_out = int(round(0.03 * 48000))  # 1440 frames
        # Each join fade is clamped to half the tiny segment.
        self.assertEqual(r["joins"], [tiny_out // 2, tiny_out // 2])
        expected = 2 * 96000 + tiny_out - 2 * (tiny_out // 2)
        self.assertAlmostEqual(r["frames"], expected, delta=4)
        self.assertAlmostEqual(r["duration"], expected / 48000, delta=0.001)

    def test_many_segments(self):
        """14 segments of varying lengths/rates splice cleanly."""
        paths = []
        for i in range(14):
            p = self.p(f"s{i}.wav")
            write_sine(p, 22050 if i % 2 else 44100, 1 + (i % 2), 0.1 + 0.05 * (i % 4),
                       freq=220 + 20 * i)
            paths.append(p)
        out = self.p("out.wav")
        segs = [{"path": p} for p in paths]
        r = concatenate(segs, out, crossfade_s=0.02)
        expect = sum(0.1 + 0.05 * (i % 4) for i in range(14)) - 13 * 0.02
        self.assertAlmostEqual(r["duration"], expect, delta=0.01)
        with audio_io.WavReader(out) as rd:
            self.assertEqual(rd.nframes, r["frames"])

    def test_segments_from_same_file(self):
        """Multiple excerpts of one file, in a custom order."""
        src = self.p("src.wav"); write_sine(src, 44100, 1, 1.0)
        out = self.p("out.wav")
        r = concatenate([
            {"path": src, "start": 0.5, "end": 0.9},
            {"path": src, "start": 0.0, "end": 0.2},
            {"path": src, "start": 0.1, "end": 0.3},
        ], out)
        self.assertEqual(r["frames"], int(0.4 * 44100) + int(0.2 * 44100) + int(0.2 * 44100))

    def test_excerpt_content_matches_source(self):
        """A spliced excerpt starts at the requested source position."""
        src = self.p("src.wav")
        # Ramp signal: sample i == i / n so position is recoverable.
        n = 44100
        audio_io.save(src, audio_io.AudioData(
            [[(i / n) * 0.8 - 0.4 for i in range(n)]], 44100))
        out = self.p("out.wav")
        concatenate([{"path": src, "start": 0.25, "end": 0.5},
                     {"path": src, "start": 0.75, "end": 1.0}], out)
        d = audio_io.load(out)
        q = int(0.25 * 44100)
        self.assertEqual(d.frames, 2 * q)
        self.assertAlmostEqual(d.samples[0][0], (q / n) * 0.8 - 0.4, places=3)
        self.assertAlmostEqual(d.samples[0][q], (3 * q / n) * 0.8 - 0.4, places=3)

    def test_output_format_overrides(self):
        a = self.p("a.wav"); write_sine(a, 44100, 2, 0.3)
        b = self.p("b.wav"); write_sine(b, 48000, 2, 0.3)
        out = self.p("out.wav")
        r = concatenate([{"path": a}, {"path": b}], out,
                        target_sr=22050, target_ch=1)
        self.assertEqual(r["sr"], 22050)
        self.assertEqual(r["channels"], 1)
        self.assertAlmostEqual(r["duration"], 0.6, delta=0.01)

    def test_mono_to_stereo_upmix(self):
        a = self.p("a.wav"); write_const(a, 44100, 1, 0.1, 0.25)
        b = self.p("b.wav"); write_const(b, 44100, 1, 0.1, -0.25)
        out = self.p("out.wav")
        r = concatenate([{"path": a}, {"path": b}], out, target_ch=2)
        d = audio_io.load(out)
        self.assertEqual(r["channels"], 2)
        self.assertAlmostEqual(d.samples[0][0], 0.25, places=3)
        self.assertAlmostEqual(d.samples[1][0], 0.25, places=3)

    def test_empty_segment_rejected(self):
        a = self.p("a.wav"); write_sine(a, 44100, 1, 0.5)
        b = self.p("b.wav"); write_sine(b, 44100, 1, 0.5)
        with self.assertRaises(ValueError):
            concatenate([{"path": a, "start": 0.2, "end": 0.2},
                         {"path": b}], self.p("out.wav"))
        with self.assertRaises(ValueError):
            concatenate([{"path": a, "start": 5.0},
                         {"path": b}], self.p("out.wav"))

    def test_missing_file_rejected(self):
        a = self.p("a.wav"); write_sine(a, 44100, 1, 0.5)
        with self.assertRaises(ValueError):
            concatenate([{"path": a}, {"path": self.p("nope.wav")}],
                        self.p("out.wav"))

    def test_long_segment_streams(self):
        """A long segment (many chunks) keeps exact length, bounded memory."""
        a = self.p("a.wav"); write_sine(a, 44100, 1, 6.0, 330)
        b = self.p("b.wav"); write_sine(b, 44100, 1, 0.2, 660)
        out = self.p("out.wav")
        r = concatenate([{"path": a}, {"path": b}, {"path": a}], out,
                        crossfade_s=0.1)
        fade_n = int(0.1 * 44100)
        self.assertEqual(r["frames"], 2 * 6 * 44100 + int(0.2 * 44100) - 2 * fade_n)


# --------------------------------------------------------------------------- #
# API (skipped without Flask)
# --------------------------------------------------------------------------- #

try:
    import flask  # noqa: F401
    HAVE_FLASK = True
except ImportError:
    HAVE_FLASK = False


@unittest.skipUnless(HAVE_FLASK, "Flask not installed — API tests skipped")
class TestConcatApi(ConcatBase):
    def setUp(self):
        super().setUp()
        os.environ["AUDIO_DATA_DIR"] = self.dir
        import importlib
        import app as webapp
        importlib.reload(webapp)
        self.webapp = webapp
        self.client = webapp.app.test_client()

    def tearDown(self):
        os.environ.pop("AUDIO_DATA_DIR", None)
        super().tearDown()

    def _gen(self, kind="sine", sr=44100, dur=1.0, freq=440):
        res = self.client.post("/api/library/generate", json={
            "kind": kind, "sr": sr, "duration": dur, "freq": freq})
        self.assertEqual(res.status_code, 200)
        return res.get_json()

    def test_concat_endpoint(self):
        f1 = self._gen(sr=44100, dur=1.0, freq=440)
        f2 = self._gen(sr=22050, dur=0.5, freq=550)
        res = self.client.post("/api/concat", json={
            "segments": [
                {"file_id": f1["id"]},
                {"file_id": f2["id"], "start": 0.1, "end": 0.4},
                {"file_id": f1["id"], "start": 0.0, "end": 0.5},
            ],
            "crossfade": 0.1,
            "curve": "equal_power",
            "name": "拼接测试.wav",
        })
        self.assertEqual(res.status_code, 200)
        entry = res.get_json()
        self.assertEqual(entry["sr"], 44100)
        # 1.0 + 0.3 + 0.5 - 2*0.1 = 1.6 s
        self.assertAlmostEqual(entry["duration"], 1.6, delta=0.01)
        self.assertEqual(entry["op"], "concat")
        # Playable + downloadable through the normal routes.
        self.assertEqual(self.client.get(f"/api/audio/{entry['id']}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/download/{entry['id']}").status_code, 200)
        # Registered in the library with concat metadata.
        lib = self.client.get("/api/library").get_json()
        hit = [f for f in lib if f["id"] == entry["id"]]
        self.assertEqual(len(hit), 1)
        self.assertEqual(hit[0]["concat"]["segments"], 3)

    def test_concat_hard_splice_endpoint(self):
        f1 = self._gen(sr=44100, dur=0.5)
        f2 = self._gen(sr=44100, dur=0.5)
        res = self.client.post("/api/concat", json={
            "segments": [{"file_id": f1["id"]}, {"file_id": f2["id"]}],
            "crossfade": 0,
        })
        self.assertEqual(res.status_code, 200)
        self.assertAlmostEqual(res.get_json()["duration"], 1.0, delta=0.01)

    def test_concat_validation(self):
        f1 = self._gen(sr=44100, dur=0.5)
        # Too few segments.
        res = self.client.post("/api/concat", json={
            "segments": [{"file_id": f1["id"]}]})
        self.assertEqual(res.status_code, 400)
        # Unknown file id.
        res = self.client.post("/api/concat", json={
            "segments": [{"file_id": f1["id"]}, {"file_id": "deadbeef"}]})
        self.assertEqual(res.status_code, 404)
        # Empty excerpt.
        res = self.client.post("/api/concat", json={
            "segments": [
                {"file_id": f1["id"], "start": 0.2, "end": 0.2},
                {"file_id": f1["id"]}]})
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()
