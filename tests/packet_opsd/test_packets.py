"""CPU-only conformance tests against captured source, without opening media."""
from __future__ import annotations

import ast
from dataclasses import replace
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

from methods.packet_opsd.packets import (
    PLANNED, TRACED, DecodeFrame, DecoderTrace, MediaSpec, PacketSpec,
    PacketConformanceError, SamplerSpec, SpanDecodeTrace, UnsupportedDecoderError,
    build_action_packet, build_packet, gaussian_packet_kernel, invalid_packet,
    kernel_matrix, normalize_span, require_factual_match, validate_factual_packet,
)


ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "baselines/videoseal_6000d/source/baselines/VideoSEAL/videoseal/utils/video"


def captured_function(filename, name, **namespace):
    """Compile only one captured function; avoid importing model/video modules."""
    tree = ast.parse((VIDEO / filename).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(VIDEO / filename), "exec"), namespace)
    return namespace[name]


class ScriptedCapture:
    """Ideal CFR read-after index backend, for exercising the actual sampler."""
    def __init__(self, fps):
        self.fps, self.position = fps, 0

    def isOpened(self):
        return True

    def get(self, prop):
        return self.fps if prop == "fps" else self.position

    def set(self, prop, value):
        self.position = int(value)

    def read(self):
        identity = self.position
        self.position += 1
        return True, SimpleNamespace(identity=identity, shape=(1080, 1920, 3))

    def release(self):
        pass


def captured_sample(source_fps, start, end, *, fps=2.0, max_frames=64):
    capture, written = ScriptedCapture(source_fps), []
    cv = SimpleNamespace(VideoCapture=lambda path: capture, CAP_PROP_FPS="fps",
                         CAP_PROP_POS_FRAMES="position", CAP_PROP_POS_MSEC="msec",
                         imwrite=lambda path, frame: written.append((path, frame.identity)))
    fake_os = SimpleNamespace(path=os.path, getenv=lambda key, default=None: default)
    sample = captured_function("frames.py", "sample_uniform_frames", cv2=cv,
                               np=SimpleNamespace(isfinite=math.isfinite), os=fake_os,
                               _ensure_dir=lambda path: None)
    paths, timestamps = sample("unused.mp4", start, end, fps=fps, max_frames=max_frames,
                               output_dir="unused")
    return paths, timestamps, [identity for _, identity in written]


class PacketConformanceTests(unittest.TestCase):
    def setUp(self):
        self.media = MediaSpec("immutable-video-1", 600.0, 25.0, 1920, 1080, 15000)

    def test_normalization_matches_captured_helper(self):
        rng = random.Random(413)
        for duration in (0.0, .0001, 4.0, 15.0, 600.0):
            media = replace(self.media, duration_s=duration)
            original = captured_function("tooling.py", "normalize_window_min_width",
                                         get_video_duration=lambda path: duration)
            for _ in range(40):
                a, b = rng.uniform(-30, 650), rng.uniform(-30, 650)
                self.assertEqual(normalize_span(media, a, b), original("video", a, b))

    def test_requested_grid_matches_actual_source_execution(self):
        # Exercise original code with an index-only backend, not a copy of its grid.
        for source_fps in (.5, 12., 25., 29.97):
            for span in ((0., 15.), (10., 25.9), (90., 240.)):
                with self.subTest(source_fps=source_fps, span=span):
                    media = replace(self.media, source_fps=source_fps)
                    packet = build_packet(media, [span])
                    _, labels, _ = captured_sample(source_fps, *span)
                    self.assertEqual(packet.timestamps, tuple(labels))

    def test_nondefault_rate_uses_original_multiply_by_period(self):
        policy = SamplerSpec(fps=3.0)
        packet = build_packet(self.media, [(10.1, 35.7)], sampler=policy)
        _, labels, _ = captured_sample(25., 10.1, 35.7, fps=3.0)
        self.assertEqual(packet.timestamps, tuple(labels))

    def test_read_after_index_is_not_the_requested_target(self):
        _, labels, identities = captured_sample(25., 0., 15.)
        self.assertEqual(labels[0], 0.)
        self.assertEqual(identities[0], 1)  # Preheat already consumed frame zero.
        _, labels, identities = captured_sample(25., 10., 25.)
        self.assertEqual(labels[0], 10.)
        self.assertEqual(identities[0], 249)  # read-after position 250.
        packet = build_packet(self.media, [(10., 25.)])
        self.assertEqual(packet.frames[0].target_frame_index, 250)
        self.assertIsNone(packet.frames[0].source_frame_index)
        self.assertIsNone(packet.frames[0].source_pts_s)
        self.assertEqual(packet.contract, PLANNED)

    def test_long_span_is_a_prefix_and_budget_is_shared(self):
        one = build_packet(self.media, [(0., 500.)])
        two = build_packet(self.media, [(0., 500.), (300., 500.)])
        self.assertEqual(one.frame_count, 64)
        self.assertEqual(one.timestamps[-1], 31.5)
        self.assertEqual(sum(f.span_index == 0 for f in two.frames), 32)
        self.assertEqual(max(f.requested_timestamp_s for f in two.frames if f.span_index == 0), 15.5)
        # Adding a span changes the old view, rather than appending evidence.
        self.assertNotEqual(one.timestamps[:64], two.timestamps)

    def test_selection_matches_captured_round_robin_across_cases(self):
        original = captured_function("tooling.py", "balanced_select_frames",
                                     _temporal_coverage_mode=lambda: False)
        spans = [(100., 200.), (0., 25.), (10., 30.)]
        for budget in (1, 5, 31, 64, 90):
            for global_order in (False, True):
                policy = SamplerSpec(max_total=budget, global_order=global_order)
                individual = [build_packet(self.media, [s], sampler=policy) for s in spans]
                per_frames = [[f"{i}:{j}" for j in range(p.frame_count)] for i, p in enumerate(individual)]
                per_ts = [list(p.timestamps) for p in individual]
                paths, labels = original(per_span_frames=per_frames, per_span_ts=per_ts,
                                         span_windows=spans, max_total=budget,
                                         fps_fallback=2., global_order=global_order)
                packet = build_packet(self.media, spans, sampler=policy)
                self.assertEqual([f"{f.span_index}:{f.span_frame_index}" for f in packet.frames], paths)
                self.assertEqual(packet.timestamps, tuple(labels))

    def test_rounding_dedup_and_fps_fallback(self):
        media = replace(self.media, source_fps=.5)
        packet = build_packet(media, [(0., 15.)])
        self.assertEqual([f.request_target_index for f in packet.frames], list(range(8)))
        self.assertEqual(packet.timestamps[:3], (0., 1.5, 3.))
        fallback = build_packet(replace(self.media, source_fps=0.), [(0., 15.)])
        self.assertEqual(fallback.frames[1].target_frame_index, 12)  # Python ties-to-even.
        self.assertEqual(fallback.contract, PLANNED)

    def test_duplicate_occurrences_and_stable_global_order_are_retained(self):
        packet = build_packet(self.media, [(0., 15.), (0., 15.)])
        self.assertEqual(packet.frame_count, 60)
        self.assertEqual(packet.timestamps[:4], (0., 0., .5, .5))
        self.assertEqual([f.span_index for f in packet.frames[:4]], [0, 1, 0, 1])
        self.assertEqual(packet.frames[0].file_key, packet.frames[1].file_key)

    def test_resize_matches_both_original_rounding_stages(self):
        # Base: 1920x1080 -> 1280x720. Dynamic: 64 images -> edge 736,
        # with the second stage's integer-floor short side = 414.
        packet = build_packet(self.media, [(0., 100.)])
        self.assertEqual(packet.sizes, ((736, 414),) * 64)
        small = build_packet(replace(self.media, width=321, height=199), [(0., 100.)])
        self.assertEqual(small.sizes, ((321, 199),) * 64)
        odd = build_packet(replace(self.media, width=1921, height=1081), [(0., 100.)],
                           sampler=SamplerSpec(dynamic_resize=False))
        self.assertEqual(odd.sizes[0], (1280, 720))

    def test_zero_dimension_original_resize_is_explicit_invalid(self):
        packet = build_packet(replace(self.media, width=1, height=100000), [(0., 15.)])
        self.assertFalse(packet.valid)

    def test_training_duration_override_matches_original_reader(self):
        import sys
        from unittest.mock import patch

        before_import = set(sys.modules)
        from methods.packet_opsd.train import media_spec
        self.assertNotIn("torch", set(sys.modules) - before_import)
        metadata = {"fps": 25., "count": 15000., "width": 1920., "height": 1080.}
        fake_cv = SimpleNamespace(
            VideoCapture=lambda path: SimpleNamespace(
                isOpened=lambda: True, get=lambda prop: metadata[prop], release=lambda: None),
            CAP_PROP_FPS="fps", CAP_PROP_FRAME_COUNT="count",
            CAP_PROP_FRAME_WIDTH="width", CAP_PROP_FRAME_HEIGHT="height",
        )
        original = captured_function("io.py", "get_video_duration", os=os, cv2=fake_cv)
        for override in (None, "", 0, -5, "invalid", "nan", 12.5, " 22.5 ", 600.):
            with self.subTest(override=override), patch.dict(sys.modules, {"cv2": fake_cv}), patch.dict(os.environ):
                task = {"video_id": "video", "video_path": "unused.mp4"}
                if override is None:
                    os.environ.pop("VIDEO_DURATION_SEC", None)
                else:
                    task["video_duration_sec"] = override
                    os.environ["VIDEO_DURATION_SEC"] = str(override)
                self.assertEqual(media_spec(task).duration_s, original("unused.mp4"))

    def test_action_parser_matches_valid_original_forms(self):
        original = captured_function("time.py", "srt_timestamp_to_seconds")
        for a, b in (("00:00:10,500", "00:01:10.5"), ("01:30", "03:00"),
                     ("10.5", 50), ("<00:00:20>", "<00:00:40>")):
            packet = build_action_packet(self.media, [{"start_time": a, "end_time": b}])
            numeric = build_packet(self.media, [(original(str(a)), original(str(b)))])
            self.assertEqual(packet, numeric)

    def test_malformed_actions_remain_invalid_without_resampling(self):
        bad = [None, [], {}, "spans", [None], [(0, 15)], [{"start_time": 0}],
               [{"end_time": 10}], [{"start_time": "", "end_time": 10}],
               [{"start_time": "junk", "end_time": 10}],
               [{"start_time": float("nan"), "end_time": 10}],
               [{"start_time": 0, "end_time": "inf"}]]
        for action in bad:
            with self.subTest(action=action):
                self.assertFalse(build_action_packet(self.media, action).valid)
        self.assertFalse(build_packet(self.media, [(float("inf"), 15)]).valid)
        self.assertFalse(build_packet(self.media, None).valid)


class DecoderContractTests(unittest.TestCase):
    def setUp(self):
        self.media = MediaSpec("video", 100., 25., 1920, 1080, 2500, "opencv:build:backend:asset-sha")
        self.policy = SamplerSpec(max_total=2, dynamic_resize=False)
        self.outputs = (DecodeFrame(0., 1, .04, 1920, 1080), DecodeFrame(.5, 11, .44, 1920, 1080))
        self.trace = DecoderTrace("video", self.media.decoder_identity,
                                  (SpanDecodeTrace((0., 15.), self.outputs),), complete=True)

    def packet(self, **kwargs):
        return build_packet(self.media, [(0., 15.)], sampler=self.policy,
                            decoder_trace=kwargs.get("trace", self.trace), require_decoder_trace=True)

    def test_exact_mode_rejects_absence_incomplete_foreign_and_unseen_spans(self):
        with self.assertRaises(UnsupportedDecoderError):
            build_packet(self.media, [(0., 15.)], require_decoder_trace=True)
        for trace in (replace(self.trace, complete=False), replace(self.trace, media_id="other"),
                      replace(self.trace, decoder_identity="other")):
            with self.assertRaises(UnsupportedDecoderError):
                self.packet(trace=trace)
        with self.assertRaises(UnsupportedDecoderError):
            build_packet(self.media, [(1., 16.)], sampler=self.policy, decoder_trace=self.trace)

    def test_actual_identity_is_not_fabricated_from_timestamp(self):
        packet = self.packet()
        self.assertEqual(packet.contract, TRACED)
        self.assertEqual(packet.frames[0].request_target_index, 0)
        self.assertEqual(packet.frames[0].source_frame_index, 1)
        self.assertEqual(packet.frames[1].source_pts_s, .44)
        good = dict(actual_sizes=packet.sizes, actual_frame_indices=(1, 11), actual_pts=(.04, .44),
                    decoder_identity=self.media.decoder_identity)
        self.assertTrue(validate_factual_packet(packet, packet.timestamps, **good).accepted)
        for key, wrong in (("actual_sizes", ((1, 1), (1, 1))), ("actual_frame_indices", (0, 12)),
                           ("actual_pts", (0., .5)), ("decoder_identity", "other")):
            values = dict(good, **{key: wrong})
            self.assertFalse(validate_factual_packet(packet, packet.timestamps, **values).accepted)
        self.assertFalse(validate_factual_packet(packet, packet.timestamps).accepted)

    def test_trace_cannot_skip_an_earlier_target(self):
        bad = replace(self.trace, spans=(SpanDecodeTrace((0., 15.), self.outputs[1:]),))
        with self.assertRaises(UnsupportedDecoderError):
            self.packet(trace=bad)

    def test_shared_file_later_writer_controls_every_occurrence(self):
        later = (DecodeFrame(0., 101, 4.04, 640, 480), DecodeFrame(.5, 111, 4.44, 640, 480))
        trace = replace(self.trace, spans=self.trace.spans + (SpanDecodeTrace((0., 16.), later),))
        packet = build_packet(self.media, [(0., 15.), (0., 16.)], sampler=self.policy, decoder_trace=trace)
        self.assertEqual(packet.timestamps, (0., 0.))
        self.assertEqual([f.source_frame_index for f in packet.frames], [101, 101])
        self.assertEqual(packet.sizes, ((640, 480), (640, 480)))
        self.assertEqual([f.span_index for f in packet.frames], [0, 1])

    def test_mismatch_rejects_state_and_labels_cannot_certify_decoded_identity(self):
        packet = build_packet(self.media, [(0., 15.)], sampler=self.policy)
        self.assertTrue(validate_factual_packet(packet, packet.timestamps).accepted)
        self.assertFalse(validate_factual_packet(packet, packet.timestamps, require_decoder_identity=True).accepted)
        for labels in ([], [0., .6], [float("nan"), .5]):
            self.assertFalse(validate_factual_packet(packet, labels).accepted)
        with self.assertRaises(PacketConformanceError):
            require_factual_match(packet, [0., .6])


class WholePacketKernelTests(unittest.TestCase):
    def setUp(self):
        self.media = MediaSpec("video", 100., 25., 1920, 1080)
        self.base = build_packet(self.media, [(0., 15.)], sampler=SamplerSpec(max_total=2))
        self.a, self.b = self.base.frames

    def packet(self, frames):
        return replace(self.base, frames=tuple(frames))

    def test_kernel_is_sensitive_to_order_multiplicity_count_and_dimensions(self):
        base = self.base
        self.assertEqual(gaussian_packet_kernel(base, base), 1.)
        variants = [self.packet([self.b, self.a]), self.packet([self.a, self.a]), self.packet([self.a]),
                    self.packet([replace(self.a, final_width=400), self.b])]
        for variant in variants:
            self.assertLess(gaussian_packet_kernel(base, variant), 1.)
            self.assertGreater(gaussian_packet_kernel(base, variant), 0.)

    def test_same_frame_marginals_can_have_different_packet_laws(self):
        # Both laws have P(frame=A)=P(frame=B)=1/2 at either slot; their
        # co-occurrence differs. A sum of frame marginals cannot detect this.
        laws = [self.packet([self.a, self.a]), self.packet([self.b, self.b]),
                self.packet([self.a, self.b]), self.packet([self.b, self.a])]
        weights = [.5, .5, -.5, -.5]
        gram = kernel_matrix(laws, bandwidth=.01)
        mmd2 = sum(weights[i] * weights[j] * gram[i][j] for i in range(4) for j in range(4))
        self.assertGreater(mmd2, 0.)

    def test_invalid_is_one_explicit_symbol_and_domains_are_separate(self):
        first, second = invalid_packet("parse"), invalid_packet("truncated")
        self.assertEqual(gaussian_packet_kernel(first, second), 1.)
        self.assertEqual(gaussian_packet_kernel(first, self.base), 0.)
        self.assertEqual(gaussian_packet_kernel(self.base, replace(self.base, presentation_id="changed")), 0.)
        self.assertEqual(gaussian_packet_kernel(self.base, replace(self.base, contract=TRACED)), 0.)
        for bad in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                gaussian_packet_kernel(self.base, self.base, bandwidth=bad)

    def test_kernel_matrix_is_symmetric_psd(self):
        packets = [self.base, self.packet([self.b, self.a]), self.packet([self.a]), invalid_packet("bad")]
        gram = kernel_matrix(packets)
        rng = random.Random(9)
        for i in range(len(gram)):
            for j in range(len(gram)):
                self.assertEqual(gram[i][j], gram[j][i])
        for _ in range(20):
            weights = [rng.uniform(-1, 1) for _ in packets]
            form = sum(weights[i] * weights[j] * gram[i][j] for i in range(4) for j in range(4))
            self.assertGreaterEqual(form, -1e-12)


if __name__ == "__main__":
    unittest.main()
