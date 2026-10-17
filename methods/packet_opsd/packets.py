"""Reproduce VideoSEAL's *planned* frame packet without decoding candidate pixels.

The captured sampler uses requested timestamps as labels. Its OpenCV seek,
15-read preheat and read-after-position test do NOT establish that
round(timestamp * source_fps) is the decoded frame index or PTS. Consequently
the default contract is explicitly a planned-frame-packet approximation.

An exact identity contract requires a complete, backend-identified output trace
for every requested effective span. Unsupported exact candidates raise, rather
than silently becoming invalid actions or falling back to the approximation.
These traces describe frames emitted by the original sampler, before its JPEG
resize stages; they must be collected by the caller from actual execution.
This module never opens video/image files and does not certify JPEG bytes or
the Inspector's subsequent vision-processor transformation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Optional, Sequence, Tuple


PLANNED = "planned-frame-packet-approximation"
TRACED = "decoder-trace-verified-frame-identity"


class UnsupportedDecoderError(ValueError):
    """Reject the whole state when an exact decoder contract cannot be met."""


class PacketConformanceError(ValueError):
    """The factual packet did not agree with the metadata prediction."""


@dataclass(frozen=True)
class MediaSpec:
    media_id: str
    duration_s: float
    source_fps: float
    width: int
    height: int
    frame_count: Optional[int] = None
    decoder_identity: Optional[str] = None

    def __post_init__(self):
        if not self.media_id:
            raise ValueError("media_id must identify the immutable source asset")
        if not math.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError("duration_s must be finite and nonnegative")
        if not math.isfinite(self.source_fps) or self.source_fps < 0:
            raise ValueError("source_fps must be finite and nonnegative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("decoded-orientation width and height are required")
        if self.frame_count is not None and self.frame_count < 0:
            raise ValueError("frame_count must be nonnegative")

    @property
    def effective_fps(self) -> float:
        # Exact fallback in sample_uniform_frames, not a CFR identity claim.
        return self.source_fps or 25.0


@dataclass(frozen=True)
class SamplerSpec:
    fps: float = 2.0
    max_total: int = 64
    min_width_s: float = 15.0
    base_max_edge: int = 1280
    dynamic_resize: bool = True
    total_pixels: int = 20971520
    min_pixels: int = 12544
    edge_multiple: int = 32
    global_order: bool = True
    selection_mode: str = "released_round_robin"

    def __post_init__(self):
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.max_total <= 0 or self.min_width_s <= 0:
            raise ValueError("positive frame budget and minimum span width required")
        if self.selection_mode not in ("", "released_round_robin"):
            raise ValueError("Only the captured prefix/round-robin mode is supported")
        if min(self.base_max_edge, self.total_pixels, self.min_pixels) < 0:
            raise ValueError("resize limits cannot be negative")


@dataclass(frozen=True)
class DecodeFrame:
    requested_timestamp_s: float
    source_frame_index: int
    source_pts_s: float
    width: int
    height: int


@dataclass(frozen=True)
class SpanDecodeTrace:
    effective_span: Tuple[float, float]
    frames: Tuple[DecodeFrame, ...]


@dataclass(frozen=True)
class DecoderTrace:
    media_id: str
    decoder_identity: str
    spans: Tuple[SpanDecodeTrace, ...]
    complete: bool = False


@dataclass(frozen=True)
class FrameSpec:
    # The occurrence label is sent to the Inspector, not actual decoded PTS.
    requested_timestamp_s: float
    request_target_index: int
    # A shared frame_{timestamp:.3f}.jpg path can be overwritten by a later span.
    target_frame_index: int
    payload_requested_timestamp_s: float
    file_key: str
    span_index: int
    span_frame_index: int
    final_width: int
    final_height: int
    source_frame_index: Optional[int] = None
    source_pts_s: Optional[float] = None


@dataclass(frozen=True)
class PacketSpec:
    media: Optional[MediaSpec]
    frames: Tuple[FrameSpec, ...] = ()
    effective_spans: Tuple[Tuple[float, float], ...] = ()
    max_total: int = 64
    contract: str = PLANNED
    valid: bool = True
    reason: Optional[str] = None
    # Optional caller-provided Inspector presentation fingerprint. No pixels.
    presentation_id: str = ""

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def timestamps(self) -> Tuple[float, ...]:
        return tuple(f.requested_timestamp_s for f in self.frames)

    @property
    def sizes(self) -> Tuple[Tuple[int, int], ...]:
        return tuple((f.final_width, f.final_height) for f in self.frames)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FactualValidation:
    accepted: bool
    reason: Optional[str]
    contract: str


def invalid_packet(reason: str, *, media: Optional[MediaSpec] = None,
                   max_total: int = 64, presentation_id: str = "") -> PacketSpec:
    """One explicit invalid-action symbol; diagnostic reasons are not features."""
    return PacketSpec(media=media, max_total=max_total, valid=False,
                      reason=str(reason), presentation_id=presentation_id)


def normalize_span(media: MediaSpec, start: float, end: float,
                   min_width_s: float = 15.0) -> Tuple[float, float]:
    """Copy normalize_window_min_width, using the supplied duration metadata."""
    s, e = float(start), float(end)
    if not math.isfinite(s) or not math.isfinite(e):
        raise ValueError("span endpoints must be finite")
    if e <= s:
        e = s + 1e-3
    duration = media.duration_s
    width = min_width_s if duration <= 0 else min(min_width_s, duration)
    if e - s < width:
        center = .5 * (s + e)
        s, e = center - .5 * width, center + .5 * width
    if duration > 0:
        if s < 0:
            s, e = 0.0, min(duration, max(width, 1e-3))
        if e > duration:
            e, s = duration, max(0.0, duration - max(width, 1e-3))
    if e <= s:
        e = s + 1e-3 if duration <= 0 else min(duration, s + 1e-3)
    return float(max(0.0, s)), float(max(s + 1e-3, e))


def _targets(media: MediaSpec, span: Tuple[float, float], policy: SamplerSpec):
    a, b = span
    count = max(int((b - a) * policy.fps), 1)
    period = 1.0 / policy.fps
    times = [a + i * period for i in range(count)
             if a + i * period < b]
    if not times:
        epsilon = 1 / max(media.effective_fps, 1.0)
        times = [min(max(a, a), max(a, b - epsilon))]
    indices = sorted({max(0, int(round(t * media.effective_fps))) for t in times})
    # _ts_for_idx ties retain the first requested time (strict < in original).
    return [(index, min(times, key=lambda t: abs(index - int(round(t * media.effective_fps)))))
            for index in indices[:policy.max_total]]


def _resize(width: int, height: int, edge: int, *, first: bool):
    if edge <= 0 or max(width, height) <= edge:
        return width, height
    if width >= height:
        other = int(round(height * (edge / width))) if first else max(1, (height * edge) // width)
        return edge, other
    other = int(round(width * (edge / height))) if first else max(1, (width * edge) // height)
    return other, edge


def _action_time(value: Any) -> float:
    """Accept the original parser's numeric/time forms without its error-to-zero fallback."""
    if value is None or isinstance(value, bool):
        raise ValueError("missing or invalid timestamp")
    text = str(value).strip().replace(",", ".")
    # The released parser also strips annotations (e.g. '<00:00:15>').
    # Unknown text must not silently turn an invalid candidate into [0, 15].
    if not text:
        raise ValueError("missing timestamp")
    parts = text.split(":")
    try:
        if len(parts) == 3:
            result = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            result = int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1:
            result = float(text)
        else:
            raise ValueError("invalid timestamp")
    except (ValueError, OverflowError):
        stripped = re.sub(r"[^0-9:.,]", "", str(value))
        if not stripped or stripped == text:
            raise ValueError("invalid timestamp") from None
        return _action_time(stripped)
    if not math.isfinite(result):
        raise ValueError("non-finite timestamp")
    return result


def build_action_packet(media: MediaSpec, spans: Optional[Sequence[dict]],
                        policy: Optional[SamplerSpec] = None, *,
                        decoder_trace: Optional[DecoderTrace] = None,
                        require_decoder_trace: bool = False,
                        presentation_id: str = "") -> PacketSpec:
    """Parse released start_time/end_time fields, preserving malformed action mass.

    Valid timestamp forms follow srt_timestamp_to_seconds. Missing, malformed or
    non-finite fields are an explicit invalid packet, rather than that helper's
    silent zero fallback. Candidate generation must not resample these actions.
    """
    policy = policy or SamplerSpec()
    try:
        if spans is None or isinstance(spans, (str, bytes, dict)) or not spans:
            raise ValueError("no spans parsed")
        numeric = []
        for span in spans:
            if not isinstance(span, dict) or "start_time" not in span or "end_time" not in span:
                raise ValueError("span requires start_time and end_time")
            numeric.append((_action_time(span["start_time"]), _action_time(span["end_time"])))
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        return invalid_packet(str(exc), media=media, max_total=policy.max_total,
                              presentation_id=presentation_id)
    return build_packet(media, numeric, sampler=policy, decoder_trace=decoder_trace,
                        require_decoder_trace=require_decoder_trace,
                        presentation_id=presentation_id)


def build_packet(media: MediaSpec, spans: Sequence[Sequence[float]], *,
                 sampler: Optional[SamplerSpec] = None,
                 decoder_trace: Optional[DecoderTrace] = None,
                 require_decoder_trace: bool = False,
                 presentation_id: str = "") -> PacketSpec:
    """Plan numeric spans; caller must use the original action/time parser first.

    In planned mode no decode success, frame ID, PTS or JPEG identity is assumed.
    Exact mode only replays complete traces for identical effective span calls.
    It never extrapolates a factual trace to an unexecuted candidate window.
    """
    policy = sampler or SamplerSpec()
    try:
        windows = tuple(normalize_span(media, s[0], s[1], policy.min_width_s) for s in spans)
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        return invalid_packet(str(exc), media=media, max_total=policy.max_total,
                              presentation_id=presentation_id)
    if not windows:
        return invalid_packet("no spans parsed", media=media, max_total=policy.max_total,
                              presentation_id=presentation_id)
    traced = decoder_trace is not None
    if require_decoder_trace and not traced:
        raise UnsupportedDecoderError("Exact mode requires a complete decoder trace; CFR metadata is insufficient")
    trace_map = {}
    if traced:
        trace = decoder_trace
        if (not trace.complete or not media.decoder_identity or not trace.decoder_identity
                or trace.media_id != media.media_id or trace.decoder_identity != media.decoder_identity):
            raise UnsupportedDecoderError("Incomplete, foreign or unbound decoder identity")
        for item in trace.spans:
            key = tuple(item.effective_span)
            if key in trace_map and trace_map[key] != item.frames:
                raise UnsupportedDecoderError("Conflicting traces for the same effective span")
            trace_map[key] = item.frames
        if any(w not in trace_map for w in windows):
            raise UnsupportedDecoderError("Candidate contains an untraced effective span")

    per_span = []
    writes = {}
    for span_index, window in enumerate(windows):
        targets = _targets(media, window, policy)
        target_by_time = {timestamp: index for index, timestamp in targets}
        if traced:
            outputs = trace_map[window]
            if len(outputs) > policy.max_total:
                raise UnsupportedDecoderError("Trace exceeds the original per-span output cap")
            sequence = []
            for output_index, frame in enumerate(outputs):
                t = frame.requested_timestamp_s
                if (t not in target_by_time or t != targets[output_index][1] or frame.source_frame_index < 0
                        or not math.isfinite(frame.source_pts_s) or frame.source_pts_s < 0
                        or frame.width <= 0 or frame.height <= 0
                        or (media.frame_count is not None and frame.source_frame_index >= media.frame_count)):
                    raise UnsupportedDecoderError("Trace does not match the original emitted-frame contract")
                sequence.append((target_by_time[t], t, frame))
        else:
            sequence = [(index, timestamp, None) for index, timestamp in targets]
        occurrences = []
        for local_index, (index, timestamp, decoded) in enumerate(sequence):
            key = f"frame_{timestamp:.3f}.jpg"
            dims = (decoded.width, decoded.height) if decoded else (media.width, media.height)
            dims = _resize(*dims, policy.base_max_edge, first=True)
            if min(dims) <= 0:
                return invalid_packet("original base resize has a zero dimension", media=media,
                                      max_total=policy.max_total, presentation_id=presentation_id)
            writes[key] = (index, timestamp, dims, decoded)
            occurrences.append((timestamp, index, key, span_index, local_index))
        per_span.append(occurrences)

    # First select per-span prefixes by round robin, then flatten by span.
    selected = [[] for _ in per_span]
    if sum(map(len, per_span)) <= policy.max_total:
        selected = [list(items) for items in per_span]
    else:
        offsets = [0] * len(per_span)
        taken = 0
        while taken < policy.max_total:
            progressed = False
            for i, items in enumerate(per_span):
                if taken >= policy.max_total:
                    break
                if offsets[i] < len(items):
                    selected[i].append(items[offsets[i]])
                    offsets[i] += 1
                    taken += 1
                    progressed = True
            if not progressed:
                break
    packed = [item for items in selected for item in items]
    if policy.global_order:
        packed.sort(key=lambda item: item[0])  # stable: equal labels retain span order
    if not packed:
        return invalid_packet("no frames extracted for spans", media=media,
                              max_total=policy.max_total, presentation_id=presentation_id)

    final_dims = {key: value[2] for key, value in writes.items()}
    if policy.dynamic_resize and policy.total_pixels > 0:
        w, h = final_dims[packed[0][2]]
        aspect = max(w, h, 1) / max(1, min(w, h))
        pixels = max(policy.total_pixels / len(packed), policy.min_pixels)
        edge = int(math.floor(math.sqrt(pixels * max(1.0, aspect))))
        if policy.edge_multiple > 1:
            edge = max(policy.edge_multiple, (edge // policy.edge_multiple) * policy.edge_multiple)
        if policy.base_max_edge > 0:
            edge = min(edge, policy.base_max_edge)
        if edge > 0 and not (policy.base_max_edge > 0 and edge >= policy.base_max_edge):
            for key in {item[2] for item in packed}:
                final_dims[key] = _resize(*final_dims[key], edge, first=False)

    frames = []
    for timestamp, request_index, key, span_index, local_index in packed:
        index, image_time, _, decoded = writes[key]
        width, height = final_dims[key]
        frames.append(FrameSpec(timestamp, request_index, index, image_time, key,
                                span_index, local_index, width, height,
                                decoded.source_frame_index if decoded else None,
                                decoded.source_pts_s if decoded else None))
    return PacketSpec(media, tuple(frames), windows, policy.max_total,
                      TRACED if traced else PLANNED, presentation_id=presentation_id)


def validate_factual_packet(packet: PacketSpec, actual_timestamps: Sequence[float], *,
                            actual_sizes: Optional[Sequence[Sequence[int]]] = None,
                            actual_frame_indices: Optional[Sequence[int]] = None,
                            actual_pts: Optional[Sequence[float]] = None,
                            decoder_identity: Optional[str] = None,
                            require_decoder_identity: bool = False,
                            atol: float = 1e-6) -> FactualValidation:
    """Reject a state on any observed mismatch; labels alone never certify PTS."""
    def fail(reason):
        return FactualValidation(False, reason, packet.contract)
    if not packet.valid:
        return fail("Factual packet cannot be an invalid action")
    if not math.isfinite(atol) or atol < 0:
        return fail("Invalid timestamp tolerance")
    if len(actual_timestamps) != packet.frame_count:
        return fail("Factual frame count differs")
    if any(not math.isfinite(float(a)) or abs(float(a) - b) > atol
           for a, b in zip(actual_timestamps, packet.timestamps)):
        return fail("Factual requested timestamp labels differ")
    if actual_sizes is not None and tuple(tuple(x) for x in actual_sizes) != packet.sizes:
        return fail("Factual post-resize sizes differ")
    if require_decoder_identity and packet.contract != TRACED:
        return fail("Only planned-frame approximation is available")
    if packet.contract == TRACED:
        if actual_sizes is None:
            return fail("Factual post-resize sizes are missing")
        if decoder_identity != packet.media.decoder_identity:
            return fail("Factual decoder identity differs or is missing")
        if actual_frame_indices is None or tuple(actual_frame_indices) != tuple(f.source_frame_index for f in packet.frames):
            return fail("Factual decoded frame indices differ or are missing")
        if actual_pts is None or len(actual_pts) != packet.frame_count:
            return fail("Factual decoded PTS are missing")
        if any(not math.isfinite(float(a)) or abs(float(a) - f.source_pts_s) > atol
               for a, f in zip(actual_pts, packet.frames)):
            return fail("Factual decoded PTS differ")
    return FactualValidation(True, None, packet.contract)


def require_factual_match(packet: PacketSpec, actual_timestamps: Sequence[float], **kwargs) -> None:
    result = validate_factual_packet(packet, actual_timestamps, **kwargs)
    if not result.accepted:
        raise PacketConformanceError(result.reason)


def _features(packet: PacketSpec) -> Tuple[float, ...]:
    media = packet.media
    time_scale = max(media.duration_s, 1.0)
    index_scale = max(media.frame_count or media.duration_s * media.effective_fps, 1.0)
    features = [packet.frame_count / packet.max_total]
    for f in packet.frames:
        identity = f.source_frame_index if packet.contract == TRACED else f.target_frame_index
        pts = f.source_pts_s if packet.contract == TRACED else f.payload_requested_timestamp_s
        features.extend((1.0, f.requested_timestamp_s / time_scale, identity / index_scale,
                         pts / time_scale, f.final_width / 2048.0, f.final_height / 2048.0))
    features.extend([0.0] * (6 * (packet.max_total - packet.frame_count)))
    return tuple(features)


def gaussian_packet_kernel(a: PacketSpec, b: PacketSpec, *, bandwidth: float = 1.0) -> float:
    """Gaussian on the complete ordered, multiplicity-preserving metadata packet.

    Count and occupied padding slots distinguish a duplicate frame from no frame.
    The finite metadata encoding is injective on the declared packet identity;
    a Gaussian is therefore characteristic on this fixed-media packet space.
    It is not a claim that nearby timestamps have nearby semantic task utility.
    Planned and traced identities are separate domains; no unseen pixels enter.
    """
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be positive and finite")
    if not a.valid or not b.valid:
        return float(not a.valid and not b.valid)
    if a.frame_count > a.max_total or b.frame_count > b.max_total:
        raise ValueError("Packet exceeds its declared frame budget")
    if a.max_total != b.max_total:
        raise ValueError("Compare packets under the same declared frame budget")
    if (a.media != b.media or a.contract != b.contract
            or a.presentation_id != b.presentation_id):
        return 0.0
    distance = math.fsum((x - y) ** 2 for x, y in zip(_features(a), _features(b)))
    return math.exp(-distance / (2 * bandwidth ** 2))


def kernel_matrix(packets: Sequence[PacketSpec], *, bandwidth: float = 1.0):
    return [[gaussian_packet_kernel(a, b, bandwidth=bandwidth) for b in packets]
            for a in packets]
