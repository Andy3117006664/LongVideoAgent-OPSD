"""Metadata-only contracts for the captured VideoSEAL frame sampler."""

from .packets import (
    DecodeFrame, DecoderTrace, FactualValidation, FrameSpec, MediaSpec,
    PacketConformanceError, PacketSpec, SamplerSpec, SpanDecodeTrace,
    UnsupportedDecoderError, build_action_packet, build_packet, gaussian_packet_kernel,
    invalid_packet, kernel_matrix, normalize_span, require_factual_match,
    validate_factual_packet,
)

__all__ = [
    "DecodeFrame", "DecoderTrace", "FactualValidation", "FrameSpec", "MediaSpec",
    "PacketConformanceError", "PacketSpec", "SamplerSpec", "SpanDecodeTrace",
    "UnsupportedDecoderError", "build_action_packet", "build_packet", "gaussian_packet_kernel",
    "invalid_packet", "kernel_matrix", "normalize_span", "require_factual_match",
    "validate_factual_packet",
]
