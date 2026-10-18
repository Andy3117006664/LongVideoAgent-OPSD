"""Bound the auxiliary's gradient relative to the factual OPSD gradient.

Scores remain sequence sums. This is an adaptive optimization rule, not an
unbiased gradient estimator of one fixed weighted scalar MMD objective.
"""
from __future__ import annotations

import math


def packet_gradient_scale(reference_norm: float, packet_norm: float, max_ratio: float) -> float:
    if not all(math.isfinite(v) and v >= 0 for v in (reference_norm, packet_norm, max_ratio)):
        raise ValueError("finite nonnegative gradient norms and ratio are required")
    if packet_norm == 0:
        return 1.0
    return min(1.0, max_ratio * reference_norm / packet_norm)


def bounded_packet_gradients(parameters, reference, *, max_ratio: float) -> dict:
    """Combine saved token-KL gradients with current, pure weighted packet grads.

    No gradient of the norm/scale is taken. A zero reference gradient disables
    the auxiliary for this update. Measure the actual contribution after FP32
    accumulation; a nonzero coefficient alone is not evidence of an update.
    """
    import torch
    parameters = list(parameters)
    if len(parameters) != len(reference):
        raise ValueError("reference must cover exactly the supplied parameters")
    with torch.no_grad():
        packets = [None if p.grad is None else p.grad.detach().clone() for p in parameters]

        def norm(values):
            squares = [v.double().square().sum() for v in values if v is not None]
            return float(torch.stack(squares).sum().sqrt()) if squares else 0.0

        ref_norm, raw_norm = norm(reference), norm(packets)
        scale = packet_gradient_scale(ref_norm, raw_norm, float(max_ratio))
        products = [(a.double() * b.double()).sum() for a, b in zip(reference, packets)
                    if a is not None and b is not None]
        dot = float(torch.stack(products).sum()) if products else 0.0
        differences = []
        for parameter, old, packet in zip(parameters, reference, packets):
            if old is None and packet is None:
                parameter.grad = None
                continue
            combined = torch.zeros_like(parameter) if old is None else old.clone()
            if packet is not None:
                combined.add_(packet, alpha=scale)
            parameter.grad = combined
            differences.append(combined.double() if old is None else combined.double() - old.double())
        actual_norm = norm(differences)
        bound = float(max_ratio) * ref_norm
        if actual_norm > bound * (1 + 2e-5) + 1e-12:
            raise RuntimeError("packet gradient exceeded its configured relative bound")
        return {
            "raw_weighted_packet_gradient_norm": raw_norm,
            "packet_gradient_scale": scale,
            "packet_gradient_bound": bound,
            "packet_gradient_clipped": scale < 1.0,
            "packet_kl_gradient_cosine": dot / (ref_norm * raw_norm) if ref_norm and raw_norm else None,
            "packet_gradient_norm": actual_norm,
            "actual_accumulated_packet_delta_norm": actual_norm,
            "gradient_before_packet_norm": ref_norm,
            "packet_gradient_ratio": actual_norm / ref_norm if ref_norm else None,
        }
