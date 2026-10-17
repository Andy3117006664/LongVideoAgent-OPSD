from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageStat

from .schema import CandidateSpan


def extract_packet(
    helper: object,
    video: Path,
    spans: tuple[CandidateSpan, ...],
    directory: Path,
    *,
    fps: float,
    cap: int,
) -> tuple[list[Path], list[float], int]:
    _, _, frames, timestamps, _ = helper.extract_production_packet(
        video,
        [(span.start, span.end) for span in spans],
        fps=fps,
        cap=cap,
        directory=directory,
    )
    if not frames:
        raise RuntimeError("frame extraction returned an empty packet")
    edge = int(helper.apply_formal_dynamic_resize(frames))
    return frames, [float(value) for value in timestamps], edge


def make_pixel_nulls(frames: list[Path], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    output: list[Path] = []
    for index, source in enumerate(frames):
        with Image.open(source) as raw:
            image = raw.convert("RGB")
        mean = tuple(round(value) for value in ImageStat.Stat(image).mean)
        target = directory / f"null_{index:03d}.jpg"
        Image.new("RGB", image.size, mean).save(target, format="JPEG", quality=90)
        output.append(target)
    return output

