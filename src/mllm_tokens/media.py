"""Read lightweight source-file metadata for media inputs.

Only container headers are read: no frames are decoded and no audio samples
are loaded. A missing file yields no properties; an unreadable file yields a
``probe_error`` entry instead of raising, because analysis itself has already
succeeded by the time media is probed.
"""

from pathlib import Path
from typing import Any


def probe_media(path: str | Path, modality: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {}

    probes = {"image": _probe_image, "video": _probe_video, "audio": _probe_audio}
    probe = probes.get(modality)
    if probe is None:
        return {}

    properties: dict[str, Any] = {"file_bytes": path.stat().st_size}
    try:
        properties.update(probe(path))
    except Exception as error:  # noqa: BLE001 - decoder libraries raise widely
        properties["probe_error"] = f"{type(error).__name__}: {error}"

    return properties


def _probe_image(path: Path) -> dict[str, Any]:
    import PIL.Image as PImage

    with PImage.open(path) as image:
        return {"width": image.width, "height": image.height, "mode": image.mode}


def _probe_video(path: Path) -> dict[str, Any]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate

        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / 1_000_000
        else:
            duration = None

        properties: dict[str, Any] = {
            "width": stream.codec_context.width,
            "height": stream.codec_context.height,
            "source_fps": round(float(rate), 3) if rate else None,
            "source_frames": int(stream.frames) or None,
            "duration_seconds": round(duration, 3) if duration else None,
        }

    return properties


def _probe_audio(path: Path) -> dict[str, Any]:
    import librosa

    return {
        "sample_rate": librosa.get_samplerate(str(path)),
        "duration_seconds": round(librosa.get_duration(path=str(path)), 3),
    }
