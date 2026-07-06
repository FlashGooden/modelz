import subprocess
from pathlib import Path

from ..errors import InputValidationError


def _is_readable_media(path: Path) -> bool:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and "video" in result.stdout


def validate_image(path: Path) -> None:
    if not path.exists():
        raise InputValidationError(f"Avatar image not found: {path}")
    if not _is_readable_media(path):
        raise InputValidationError(f"File is not a readable image: {path}")


def validate_video(path: Path) -> None:
    if not path.exists():
        raise InputValidationError(f"Driving video not found: {path}")
    if not _is_readable_media(path):
        raise InputValidationError(f"File is not a readable video: {path}")


def extract_audio(driving_video: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(driving_video), "-vn", "-acodec", "aac", str(dest)],
        capture_output=True, check=True,
    )
    return dest
