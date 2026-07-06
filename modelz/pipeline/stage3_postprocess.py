import subprocess
from pathlib import Path


def mux(src: Path, dest: Path) -> Path:
    """Normalize the final container/format (re-mux without re-encoding)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-c", "copy", str(dest)],
        capture_output=True, check=True,
    )
    return dest
