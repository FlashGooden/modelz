import subprocess
from pathlib import Path

from ..errors import StageFailedError


def mux(src: Path, dest: Path) -> Path:
    """Normalize the final container/format (re-mux without re-encoding)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-c", "copy", str(dest)],
            capture_output=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise StageFailedError(
            f"Failed to mux {src} to {dest}: {exc.stderr.decode(errors='replace') if exc.stderr else exc}"
        ) from exc
    return dest
