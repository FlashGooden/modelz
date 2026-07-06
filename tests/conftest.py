import subprocess

import pytest


@pytest.fixture
def tiny_image(tmp_path):
    path = tmp_path / "avatar.jpg"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=64x64",
            "-frames:v", "1", str(path),
        ],
        capture_output=True, check=True,
    )
    return path


@pytest.fixture
def tiny_video(tmp_path):
    path = tmp_path / "driving.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path),
        ],
        capture_output=True, check=True,
    )
    return path


@pytest.fixture
def not_a_video(tmp_path):
    path = tmp_path / "not_video.txt"
    path.write_text("hello")
    return path
