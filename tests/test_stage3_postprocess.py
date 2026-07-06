import subprocess

import pytest

from modelz.errors import StageFailedError
from modelz.pipeline import stage3_postprocess


def test_mux_produces_a_playable_output_file(tiny_video, tmp_path):
    dest = tmp_path / "out" / "final.mp4"

    result = stage3_postprocess.mux(tiny_video, dest)

    assert result == dest
    assert dest.exists()
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(dest),
        ],
        capture_output=True, text=True,
    )
    assert probe.returncode == 0
    assert probe.stdout.strip() != ""


def test_mux_raises_stage_failed_error_on_ffmpeg_failure(not_a_video, tmp_path):
    dest = tmp_path / "out" / "final.mp4"

    with pytest.raises(StageFailedError):
        stage3_postprocess.mux(not_a_video, dest)
