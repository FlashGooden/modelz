import subprocess

import pytest

from modelz.errors import InputValidationError, StageFailedError
from modelz.pipeline import stage0_prep


def test_validate_image_accepts_real_image(tiny_image):
    stage0_prep.validate_image(tiny_image)  # should not raise


def test_validate_image_rejects_missing_file(tmp_path):
    with pytest.raises(InputValidationError):
        stage0_prep.validate_image(tmp_path / "missing.jpg")


def test_validate_image_rejects_non_media_file(not_a_video):
    with pytest.raises(InputValidationError):
        stage0_prep.validate_image(not_a_video)


def test_validate_video_accepts_real_video(tiny_video):
    stage0_prep.validate_video(tiny_video)  # should not raise


def test_validate_video_rejects_missing_file(tmp_path):
    with pytest.raises(InputValidationError):
        stage0_prep.validate_video(tmp_path / "missing.mp4")


def test_validate_video_rejects_non_media_file(not_a_video):
    with pytest.raises(InputValidationError):
        stage0_prep.validate_video(not_a_video)


def test_extract_audio_creates_file_with_audio_stream(tiny_video, tmp_path):
    dest = tmp_path / "out" / "audio.aac"

    result = stage0_prep.extract_audio(tiny_video, dest)

    assert result == dest
    assert dest.exists()
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(dest),
        ],
        capture_output=True, text=True,
    )
    assert "audio" in probe.stdout


def test_extract_audio_raises_stage_failed_error_on_ffmpeg_failure(not_a_video, tmp_path):
    dest = tmp_path / "out" / "audio.aac"

    with pytest.raises(StageFailedError):
        stage0_prep.extract_audio(not_a_video, dest)
