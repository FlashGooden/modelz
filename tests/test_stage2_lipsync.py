from unittest.mock import patch

from modelz.pipeline import stage2_lipsync


def test_run_calls_replicate_client_with_expected_inputs(tmp_path):
    video = tmp_path / "stage1_motion.mp4"
    audio = tmp_path / "audio.aac"
    dest = tmp_path / "stage2_lipsync.mp4"

    with patch(
        "modelz.pipeline.stage2_lipsync.replicate_client.run_model"
    ) as mock_run, patch(
        "modelz.pipeline.stage2_lipsync.replicate_client.download"
    ) as mock_download:
        mock_run.return_value = "https://replicate.delivery/lipsync.mp4"
        mock_download.return_value = dest

        result = stage2_lipsync.run(video, audio, dest)

    mock_run.assert_called_once_with(
        stage2_lipsync.MODEL_ID, {"video": video, "audio": audio}
    )
    mock_download.assert_called_once_with("https://replicate.delivery/lipsync.mp4", dest)
    assert result == dest
