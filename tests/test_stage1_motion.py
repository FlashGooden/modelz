from unittest.mock import patch

from modelz.pipeline import stage1_motion


def test_run_calls_replicate_client_with_expected_inputs(tmp_path):
    appearance_image = tmp_path / "avatar.jpg"
    motion_video = tmp_path / "driving.mp4"
    dest = tmp_path / "stage1_motion.mp4"

    with patch(
        "modelz.pipeline.stage1_motion.replicate_client.run_model"
    ) as mock_run, patch(
        "modelz.pipeline.stage1_motion.replicate_client.download"
    ) as mock_download:
        mock_run.return_value = "https://replicate.delivery/motion.mp4"
        mock_download.return_value = dest

        result = stage1_motion.run(appearance_image, motion_video, dest)

    mock_run.assert_called_once_with(
        stage1_motion.MODEL_ID,
        {"appearance_image": appearance_image, "motion_video": motion_video},
    )
    mock_download.assert_called_once_with("https://replicate.delivery/motion.mp4", dest)
    assert result == dest
