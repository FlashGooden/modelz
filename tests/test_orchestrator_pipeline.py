from unittest.mock import patch

import pytest

from modelz import orchestrator
from modelz.errors import StageFailedError


@pytest.fixture(autouse=True)
def isolated_jobs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "JOBS_DIR", tmp_path)
    yield tmp_path


def test_dry_run_only_runs_prep_stage(tiny_image, tiny_video):
    with patch("modelz.orchestrator.stage1_motion.run") as mock_motion, patch(
        "modelz.orchestrator.stage2_lipsync.run"
    ) as mock_lipsync, patch("modelz.orchestrator.stage3_postprocess.mux") as mock_mux:
        orchestrator.run_pipeline(
            avatar_image=tiny_image,
            driving_video=tiny_video,
            job_id="jobA",
            out_path=None,
            dry_run=True,
        )

    mock_motion.assert_not_called()
    mock_lipsync.assert_not_called()
    mock_mux.assert_not_called()
    meta = orchestrator.load_meta("jobA")
    assert meta["stages"]["prep"]["status"] == "done"


def test_full_pipeline_runs_all_stages_in_order(tiny_image, tiny_video, monkeypatch):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "r8_test")

    def fake_motion(appearance_image, motion_video, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"motion")
        return dest

    def fake_lipsync(video, audio, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"lipsync")
        return dest

    def fake_mux(src, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"final")
        return dest

    with patch(
        "modelz.orchestrator.stage1_motion.run", side_effect=fake_motion
    ) as mock_motion, patch(
        "modelz.orchestrator.stage2_lipsync.run", side_effect=fake_lipsync
    ) as mock_lipsync, patch(
        "modelz.orchestrator.stage3_postprocess.mux", side_effect=fake_mux
    ) as mock_mux:
        result = orchestrator.run_pipeline(
            avatar_image=tiny_image,
            driving_video=tiny_video,
            job_id="jobB",
            out_path=None,
            dry_run=False,
        )

    mock_motion.assert_called_once()
    mock_lipsync.assert_called_once()
    mock_mux.assert_called_once()
    assert result.read_bytes() == b"final"


def test_resume_skips_already_done_motion_stage(tiny_image, tiny_video, monkeypatch, tmp_path):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "r8_test")
    jdir = orchestrator.job_dir("jobC")
    jdir.mkdir(parents=True)
    existing_motion = jdir / "stage1_motion.mp4"
    existing_motion.write_bytes(b"already done")
    (jdir / "audio.aac").write_bytes(b"audio")

    meta = orchestrator.load_meta("jobC")
    orchestrator.mark_stage(meta, "prep", "done", audio=str(jdir / "audio.aac"))
    orchestrator.mark_stage(
        meta, "motion", "done", output=str(existing_motion), model="zsxkib/mimic-motion"
    )
    orchestrator.save_meta("jobC", meta)

    def fake_lipsync(video, audio, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"lipsync")
        return dest

    def fake_mux(src, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"final")
        return dest

    with patch("modelz.orchestrator.stage1_motion.run") as mock_motion, patch(
        "modelz.orchestrator.stage2_lipsync.run", side_effect=fake_lipsync
    ), patch("modelz.orchestrator.stage3_postprocess.mux", side_effect=fake_mux):
        orchestrator.run_pipeline(
            avatar_image=tiny_image,
            driving_video=tiny_video,
            job_id="jobC",
            out_path=None,
            dry_run=False,
        )

    mock_motion.assert_not_called()


def test_motion_stage_failure_marks_meta_and_raises(tiny_image, tiny_video, monkeypatch):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "r8_test")

    with patch(
        "modelz.orchestrator.stage1_motion.run", side_effect=RuntimeError("boom")
    ):
        with pytest.raises(StageFailedError):
            orchestrator.run_pipeline(
                avatar_image=tiny_image,
                driving_video=tiny_video,
                job_id="jobD",
                out_path=None,
                dry_run=False,
            )

    meta = orchestrator.load_meta("jobD")
    assert meta["stages"]["motion"]["status"] == "failed"


def test_postprocess_stage_failure_marks_meta_and_raises(tiny_image, tiny_video, monkeypatch):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "r8_test")

    def fake_motion(appearance_image, motion_video, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"motion")
        return dest

    def fake_lipsync(video, audio, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"lipsync")
        return dest

    with patch(
        "modelz.orchestrator.stage1_motion.run", side_effect=fake_motion
    ), patch(
        "modelz.orchestrator.stage2_lipsync.run", side_effect=fake_lipsync
    ), patch(
        "modelz.orchestrator.stage3_postprocess.mux", side_effect=RuntimeError("mux boom")
    ):
        with pytest.raises(StageFailedError):
            orchestrator.run_pipeline(
                avatar_image=tiny_image,
                driving_video=tiny_video,
                job_id="jobE",
                out_path=None,
                dry_run=False,
            )

    meta = orchestrator.load_meta("jobE")
    assert meta["stages"]["postprocess"]["status"] == "failed"
