from pathlib import Path
from unittest.mock import patch

from modelz import cli
from modelz.errors import StageFailedError


def test_build_parser_requires_avatar_and_driving():
    parser = cli.build_parser()
    args = parser.parse_args(["generate", "--avatar", "a.jpg", "--driving", "d.mp4"])

    assert args.avatar == Path("a.jpg")
    assert args.driving == Path("d.mp4")
    assert args.dry_run is False
    assert args.job_id is None


def test_main_success_prints_job_id_and_output(tmp_path, capsys):
    with patch(
        "modelz.cli.orchestrator.new_job_id", return_value="job123"
    ), patch(
        "modelz.cli.orchestrator.run_pipeline", return_value=tmp_path / "final.mp4"
    ) as mock_run:
        exit_code = cli.main(["generate", "--avatar", "a.jpg", "--driving", "d.mp4"])

    assert exit_code == 0
    mock_run.assert_called_once()
    captured = capsys.readouterr()
    assert "job_id=job123" in captured.out
    assert "final.mp4" in captured.out


def test_main_failure_prints_error_and_resume_hint(capsys):
    with patch(
        "modelz.cli.orchestrator.new_job_id", return_value="job456"
    ), patch(
        "modelz.cli.orchestrator.run_pipeline",
        side_effect=StageFailedError("Stage 1 (motion) failed: boom"),
    ):
        exit_code = cli.main(["generate", "--avatar", "a.jpg", "--driving", "d.mp4"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Stage 1 (motion) failed: boom" in captured.err
    assert "--resume job456" in captured.err
