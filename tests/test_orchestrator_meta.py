from modelz import orchestrator


def test_new_job_id_is_unique():
    assert orchestrator.new_job_id() != orchestrator.new_job_id()


def test_load_meta_returns_default_structure_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "JOBS_DIR", tmp_path)

    meta = orchestrator.load_meta("job123")

    assert meta["job_id"] == "job123"
    for stage in ("prep", "motion", "lipsync", "postprocess"):
        assert meta["stages"][stage]["status"] == "pending"


def test_save_and_load_meta_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "JOBS_DIR", tmp_path)
    meta = orchestrator.load_meta("job123")
    meta["stages"]["prep"]["status"] = "done"

    orchestrator.save_meta("job123", meta)
    reloaded = orchestrator.load_meta("job123")

    assert reloaded["stages"]["prep"]["status"] == "done"


def test_mark_stage_sets_status_and_extra_fields():
    meta = {"stages": {"motion": {"status": "pending"}}}

    orchestrator.mark_stage(
        meta, "motion", "done", output="foo.mp4", model="zsxkib/mimic-motion"
    )

    assert meta["stages"]["motion"] == {
        "status": "done",
        "output": "foo.mp4",
        "model": "zsxkib/mimic-motion",
    }


def test_stage_done_true_only_when_status_done_and_output_file_exists(tmp_path):
    output_file = tmp_path / "out.mp4"
    output_file.write_bytes(b"data")
    meta = {"stages": {"motion": {"status": "done", "output": str(output_file)}}}

    assert orchestrator.stage_done(meta, "motion", "output") is True

    output_file.unlink()
    assert orchestrator.stage_done(meta, "motion", "output") is False
