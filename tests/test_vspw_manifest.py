from driftguard.datasets.vspw import inspect_vspw_structure


def test_missing_vspw_root_is_reported_without_fabricating_samples(tmp_path):
    report = inspect_vspw_structure(tmp_path / "missing", split="val")
    assert report["exists"] is False
    assert report["video_count"] == 0
    assert report["sample_count"] == 0
    assert report["warnings"]
