from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_checkpoint_inventory.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_inventory", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sha256_file_and_exact_classification(tmp_path: Path) -> None:
    target = tmp_path / "model.safetensors"
    target.write_bytes(b"checkpoint-test")
    expected = hashlib.sha256(b"checkpoint-test").hexdigest()
    assert MODULE.sha256_file(target) == expected
    assert MODULE.classify_hash(MODULE.EXPECTED_HASHES["EXACT_B0"], str(target)) == "EXACT_B0"
    assert MODULE.classify_hash(MODULE.EXPECTED_HASHES["EXACT_B1"], str(target)) == "EXACT_B1"


def test_base_and_mismatch_classification() -> None:
    base = next(iter(MODULE.KNOWN_BASE_HASHES))
    assert MODULE.classify_hash(base, "segformer-b0-ade/model.safetensors") == "BASE_MODEL_ONLY"
    assert MODULE.classify_hash("0" * 64, "segformer-b0-vspw124/model.safetensors") == "HASH_MISMATCH"
    assert MODULE.classify_hash(None, "archive.tar.gz", is_archive=True) == "UNKNOWN_CANDIDATE"


def test_archive_member_filter_and_directory_exclusion() -> None:
    assert MODULE.suspicious_archive_member("models/segformer-b0-vspw124/model.safetensors")
    assert not MODULE.suspicious_archive_member("data/VSPW/frame.png")
    assert MODULE.is_excluded(Path("D:/datasets/VSPW_480p/data/a"), MODULE.DEFAULT_EXCLUDED_DIRS)
