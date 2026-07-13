import zipfile
from pathlib import Path

from ppmat.datasets.md17_dataset import MD17Dataset
from ppmat.utils import download
from ppmat.utils.download import _uncompress_file_zip


def test_uncompress_single_directory_preserves_archive_root(tmp_path):
    archive_path = tmp_path / "model.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("model/", b"")
        archive.writestr("model/checkpoints/best.pdparams", b"weights")
        archive.writestr("model/model.yaml", b"Model: {}")

    extracted_path = _uncompress_file_zip(str(archive_path))

    assert Path(extracted_path) == tmp_path / "model"
    assert (tmp_path / "model" / "model" / "model.yaml").exists()


def test_md17_resolves_legacy_single_directory_return(tmp_path, monkeypatch):
    extraction_root = tmp_path / "md17.tar"
    data_path = extraction_root / "md17" / "md17_aspirin.npz"
    data_path.parent.mkdir(parents=True)
    data_path.touch()
    legacy_return = extraction_root / "first_member"
    monkeypatch.setattr(
        download,
        "get_datasets_path_from_url",
        lambda *_args: str(legacy_return),
    )

    dataset = MD17Dataset.__new__(MD17Dataset)
    resolved_path = dataset._resolve_data_path(
        str(tmp_path / "missing.npz"), "aspirin"
    )

    assert resolved_path == str(data_path)
