import zipfile

from ppmat.utils.download import _uncompress_file_zip


def test_uncompress_single_directory_uses_top_level_name(tmp_path):
    archive_path = tmp_path / "model.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("model/checkpoints/best.pdparams", b"weights")
        archive.writestr("model/model.yaml", b"Model: {}")

    extracted_path = _uncompress_file_zip(str(archive_path))

    assert extracted_path == str(tmp_path / "model" / "model")
    assert (tmp_path / "model" / "model" / "model.yaml").exists()
