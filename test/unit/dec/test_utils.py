from pyodsp.dec.utils import create_directory


def test_create_directory_creates_missing_dirs(tmp_path):
    target = tmp_path / "a" / "b" / "c"

    create_directory(target)

    assert target.is_dir()


def test_create_directory_is_idempotent(tmp_path):
    target = tmp_path / "a"
    target.mkdir()

    create_directory(target)

    assert target.is_dir()


def test_create_directory_swallows_errors(tmp_path, capsys):
    # a file in the path where a directory is expected makes mkdir raise
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    target = blocker / "child"

    create_directory(target)

    assert not target.exists()
    assert "An error occurred" in capsys.readouterr().out
