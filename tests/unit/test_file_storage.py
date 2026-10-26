from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from aevrel.storage.file import FileStorage, StorageError, _is_windows_drive_path


def test_file_storage_read_write_ranges(tmp_path) -> None:
    path = tmp_path / "data.bin"
    storage = FileStorage()
    uri = path.as_uri()

    storage.ensure_size(uri, 12)
    storage.write_range(uri, 2, b"abcd")
    assert storage.read_range(uri, 0, 12) == b"\x00\x00abcd\x00\x00\x00\x00\x00\x00"
    assert storage.size(uri) == 12


def test_file_storage_rejects_short_read(tmp_path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc")
    storage = FileStorage()
    with pytest.raises(StorageError):
        storage.read_range(path.as_uri(), 0, 4)


def test_file_storage_rejects_non_file_uri() -> None:
    storage = FileStorage()
    with pytest.raises(StorageError):
        storage.exists("https://example.com/data.bin")


def test_file_storage_can_fsync_written_file(tmp_path) -> None:
    path = tmp_path / "written.bin"
    storage = FileStorage()
    storage.ensure_size(path.as_uri(), 4)
    storage.write_range(path.as_uri(), 0, b"data")
    storage.fsync(path.as_uri())
    assert path.read_bytes() == b"data"


def test_windows_drive_uri_path_detection() -> None:
    assert _is_windows_drive_path("/C:/Users/demo/data.bin")
    assert not _is_windows_drive_path("/tmp/data.bin")


def _fail_open_for(monkeypatch, target: Path, error: OSError) -> None:
    """Make ``Path.open`` raise ``error`` for ``target`` only."""
    real_open = Path.open

    def fake_open(self, *args, **kwargs):
        if self == target:
            raise error
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fake_open)


def test_read_range_rejects_negative_offset_and_size(tmp_path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abcdef")
    storage = FileStorage()
    for offset, size in ((-1, 2), (0, -1)):
        with pytest.raises(StorageError) as excinfo:
            storage.read_range(path.as_uri(), offset, size)
        assert str(excinfo.value) == "file range must be non-negative"


def test_read_range_zero_size_returns_empty_bytes(tmp_path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abcdef")
    storage = FileStorage()
    assert storage.read_range(path.as_uri(), 0, 0) == b""
    # A zero-size read at the end of the file is still a full read.
    assert storage.read_range(path.as_uri(), 6, 0) == b""


def test_read_range_past_eof_reports_short_read(tmp_path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abcdef")
    storage = FileStorage()

    with pytest.raises(StorageError) as excinfo:
        storage.read_range(path.as_uri(), 4, 4)
    assert str(excinfo.value) == (
        f"short read from {path}: wanted 4 bytes, got 2"
    )

    with pytest.raises(StorageError) as excinfo:
        storage.read_range(path.as_uri(), 100, 1)
    assert str(excinfo.value) == (
        f"short read from {path}: wanted 1 bytes, got 0"
    )


def test_read_range_missing_file_is_storage_error_not_os_error(tmp_path) -> None:
    path = tmp_path / "missing.bin"
    storage = FileStorage()
    with pytest.raises(StorageError) as excinfo:
        storage.read_range(path.as_uri(), 0, 1)
    assert not isinstance(excinfo.value, OSError)
    assert str(excinfo.value).startswith(f"could not read {path}: ")
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_read_range_unreadable_file_is_storage_error(tmp_path, monkeypatch) -> None:
    path = tmp_path / "locked.bin"
    path.write_bytes(b"abcdef")
    _fail_open_for(monkeypatch, path, PermissionError("denied"))
    storage = FileStorage()
    with pytest.raises(StorageError) as excinfo:
        storage.read_range(path.as_uri(), 0, 1)
    assert str(excinfo.value) == f"could not read {path}: denied"
    assert isinstance(excinfo.value.__cause__, PermissionError)


def test_read_range_on_directory_is_storage_error(tmp_path) -> None:
    storage = FileStorage()
    with pytest.raises(StorageError) as excinfo:
        storage.read_range(tmp_path.as_uri(), 0, 1)
    assert str(excinfo.value).startswith(f"could not read {tmp_path}: ")
    assert isinstance(excinfo.value.__cause__, OSError)


def test_write_range_rejects_negative_offset(tmp_path) -> None:
    path = tmp_path / "data.bin"
    storage = FileStorage()
    with pytest.raises(StorageError) as excinfo:
        storage.write_range(path.as_uri(), -1, b"x")
    assert str(excinfo.value) == "file range must be non-negative"
    assert not path.exists()


def test_write_range_creates_parent_directories(tmp_path) -> None:
    path = tmp_path / "a" / "b" / "data.bin"
    FileStorage().write_range(path.as_uri(), 0, b"xyz")
    assert path.read_bytes() == b"xyz"


def test_write_range_overwrites_in_place_and_keeps_other_bytes(tmp_path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abcdefgh")
    storage = FileStorage()
    storage.write_range(path.as_uri(), 2, b"XY")
    assert path.read_bytes() == b"abXYefgh"


def test_write_range_past_end_zero_fills_the_gap(tmp_path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"ab")
    FileStorage().write_range(path.as_uri(), 4, b"cd")
    assert path.read_bytes() == b"ab\x00\x00cd"


def test_write_range_to_new_file_at_offset_zero_fills_the_head(tmp_path) -> None:
    path = tmp_path / "data.bin"
    FileStorage().write_range(path.as_uri(), 3, b"z")
    assert path.read_bytes() == b"\x00\x00\x00z"


def test_write_range_empty_data_creates_empty_file(tmp_path) -> None:
    path = tmp_path / "data.bin"
    FileStorage().write_range(path.as_uri(), 0, b"")
    assert path.is_file()
    assert path.read_bytes() == b""


def test_write_range_os_error_is_storage_error(tmp_path, monkeypatch) -> None:
    path = tmp_path / "data.bin"
    _fail_open_for(monkeypatch, path, PermissionError("denied"))
    storage = FileStorage()
    with pytest.raises(StorageError) as excinfo:
        storage.write_range(path.as_uri(), 0, b"x")
    assert str(excinfo.value) == f"could not write {path}: denied"
    assert isinstance(excinfo.value.__cause__, PermissionError)


def test_write_range_on_directory_is_storage_error(tmp_path) -> None:
    storage = FileStorage()
    with pytest.raises(StorageError) as excinfo:
        storage.write_range(tmp_path.as_uri(), 0, b"x")
    assert str(excinfo.value).startswith(f"could not write {tmp_path}: ")


def test_ensure_size_rejects_negative_size(tmp_path) -> None:
    path = tmp_path / "data.bin"
    with pytest.raises(StorageError) as excinfo:
        FileStorage().ensure_size(path.as_uri(), -1)
    assert str(excinfo.value) == "file size must be non-negative"
    assert not path.exists()


def test_ensure_size_grows_and_shrinks_without_touching_kept_bytes(tmp_path) -> None:
    path = tmp_path / "nested" / "data.bin"
    storage = FileStorage()
    storage.ensure_size(path.as_uri(), 0)
    assert path.read_bytes() == b""
    storage.write_range(path.as_uri(), 0, b"abcdef")
    storage.ensure_size(path.as_uri(), 3)
    assert path.read_bytes() == b"abc"
    storage.ensure_size(path.as_uri(), 5)
    assert path.read_bytes() == b"abc\x00\x00"


def test_ensure_size_os_error_is_storage_error(tmp_path, monkeypatch) -> None:
    path = tmp_path / "data.bin"
    _fail_open_for(monkeypatch, path, OSError("disk full"))
    with pytest.raises(StorageError) as excinfo:
        FileStorage().ensure_size(path.as_uri(), 4)
    assert str(excinfo.value) == f"could not size {path}: disk full"


def test_ensure_size_on_directory_is_storage_error(tmp_path) -> None:
    with pytest.raises(StorageError) as excinfo:
        FileStorage().ensure_size(tmp_path.as_uri(), 4)
    assert str(excinfo.value).startswith(f"could not size {tmp_path}: ")


def test_exists_is_false_for_missing_file_and_for_directory(tmp_path) -> None:
    storage = FileStorage()
    assert not storage.exists((tmp_path / "missing.bin").as_uri())
    assert not storage.exists(tmp_path.as_uri())
    (tmp_path / "data.bin").write_bytes(b"x")
    assert storage.exists((tmp_path / "data.bin").as_uri())


def test_size_missing_file_is_storage_error(tmp_path) -> None:
    path = tmp_path / "missing.bin"
    with pytest.raises(StorageError) as excinfo:
        FileStorage().size(path.as_uri())
    assert str(excinfo.value).startswith(f"could not stat {path}: ")
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_fsync_missing_file_is_storage_error(tmp_path) -> None:
    path = tmp_path / "missing.bin"
    with pytest.raises(StorageError) as excinfo:
        FileStorage().fsync(path.as_uri())
    assert str(excinfo.value).startswith(f"could not fsync {path}: ")


def test_fsync_failure_is_storage_error_and_closes_descriptor(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"x")
    closed: list[int] = []
    real_close = os.close

    def fake_fsync(descriptor: int) -> None:
        raise OSError("io error")

    def recording_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(os, "fsync", fake_fsync)
    monkeypatch.setattr(os, "close", recording_close)
    with pytest.raises(StorageError) as excinfo:
        FileStorage().fsync(path.as_uri())
    assert str(excinfo.value) == f"could not fsync {path}: io error"
    assert len(closed) == 1


def test_checksum_matches_sha256_of_content(tmp_path) -> None:
    path = tmp_path / "data.bin"
    # Larger than the 1 MiB hashing chunk so the loop runs more than once.
    payload = b"0123456789abcdef" * (1024 * 1024 // 16 + 5)
    path.write_bytes(payload)
    assert FileStorage().checksum(path.as_uri()) == (
        "sha256:" + hashlib.sha256(payload).hexdigest()
    )


def test_checksum_of_empty_file(tmp_path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert FileStorage().checksum(path.as_uri()) == (
        "sha256:" + hashlib.sha256(b"").hexdigest()
    )


def test_checksum_missing_file_and_directory_are_storage_errors(tmp_path) -> None:
    storage = FileStorage()
    missing = tmp_path / "missing.bin"
    with pytest.raises(StorageError) as excinfo:
        storage.checksum(missing.as_uri())
    assert str(excinfo.value).startswith(f"could not hash {missing}: ")
    with pytest.raises(StorageError) as excinfo:
        storage.checksum(tmp_path.as_uri())
    assert str(excinfo.value).startswith(f"could not hash {tmp_path}: ")


def test_path_rejects_non_file_scheme_with_uri_in_message() -> None:
    for uri in ("https://example.com/data.bin", "dcpzip:///x", "/plain/path"):
        with pytest.raises(StorageError) as excinfo:
            FileStorage().path(uri)
        assert str(excinfo.value) == f"unsupported storage URI: {uri}"


def test_path_rejects_remote_host() -> None:
    uri = "file://example.com/data.bin"
    with pytest.raises(StorageError) as excinfo:
        FileStorage().path(uri)
    assert str(excinfo.value) == f"file URI must be local: {uri}"


def test_path_accepts_localhost_host(tmp_path) -> None:
    target = tmp_path / "data.bin"
    uri = "file://localhost" + target.as_uri()[len("file://"):]
    assert FileStorage().path(uri) == target


def test_path_decodes_percent_escapes(tmp_path) -> None:
    target = tmp_path / "a b#c.bin"
    assert FileStorage().path(target.as_uri()) == target


def test_relative_uri_without_root_is_rejected() -> None:
    for uri in ("file:data.bin", "file://"):
        with pytest.raises(StorageError) as excinfo:
            FileStorage().path(uri)
        assert str(excinfo.value) == f"relative file URI needs a root: {uri}"


def test_empty_root_is_treated_as_no_root() -> None:
    assert FileStorage("").root is None
    with pytest.raises(StorageError):
        FileStorage("").path("file:data.bin")


def test_relative_uri_resolves_under_root(tmp_path) -> None:
    storage = FileStorage(str(tmp_path))
    assert storage.path("file:sub/data.bin") == tmp_path / "sub" / "data.bin"
    storage.write_range("file:sub/data.bin", 0, b"hi")
    assert (tmp_path / "sub" / "data.bin").read_bytes() == b"hi"
    assert storage.read_range("file:sub/data.bin", 0, 2) == b"hi"


def test_absolute_uri_ignores_root(tmp_path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    target = tmp_path / "data.bin"
    assert FileStorage(str(other)).path(target.as_uri()) == target


def test_windows_drive_path_detection_edge_cases() -> None:
    assert _is_windows_drive_path("/c:/x")
    assert not _is_windows_drive_path("/C")
    assert not _is_windows_drive_path("C:/x")
    assert not _is_windows_drive_path("/1:/x")
    assert not _is_windows_drive_path("")


def _fake_copy_file_range(chunk: int | None = None):
    """Build a stand-in for ``os.copy_file_range`` that works on any platform."""

    def copy(source_fd, target_fd, count, source_offset, target_offset):
        if chunk is not None:
            count = min(count, chunk)
        os.lseek(source_fd, source_offset, os.SEEK_SET)
        data = os.read(source_fd, count)
        os.lseek(target_fd, target_offset, os.SEEK_SET)
        return os.write(target_fd, data)

    return copy


def test_copy_range_rejects_negative_arguments(tmp_path) -> None:
    uri = (tmp_path / "x.bin").as_uri()
    for args in ((-1, 0, 1), (0, -1, 1), (0, 0, -1)):
        with pytest.raises(StorageError) as excinfo:
            FileStorage().copy_range(uri, uri, *args)
        assert str(excinfo.value) == "file copy range must be non-negative"


def test_copy_range_without_os_support_raises_not_implemented(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delattr(os, "copy_file_range", raising=False)
    uri = (tmp_path / "x.bin").as_uri()
    with pytest.raises(NotImplementedError, match="copy_file_range is unavailable"):
        FileStorage().copy_range(uri, uri, 0, 0, 1)


def test_copy_range_missing_source_or_target_is_storage_error(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(os, "copy_file_range", _fake_copy_file_range(), raising=False)
    source = tmp_path / "source.bin"
    target = tmp_path / "target.bin"
    storage = FileStorage()

    with pytest.raises(StorageError) as excinfo:
        storage.copy_range(source.as_uri(), target.as_uri(), 0, 0, 1)
    assert str(excinfo.value).startswith("could not open copy range: ")

    source.write_bytes(b"abc")
    with pytest.raises(StorageError) as excinfo:
        storage.copy_range(source.as_uri(), target.as_uri(), 0, 0, 1)
    assert str(excinfo.value).startswith("could not open copy range: ")
    assert not target.exists()


def test_copy_range_copies_requested_bytes_in_chunks(tmp_path, monkeypatch) -> None:
    # Two bytes per call forces the loop to advance both cursors.
    monkeypatch.setattr(
        os, "copy_file_range", _fake_copy_file_range(chunk=2), raising=False
    )
    source = tmp_path / "source.bin"
    target = tmp_path / "target.bin"
    source.write_bytes(b"abcdefgh")
    target.write_bytes(b"..........")
    FileStorage().copy_range(source.as_uri(), target.as_uri(), 1, 3, 5)
    assert target.read_bytes() == b"...bcdef.."
    assert source.read_bytes() == b"abcdefgh"


def test_copy_range_zero_size_copies_nothing(tmp_path, monkeypatch) -> None:
    def must_not_run(*args):
        raise AssertionError("copy_file_range should not be called")

    monkeypatch.setattr(os, "copy_file_range", must_not_run, raising=False)
    source = tmp_path / "source.bin"
    target = tmp_path / "target.bin"
    source.write_bytes(b"abc")
    target.write_bytes(b"xyz")
    FileStorage().copy_range(source.as_uri(), target.as_uri(), 0, 0, 0)
    assert target.read_bytes() == b"xyz"


def test_copy_range_zero_progress_is_short_copy_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(os, "copy_file_range", lambda *args: 0, raising=False)
    source = tmp_path / "source.bin"
    target = tmp_path / "target.bin"
    source.write_bytes(b"abc")
    target.write_bytes(b"xyz")
    with pytest.raises(StorageError) as excinfo:
        FileStorage().copy_range(source.as_uri(), target.as_uri(), 0, 0, 3)
    assert str(excinfo.value) == (
        f"short copy_file_range from {source} to {target}"
    )
    assert target.read_bytes() == b"xyz"


def test_copy_range_os_error_is_storage_error_and_closes_descriptors(
    tmp_path, monkeypatch
) -> None:
    def failing(*args):
        raise OSError("no space")

    closed: list[int] = []
    real_close = os.close

    def recording_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(os, "copy_file_range", failing, raising=False)
    monkeypatch.setattr(os, "close", recording_close)
    source = tmp_path / "source.bin"
    target = tmp_path / "target.bin"
    source.write_bytes(b"abc")
    target.write_bytes(b"xyz")
    with pytest.raises(StorageError) as excinfo:
        FileStorage().copy_range(source.as_uri(), target.as_uri(), 0, 0, 3)
    assert str(excinfo.value) == f"could not copy {source} to {target}: no space"
    assert len(closed) == 2
