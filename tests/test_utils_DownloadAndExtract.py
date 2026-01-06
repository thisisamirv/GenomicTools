#!/usr/bin/env python
import os
import pytest
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch
from utils.DownloadAndExtract import DownloadAndExtract
from utils.LoggingUtils import log

log.setup(level="DEBUG")


@pytest.fixture
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_files(temp_dir):
    return [
        (
            "http://example.com/file1.txt",
            os.path.join(temp_dir, "file1.txt"),
            "Description 1",
        ),
        ("https://example.com/file2.gz", os.path.join(temp_dir, "file2.gz")),
        (
            "ftp://example.com/file3.zip",
            os.path.join(temp_dir, "file3.zip"),
            "Description 3",
        ),
    ]


@pytest.mark.unit
def test_initialization():
    files = [("http://example.com/test.txt", "/tmp/test.txt")]

    with patch.object(DownloadAndExtract, "process_files") as mock_process:
        downloader = DownloadAndExtract.__new__(DownloadAndExtract)
        downloader.__init__(files)

        assert downloader.files_to_download == files
        assert downloader.chunk_size == 8192
        assert downloader.timeout == 30
        assert downloader.retries == 3
        assert downloader.resume
        assert downloader.verify_ssl
        mock_process.assert_called_once()


@pytest.mark.unit
def test_create_session():
    files = [("http://example.com/test.txt", "/tmp/test.txt")]

    with patch.object(DownloadAndExtract, "process_files"):
        downloader = DownloadAndExtract(files)
        session = downloader._create_session()

        assert "User-Agent" in session.headers
        assert (
            session.headers["User-Agent"] == "DownloadAndExtract/1.0 (Python requests)"
        )


pytest.mark.unit


@patch("utils.DownloadAndExtract.Path.mkdir")
@patch("utils.DownloadAndExtract.tqdm")
def test_http_download_success(mock_tqdm, mock_mkdir, temp_dir):
    mock_progress_bar = MagicMock()
    mock_tqdm.return_value.__enter__.return_value = mock_progress_bar

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers.get.return_value = "1024"
    mock_response.iter_content.return_value = [b"test data chunk"]

    downloader = DownloadAndExtract.__new__(DownloadAndExtract)
    downloader.session = MagicMock()
    downloader.session.get.return_value = mock_response
    downloader.chunk_size = 8192
    downloader.timeout = 30
    downloader.retries = 3
    downloader.resume = True
    downloader.verify_ssl = True

    filename = os.path.join(temp_dir, "test.txt")

    with patch("builtins.open", mock_open()) as mock_file:
        mock_file.return_value.__enter__.return_value.write.return_value = 15

        result = downloader._download_http(
            "http://example.com/test.txt", filename, "Test download"
        )

        assert result
        downloader.session.get.assert_called_once()
        mock_progress_bar.update.assert_called()


@pytest.mark.unit
@patch("utils.DownloadAndExtract.FTP")
@patch("utils.DownloadAndExtract.Path.mkdir")
def test_ftp_download_success(mock_mkdir, mock_ftp_class, temp_dir):
    mock_ftp_instance = MagicMock()
    mock_ftp_instance.size.return_value = 1024
    mock_ftp_class.return_value.__enter__.return_value = mock_ftp_instance

    downloader = DownloadAndExtract.__new__(DownloadAndExtract)
    downloader.timeout = 30
    downloader.retries = 3

    filename = os.path.join(temp_dir, "test.txt")

    def mock_retrbinary(command, callback):
        callback(b"test data")

    mock_ftp_instance.retrbinary = mock_retrbinary

    with patch("builtins.open", mock_open()):
        result = downloader._download_ftp(
            "ftp://example.com/test.txt", filename, "Test FTP"
        )

        assert result
        mock_ftp_class.assert_called()
        mock_ftp_instance.login.assert_called()


@pytest.mark.unit
def test_get_archive_type():
    downloader = DownloadAndExtract.__new__(DownloadAndExtract)

    test_cases = [
        ("file.gz", "gzip"),
        ("file.zip", "zip"),
        ("file.tar", "tar"),
        ("file.tar.gz", "tar_gz"),
        ("file.tgz", "tar_gz"),
        ("file.tar.bz2", "tar_bz2"),
        ("file.tbz2", "tar_bz2"),
        ("file.tar.xz", "tar_xz"),
        ("file.txz", "tar_xz"),
        ("file.txt", None),
    ]

    for filename, expected in test_cases:
        result = downloader._get_archive_type(filename)
        assert (
            result == expected
        ), f"Failed for {filename}: expected {expected}, got {result}"


@pytest.mark.unit
@patch("utils.DownloadAndExtract.shutil.copyfileobj")
@patch("utils.DownloadAndExtract.gzip.open")
@patch("builtins.open", new_callable=mock_open)
def test_extract_gzip(mock_file, mock_gzip, mock_copyfileobj, temp_dir):
    downloader = DownloadAndExtract.__new__(DownloadAndExtract)

    gz_file = Path(temp_dir) / "test.gz"

    mock_gzip_context = MagicMock()
    mock_gzip.return_value.__enter__.return_value = mock_gzip_context

    with patch.object(Path, "unlink") as mock_unlink:
        result = downloader._extract_gzip(gz_file)

        expected_path = str(gz_file.with_suffix(""))
        assert result == expected_path
        mock_gzip.assert_called_once_with(gz_file, "rb")
        mock_copyfileobj.assert_called_once()
        mock_unlink.assert_called_once()


@pytest.mark.unit
@patch("utils.DownloadAndExtract.zipfile.ZipFile")
def test_extract_zip(mock_zipfile, temp_dir):
    downloader = DownloadAndExtract.__new__(DownloadAndExtract)

    zip_file = Path(temp_dir) / "test.zip"

    mock_zip = MagicMock()
    mock_zip.infolist.return_value = [MagicMock(file_size=1024)]
    mock_zip.namelist.return_value = ["extracted_file.txt"]
    mock_zipfile.return_value.__enter__.return_value = mock_zip

    with patch.object(Path, "unlink") as mock_unlink, patch.object(
        Path, "rename"
    ) as mock_rename:

        downloader._extract_zip(zip_file)

        mock_zipfile.assert_called_once_with(zip_file, "r")
        mock_zip.extractall.assert_called_once_with(zip_file.parent)
        mock_unlink.assert_called_once()
        mock_rename.assert_called_once()


@pytest.mark.unit
def test_debug_os_commonpath():
    import tempfile
    import os
    from pathlib import Path

    temp_dir = tempfile.mkdtemp()
    base_path = Path(temp_dir)

    member_path = os.path.join(str(base_path), "safe_file.txt")
    result = os.path.commonpath([member_path, str(base_path)])

    print(f"Base path: {base_path}")
    print(f"Member path: {member_path}")
    print(f"Commonpath result: {result}")
    print(f"Are they equal? {result == str(base_path)}")

    import shutil

    shutil.rmtree(temp_dir)


@pytest.mark.unit
@patch("utils.DownloadAndExtract.tarfile.open")
def test_extract_tar_safe(mock_tarfile, temp_dir):
    downloader = DownloadAndExtract.__new__(DownloadAndExtract)

    tar_file = Path(temp_dir) / "test.tar"

    mock_tar = MagicMock()
    mock_member = MagicMock()
    mock_member.name = "safe_file.txt"
    mock_tar.getmembers.return_value = [mock_member]
    mock_tarfile.return_value.__enter__.return_value = mock_tar

    with patch.object(Path, "unlink"), patch(
        "utils.DownloadAndExtract.os.path.commonpath", return_value=str(tar_file.parent)
    ):

        result = downloader._extract_tar(tar_file, "tar")

        assert result == str(tar_file.parent)
        mock_tar.extractall.assert_called_once()


@pytest.mark.unit
@patch("utils.DownloadAndExtract.tarfile.open")
def test_extract_tar_unsafe(mock_tarfile, temp_dir):
    downloader = DownloadAndExtract.__new__(DownloadAndExtract)

    tar_file = Path(temp_dir) / "test.tar"

    mock_tar = MagicMock()
    mock_member = MagicMock()
    mock_member.name = "../../etc/passwd"
    mock_tar.getmembers.return_value = [mock_member]
    mock_tarfile.return_value.__enter__.return_value = mock_tar

    with patch(
        "utils.DownloadAndExtract.os.path.commonpath", return_value="/different/path"
    ):
        with pytest.raises(Exception, match="Unsafe path in archive"):
            downloader._extract_tar(tar_file, "tar")


@pytest.mark.unit
@patch("utils.DownloadAndExtract.tarfile.open")
@patch("os.path.commonpath")
def test_extract_tar_modes(mock_commonpath, mock_tarfile, temp_dir):
    downloader = DownloadAndExtract.__new__(DownloadAndExtract)

    tar_file = Path(temp_dir) / "test.tar"
    base_path = str(tar_file.parent)

    mock_tar = MagicMock()
    mock_tar.getmembers.return_value = []
    mock_tarfile.return_value.__enter__.return_value = mock_tar

    mock_commonpath.return_value = base_path

    test_cases = [
        ("tar", "r"),
        ("tar_gz", "r:gz"),
        ("tar_bz2", "r:bz2"),
        ("tar_xz", "r:xz"),
    ]

    with patch.object(Path, "unlink"):
        for archive_type, expected_mode in test_cases:
            mock_tarfile.reset_mock()

            result = downloader._extract_tar(tar_file, archive_type)

            assert result == base_path
            mock_tarfile.assert_called_with(tar_file, expected_mode)


@pytest.mark.unit
def test_extract_tar_invalid_type(temp_dir):
    downloader = DownloadAndExtract.__new__(DownloadAndExtract)
    tar_file = Path(temp_dir) / "test.tar"

    with pytest.raises(Exception, match="Unknown tar archive type"):
        downloader._extract_tar(tar_file, "invalid_type")


@pytest.mark.unit
@patch("utils.DownloadAndExtract.tarfile.open")
def test_extract_tar_blocks_unsafe_paths(mock_tarfile, temp_dir):
    downloader = DownloadAndExtract.__new__(DownloadAndExtract)

    tar_file = Path(temp_dir) / "test.tar"

    mock_tar = MagicMock()
    mock_member = MagicMock()
    mock_member.name = "../../etc/passwd"
    mock_tar.getmembers.return_value = [mock_member]
    mock_tarfile.return_value.__enter__.return_value = mock_tar

    def mock_commonpath(paths):
        return "/different/unsafe/path"

    with patch(
        "utils.DownloadAndExtract.os.path.commonpath", side_effect=mock_commonpath
    ):
        with pytest.raises(Exception, match="Unsafe path in archive"):
            downloader._extract_tar(tar_file, "tar")


@pytest.mark.unit
def test_unsupported_url_scheme(temp_dir):
    files = [("unsupported://example.com/file.txt", os.path.join(temp_dir, "file.txt"))]

    with patch.object(DownloadAndExtract, "extract_file"):
        DownloadAndExtract(files)
        assert True


@pytest.mark.unit
@patch("requests.Session.get")
def test_http_download_with_resume(mock_get, temp_dir):
    filename = os.path.join(temp_dir, "test.txt")
    with open(filename, "wb") as f:
        f.write(b"partial")

    mock_response = MagicMock()
    mock_response.status_code = 206
    mock_response.headers.get.return_value = "512"
    mock_response.iter_content.return_value = [b" content"]
    mock_get.return_value = mock_response

    downloader = DownloadAndExtract.__new__(DownloadAndExtract)
    downloader.session = MagicMock()
    downloader.session.get.return_value = mock_response
    downloader.chunk_size = 8192
    downloader.timeout = 30
    downloader.retries = 3
    downloader.resume = True
    downloader.verify_ssl = True

    result = downloader._download_http(
        "http://example.com/test.txt", filename, "Resume test"
    )

    assert result
    call_args = downloader.session.get.call_args
    assert "Range" in call_args[1]["headers"]


@pytest.mark.unit
@patch("utils.DownloadAndExtract.tqdm")
@patch("requests.Session.get")
def test_http_download_retry_logic(mock_get, mock_tqdm, temp_dir):
    import requests

    mock_progress_bar = MagicMock()
    mock_tqdm.return_value.__enter__.return_value = mock_progress_bar

    mock_response_success = MagicMock()
    mock_response_success.status_code = 200
    mock_response_success.headers.get.return_value = "1024"
    mock_response_success.iter_content.return_value = [b"data"]

    mock_get.side_effect = [
        requests.exceptions.ConnectionError("Connection failed"),
        requests.exceptions.ConnectionError("Connection failed"),
        mock_response_success,
    ]

    downloader = DownloadAndExtract.__new__(DownloadAndExtract)
    downloader.session = MagicMock()
    downloader.session.get = mock_get
    downloader.chunk_size = 8192
    downloader.timeout = 30
    downloader.retries = 3
    downloader.resume = False
    downloader.verify_ssl = True

    filename = os.path.join(temp_dir, "test.txt")

    with patch("builtins.open", mock_open()) as mock_file:
        mock_file.return_value.__enter__.return_value.write.return_value = 4

        result = downloader._download_http(
            "http://example.com/test.txt", filename, "Retry test"
        )

        assert result
        assert mock_get.call_count == 3


@pytest.mark.unit
def test_extract_file_not_archive(temp_dir):
    downloader = DownloadAndExtract.__new__(DownloadAndExtract)

    filename = os.path.join(temp_dir, "test.txt")
    result = downloader.extract_file(filename)

    assert result is None


@pytest.mark.unit
def test_extraction_error_handling(temp_dir):
    downloader = DownloadAndExtract.__new__(DownloadAndExtract)

    gz_file = os.path.join(temp_dir, "test.gz")

    with patch(
        "utils.DownloadAndExtract.gzip.open", side_effect=Exception("Extraction failed")
    ):
        result = downloader.extract_file(gz_file)

        assert result is None


@pytest.mark.unit
def test_zip_bomb_protection(temp_dir):
    downloader = DownloadAndExtract.__new__(DownloadAndExtract)

    zip_file = Path(temp_dir) / "test.zip"

    with patch("utils.DownloadAndExtract.zipfile.ZipFile") as mock_zipfile:
        mock_zip = MagicMock()
        mock_file_info = MagicMock()
        mock_file_info.file_size = 15 * 1024 * 1024 * 1024
        mock_zip.infolist.return_value = [mock_file_info]
        mock_zipfile.return_value.__enter__.return_value = mock_zip

        with pytest.raises(Exception, match="Archive too large"):
            downloader._extract_zip(zip_file)


@pytest.mark.unit
def test_tar_directory_traversal_protection(temp_dir):
    downloader = DownloadAndExtract.__new__(DownloadAndExtract)

    tar_file = Path(temp_dir) / "test.tar"

    with patch("utils.DownloadAndExtract.tarfile.open") as mock_tarfile:
        mock_tar = MagicMock()
        mock_member = MagicMock()
        mock_member.name = "../../etc/passwd"
        mock_tar.getmembers.return_value = [mock_member]
        mock_tarfile.return_value.__enter__.return_value = mock_tar

        with patch(
            "utils.DownloadAndExtract.os.path.commonpath",
            return_value="/different/path",
        ):
            with pytest.raises(Exception, match="Unsafe path in archive"):
                downloader._extract_tar(tar_file, "tar")


@pytest.mark.unit
def test_invalid_file_info_format(temp_dir):
    invalid_files = [
        ("url_only",),
        ("url", "filename", "desc", "extra"),
        ("", "filename"),
        ("url", ""),
    ]

    with patch.object(
        DownloadAndExtract, "download_file", return_value=False
    ), patch.object(DownloadAndExtract, "extract_file"):

        DownloadAndExtract(invalid_files)
        assert True


@pytest.mark.unit
def test_empty_file_list():
    DownloadAndExtract([])
    assert True


@pytest.mark.integration
def test_end_to_end_workflow(temp_dir):
    files = [
        (
            "http://example.com/test.gz",
            os.path.join(temp_dir, "test.gz"),
            "Test gzip file",
        ),
        ("https://example.com/test.zip", os.path.join(temp_dir, "test.zip")),
    ]

    mock_response_gz = MagicMock()
    mock_response_gz.status_code = 200
    mock_response_gz.headers.get.return_value = "100"
    mock_response_gz.iter_content.return_value = [b"gzip content"]

    mock_response_zip = MagicMock()
    mock_response_zip.status_code = 200
    mock_response_zip.headers.get.return_value = "200"
    mock_response_zip.iter_content.return_value = [b"zip content"]

    with patch(
        "requests.Session.get", side_effect=[mock_response_gz, mock_response_zip]
    ), patch("builtins.open", mock_open()), patch(
        "utils.DownloadAndExtract.gzip.open"
    ), patch(
        "utils.DownloadAndExtract.zipfile.ZipFile"
    ), patch(
        "utils.DownloadAndExtract.shutil.copyfileobj"
    ), patch.object(
        Path, "unlink"
    ), patch.object(
        Path, "rename"
    ):

        DownloadAndExtract(files)
        assert True


@pytest.mark.integration
def test_visual_inspection(temp_dir):
    print("\n==== DOWNLOAD AND EXTRACT VISUAL INSPECTION ====")

    print("Sample file formats supported:")
    file_examples = [
        ("https://example.com/data.txt", "Plain text file"),
        ("https://example.com/archive.zip", "ZIP archive"),
        ("https://example.com/data.tar.gz", "Gzipped tar archive"),
        ("https://example.com/compressed.gz", "Gzip compressed file"),
        ("ftp://ftp.example.com/dataset.tar", "FTP tar archive"),
        ("http://example.com/file.bz2", "Bzip2 compressed file"),
    ]

    for url, description in file_examples:
        print(f"  {url} - {description}")

    print("\nNew Features in Updated Version:")
    print("  - Resume interrupted downloads (HTTP only)")
    print("  - Configurable retry attempts with exponential backoff")
    print("  - Enhanced security (zip bomb + directory traversal protection)")
    print("  - Context manager support for proper cleanup")
    print("  - Better error handling with custom exceptions")
    print("  - Type hints throughout the codebase")

    print("\nConfiguration Options:")
    print("  - chunk_size: Download chunk size (default: 8KB)")
    print("  - timeout: Request timeout (default: 30s)")
    print("  - retries: Retry attempts (default: 3)")
    print("  - resume: Enable resume functionality (default: True)")
    print("  - verify_ssl: SSL certificate verification (default: True)")

    print("\nExample usage with new features:")
    print("  with DownloadAndExtract(file_list, retries=5, resume=True) as downloader:")
    print("      # Downloads happen automatically in __init__")
    print("      pass")
    print("  # Session cleanup happens automatically")

    print("=================================================")
    assert True


if __name__ == "__main__":
    pytest.main([__file__])
