#!/usr/bin/env python
# Import required modules
import gzip
import os
import requests
import shutil
import tarfile
import zipfile
from ftplib import FTP
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Tuple, Union
from urllib.parse import urlparse
from .LoggingUtils import log


class DownloadAndExtract:
    ARCHIVE_EXTENSIONS = {
        ".gz": "gzip",
        ".zip": "zip",
        ".tar": "tar",
        ".tar.gz": "tar_gz",
        ".tgz": "tar_gz",
        ".tar.bz2": "tar_bz2",
        ".tbz2": "tar_bz2",
        ".tar.xz": "tar_xz",
        ".txz": "tar_xz",
    }

    def __init__(
        self,
        files_to_download: List[Union[Tuple[str, str], Tuple[str, str, str]]],
        chunk_size: int = 8192,
        timeout: int = 30,
        retries: int = 3,
        resume: bool = True,
        verify_ssl: bool = True,
    ) -> None:
        self.files_to_download: List[Union[Tuple[str, str], Tuple[str, str, str]]] = (
            files_to_download
        )
        self.chunk_size: int = chunk_size
        self.timeout: int = timeout
        self.retries: int = retries
        self.resume: bool = resume
        self.verify_ssl: bool = verify_ssl
        self.session: requests.Session = self._create_session()
        self.process_files()

    def _create_session(self) -> requests.Session:
        """Create a requests session with custom headers."""
        session = requests.Session()
        session.headers.update(
            {"User-Agent": "DownloadAndExtract/1.0 (Python requests)"}
        )
        return session

    def process_files(self) -> None:
        """Download and extract all files in the list."""
        successful_downloads = 0
        failed_downloads = 0

        for file_info in self.files_to_download:
            try:
                if len(file_info) == 3:
                    url, filename, desc = file_info
                elif len(file_info) == 2:
                    url, filename = file_info
                    desc = f"Downloading {Path(filename).name}"
                else:
                    log.error(
                        f"Invalid file info format: {file_info}. Expected 2 or 3 elements."
                    )
                    failed_downloads += 1
                    continue

                if not url or not filename:
                    log.error(f"URL and filename cannot be empty: {file_info}")
                    failed_downloads += 1
                    continue

                success = self.download_file(url, filename, desc)
                if success:
                    self.extract_file(filename)
                    successful_downloads += 1
                    log.info(f"Successfully processed: {filename}")
                else:
                    failed_downloads += 1

            except Exception as e:
                log.error(f"Error processing file info {file_info}: {e}")
                failed_downloads += 1
                continue

        total_files = len(self.files_to_download)
        log.info(
            f"Download summary: {successful_downloads}/{total_files} successful, {failed_downloads} failed"
        )

    def download_file(
        self, url: str, filename: str, desc: str = "Downloading file"
    ) -> bool:
        """Download a file from a URL to a specified filename."""
        parsed_url = urlparse(url)
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if parsed_url.scheme in ["http", "https"]:
            return self._download_http(url, filename, desc)
        elif parsed_url.scheme == "ftp":
            return self._download_ftp(url, filename, desc)
        else:
            log.error(f"Unsupported URL scheme: {parsed_url.scheme}")
            return False

    def _download_http(self, url: str, filename: str, desc: str) -> bool:
        """Download a file over HTTP/HTTPS with support for resuming."""
        filepath = Path(filename)
        resume_header = {}
        initial_pos = 0

        if self.resume and filepath.exists():
            initial_pos = filepath.stat().st_size
            resume_header = {"Range": f"bytes={initial_pos}-"}
            log.info(f"Resuming download from byte {initial_pos}")

        for attempt in range(self.retries):
            try:
                response = self.session.get(
                    url,
                    stream=True,
                    timeout=self.timeout,
                    headers=resume_header,
                    verify=self.verify_ssl,
                )

                if response.status_code == 416:
                    log.warning("File already fully downloaded")
                    return True

                if response.status_code not in [200, 206]:
                    response.raise_for_status()

                content_length = response.headers.get("content-length")
                total_size = int(content_length) if content_length else 0

                if response.status_code == 206:
                    total_size += initial_pos

                mode = "ab" if self.resume and initial_pos > 0 else "wb"

                with open(filename, mode) as file, tqdm(
                    desc=desc,
                    initial=initial_pos,
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            size = file.write(chunk)
                            bar.update(size)

                log.info(f"Successfully downloaded: {filename}")
                return True

            except requests.exceptions.RequestException as e:
                log.warning(
                    f"Download attempt {attempt + 1}/{self.retries} failed: {e}"
                )
                if attempt == self.retries - 1:
                    log.error(
                        f"Failed to download {url} after {self.retries} attempts: {e}"
                    )
                    return False

        return False

    def _download_ftp(self, url: str, filename: str, desc: str) -> bool:
        """Download a file over FTP."""
        parsed_url = urlparse(url)

        for attempt in range(self.retries):
            try:
                with FTP(parsed_url.hostname, timeout=self.timeout) as ftp:
                    ftp.login(
                        parsed_url.username or "anonymous",
                        parsed_url.password or "anonymous@",
                    )

                    try:
                        total_size = ftp.size(parsed_url.path)
                    except Exception:
                        total_size = 0

                    with open(filename, "wb") as file, tqdm(
                        desc=desc,
                        total=total_size,
                        unit="iB",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:

                        def callback(data: bytes) -> None:
                            file.write(data)
                            bar.update(len(data))

                        ftp.retrbinary(f"RETR {parsed_url.path}", callback)

                log.info(f"Successfully downloaded via FTP: {filename}")
                return True

            except Exception as e:
                log.warning(
                    f"FTP download attempt {attempt + 1}/{self.retries} failed: {e}"
                )
                if attempt == self.retries - 1:
                    log.error(
                        f"Failed to download via FTP {url} after {self.retries} attempts: {e}"
                    )
                    return False

        return False

    def extract_file(self, filename: str) -> Optional[str]:
        """Extract an archive file if it is in a recognized format."""
        filepath = Path(filename)
        archive_type = self._get_archive_type(filename)
        if not archive_type:
            log.debug(f"File {filename} is not a recognized archive format")
            return None

        try:
            log.info(f"Extracting {archive_type} archive: {filename}")

            if archive_type == "gzip":
                return self._extract_gzip(filepath)
            elif archive_type == "zip":
                return self._extract_zip(filepath)
            elif archive_type.startswith("tar"):
                return self._extract_tar(filepath, archive_type)
            else:
                raise Exception(f"Unsupported archive type: {archive_type}")

        except Exception as e:
            log.error(f"Failed to extract {filename}: {e}")
            return None

    def _get_archive_type(self, filename: str) -> Optional[str]:
        """Determine the archive type based on file extension."""
        filename_lower = filename.lower()
        for ext, archive_type in sorted(
            self.ARCHIVE_EXTENSIONS.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if filename_lower.endswith(ext):
                return archive_type
        return None

    def _extract_gzip(self, filepath: Path) -> str:
        """Extract a gzip (.gz) file."""
        if filepath.suffix.lower() != ".gz":
            raise Exception(f"Expected .gz file, got: {filepath}")

        extracted_path = filepath.with_suffix("")

        with gzip.open(filepath, "rb") as f_in:
            with open(extracted_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        filepath.unlink()
        log.info(f"Extracted gzip to: {extracted_path}")
        return str(extracted_path)

    def _extract_zip(self, filepath: Path) -> str:
        """Extract a zip (.zip) file."""
        extract_dir = filepath.parent

        with zipfile.ZipFile(filepath, "r") as zip_ref:
            total_size = sum(file.file_size for file in zip_ref.infolist())
            if total_size > 10 * 1024 * 1024 * 1024:
                raise Exception("Archive too large, potential zip bomb")

            zip_ref.extractall(extract_dir)
            extracted_files = zip_ref.namelist()

        filepath.unlink()

        if len(extracted_files) == 1:
            old_path = extract_dir / extracted_files[0]
            new_path = filepath.with_suffix("")
            old_path.rename(new_path)
            log.info(f"Extracted zip to: {new_path}")
            return str(new_path)
        else:
            log.info(
                f"Extracted zip with {len(extracted_files)} files to: {extract_dir}"
            )
            return str(extract_dir)

    def _extract_tar(self, filepath: Path, archive_type: str) -> str:
        """Extract a tar (.tar, .tar.gz, .tar.bz2, .tar.xz) file."""
        extract_dir = filepath.parent

        if archive_type == "tar":
            mode = "r"
        elif archive_type == "tar_gz":
            mode = "r:gz"
        elif archive_type == "tar_bz2":
            mode = "r:bz2"
        elif archive_type == "tar_xz":
            mode = "r:xz"
        else:
            raise Exception(f"Unknown tar archive type: {archive_type}")

        with tarfile.open(filepath, mode) as tar_ref:

            def is_safe_path(path: str, base_path: str) -> bool:
                return os.path.commonpath([path, str(base_path)]) == str(base_path)

            members = tar_ref.getmembers()
            for member in members:
                member_path = os.path.join(extract_dir, member.name)
                if not is_safe_path(member_path, extract_dir):
                    raise Exception(f"Unsafe path in archive: {member.name}")

            tar_ref.extractall(extract_dir)
            extracted_files = [member.name for member in members]

        filepath.unlink()
        log.info(
            f"Extracted tar archive with {len(extracted_files)} items to: {extract_dir}"
        )
        return str(extract_dir)
