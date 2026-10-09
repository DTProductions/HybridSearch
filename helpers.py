import urllib.request
import tarfile
import os

def download_file(url: str, save_path: str) -> None:
    '''Downloads a file from the web and stores it locally.

    Args:
    url: The file's URL.
    save_path: The file's local save location.
    '''
    if not os.path.exists(save_path):
        print(f"Downloading file from {url}...")
        urllib.request.urlretrieve(url, save_path)

def extract_tar(file: str, extract_dir: str) -> None:
    '''Extracts a tar.gz file
    Args:
        file: the file to extract.
        extract_dir: the directory where the extracted files will be stored.
    '''
    with tarfile.open(file, "r:gz") as tar:
        tar.extractall(extract_dir)

    print(f"Extracted to: {extract_dir}")
