"""
Download the alanine dipeptide trajectories (here pairwise distances are precomputed).

https://github.com/CSML-IIT-UCL/kooplearn/blob/public/examples/alanine_dipeptide/get_dataset.py
"""
import argparse
import functools
import pathlib
import shutil
import requests
import os
from tqdm.auto import tqdm
from warnings import warn

base_url = "http://ftp.imp.fu-berlin.de/pub/cmb-data/"
files = [
    "alanine-dipeptide-3x250ns-backbone-dihedrals.npz",
    "alanine-dipeptide-3x250ns-heavy-atom-distances.npz",
    # "alanine-dipeptide-3x250ns-heavy-atom-positions.npz",
]


def download(url, filename):
    # Adapted from https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Alanine Dipeptide data downloader')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Where to store the downloaded dataset')
    args = parser.parse_args()
    _datadir = args.data_dir
    if not os.path.isdir(_datadir):
        os.makedirs(_datadir, exist_ok=True)
    else:
        warn("Warning: data folder already exists. Old data will be overwritten.")

    for file in files:
        download(base_url + file, os.path.join(_datadir, file))
