from pathlib import Path
import pickle


def read_stub(stub_path: str):
    if Path(stub_path).exists():
        with Path(stub_path).open('rb') as f:
            tracks = pickle.load(f)
            return tracks
    return {}


def save_stub(data, stub_path: str):
    with Path(stub_path).open('wb') as f:
        pickle.dump(data, f)
