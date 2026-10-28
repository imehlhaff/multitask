"""
Source: https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech
"""
from pathlib import Path

from huggingface_hub import snapshot_download

OUT_DIR = Path(__file__).parent / "raw"

if __name__ == "__main__":
    snapshot_download(
        "ucberkeley-dlab/measuring-hate-speech",
        repo_type="dataset",
        local_dir=OUT_DIR,
    )
