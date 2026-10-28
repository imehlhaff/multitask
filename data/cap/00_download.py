"""
Source: https://www.comparativeagendas.net/datasets_codebooks
"""
from pathlib import Path

import requests

BASE_URL = "https://minio.la.utexas.edu/compagendas/datasetfiles/"
FILES = [
    "uk_manifestos_v1__1.csv",
    "irl_manifestos_data_1981_2020_utf8.csv",
    "Dem_partyplatform_20.2.csv",
    "US-Party-Rep_Platform_19.3_0.csv",
]
OUT_DIR = Path(__file__).parent / "raw"

if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    for name in FILES:
        response = requests.get(BASE_URL + name, timeout=300)
        response.raise_for_status()
        (OUT_DIR / name).write_bytes(response.content)
