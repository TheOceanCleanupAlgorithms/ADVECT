import sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
from download_ECCO import download_ECCO_currents, download_ECCO_grid


if __name__ == "__main__":
    print("Directing you to the ECCO grid download...")
    download_ECCO_grid()
    print("Downloading ECCO currents...")
    download_ECCO_currents()
