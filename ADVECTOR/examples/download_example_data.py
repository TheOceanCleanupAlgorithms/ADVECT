"""Downloads all the data necessary to run ECCO_advect_2D.py and ECCO_advect_3D.py"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from data_downloaders.download_and_process_currents import download_and_process_currents
from data_downloaders.download_10m_wind import download_and_interpolate_ncep_ncar_wind
from data_downloaders.get_ECCO_credentials import get_ECCO_credentials
from data_downloaders.download_and_process_density import download_and_process_density

def main():
    data_root = Path(input("Input directory to download data into: "))
    data_root.mkdir(exist_ok=True)
    print("")
    print("Downloading ncep-ncar reanalysis ii 10m wind...")
    download_and_interpolate_ncep_ncar_wind(out_dir=data_root)

    print("")
    print("Downloading Seawater Density from JPL's ECCOv4r4...")
    density_path = data_root / "RHO_2015.nc"
    if density_path.exists():
        print(f"{density_path} already exists.  Skipping...")
    else:
        user, password = get_ECCO_credentials()
        download_and_process_density(
            out_path=density_path, user=user, password=password
        )

    print("")
    print("Downloading 3D currents from JPL's ECCOv4r4...")
    download_and_process_currents(out_dir=data_root)

    print("")
    print("All done!")


if __name__ == "__main__":
    main()
