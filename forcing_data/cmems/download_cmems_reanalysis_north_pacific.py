"""script to download a prescribed range of high-res 3d cmems_NP current data for the north pacific (0-70 lat, 100-280 lon)
A nice way to use this is to run `python download_cmems_reanalysis_north_pacific.py > cmems_NP/log.txt` in a terminal.
  for some reason, while most all the output goes to cmems_NP/log.txt (which can be watched live with tail -f cmems_NP/log.txt),
  the tqdm progress bar stays in stdout, separating the progress bar from all the logging.
"""

import pandas as pd
from datetime import timedelta as td
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

dates = pd.date_range("2000-01-01", "2000-1-31", freq="1D")
# one day at a time is simplest; 3 days is bigger than max allowed download size anyways
depth_min = 0
depth_max = 500
lon_min = 120
lon_max = 250
lat_min = 15
lat_max = 65
for date in tqdm(dates):
    filename = f"uv_{date.strftime('%Y-%m-%d')}.nc"
    print(f"Downloading {filename}")

    cmd = f"""python -m motuclient --motu http://my.cmems-du.eu/motu-web/Motu \
    --service-id GLOBAL_REANALYSIS_PHY_001_030-TDS --product-id global-reanalysis-phy-001-030-daily \
    --longitude-min {lon_min} --longitude-max {lon_max} --latitude-min {lat_min} --latitude-max {lat_max} \
    --date-min "{date}" --date-max "{date+td(days=1)}" \
    --depth-min {depth_min} --depth-max {depth_max} \
    --variable uo --variable vo \
    --out-dir ./ --out-name cmems_NP/{filename} \
    --user {os.getenv('copernicus_uname')} --pwd '{os.getenv('copernicus_pwd')}'"""

    os.system(cmd)
