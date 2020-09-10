import pandas as pd
from datetime import timedelta as td
import os
dates = pd.date_range('2000-01-01', '2000-12-31', freq='1D')

for date in dates:
    filename = f"uv_{date.strftime('%Y-%m-%d')}.nc"
    print(f'Downloading {filename}')

    cmd = f"""python -m motuclient --motu http://my.cmems-du.eu/motu-web/Motu \
    --service-id GLOBAL_REANALYSIS_PHY_001_030-TDS --product-id global-reanalysis-phy-001-030-daily \
    --longitude-min 100 --longitude-max 280 --latitude-min 0 --latitude-max 70 \
    --date-min "{date}" --date-max "{date+td(days=1)}" \
    --depth-min 0.493 --depth-max 5727.918 \
    --variable uo --variable vo \
    --out-dir ./ --out-name cmems/{filename} \
    --user dklink --pwd '#mcASJwDGB.B9jt'"""

    os.system(cmd)
