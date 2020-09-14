from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, plotTrajectoriesFile
from cartopy import crs
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

# construct fieldset
fname = 'cmems/*.nc'
filenames = {'U': fname, 'V': fname}
variables = {'U': 'uo', 'V': 'vo'}
dimensions = {'lat': 'latitude', 'lon': 'longitude', 'depth': 'depth', 'time': 'time'}
fset = FieldSet.from_netcdf(filenames, variables, dimensions)
fset.U.show(projection=crs.PlateCarree(central_longitude=180))
plt.pause(0.1)

# construct particleset
pset = ParticleSet.from_list(fieldset=fset, pclass=JITParticle,
                             lon=np.random.triangular(fset.U.lon.min(), fset.U.lon.mean(), fset.U.lon.max(), 100),
                             lat=np.random.triangular(fset.U.lat.min(), fset.U.lat.mean(), fset.U.lat.max(), 100))
pset.show(field=fset.U, projection=crs.PlateCarree(central_longitude=180))
plt.pause(0.1)

# advect
output_file = pset.ParticleFile(name='NP_particles.nc', outputdt=timedelta(hours=6))
pset.execute(AdvectionRK4,
             runtime=timedelta(days=30),
             dt=timedelta(hours=6),
             output_file=output_file)
output_file.export()

plot = plotTrajectoriesFile('NP_particles.nc')
