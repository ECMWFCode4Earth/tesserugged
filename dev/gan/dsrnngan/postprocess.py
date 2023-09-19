"""Load yearly predicted data and save as one file"""
import xarray as xr


dalist = []
for year in [2019, 2020, 2021]:
    dayear = xr.load_dataarray(
        f"/data/TESTING/GAN/predictions/prediction_gan_ensemble_00_{year}.nc"
    )
    dalist.append(dayear)

dafull = xr.concat(dalist, dim="time")
outpath = "/data/TESTING/GAN/predictions/prediction_gan_ensemble_00.nc"
dafull.to_netcdf(outpath)

print(dafull)
print(f"Saved data to {outpath}")
