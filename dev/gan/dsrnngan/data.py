""" File for handling data loading and saving. """
import os
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

import read_config

data_paths = read_config.get_data_paths()
TRUTH_PATH = data_paths["GENERAL"]["TRUTH_PATH"]
FCST_PATH = data_paths["GENERAL"]["FORECAST_PATH"]
CONSTANTS_PATH = data_paths["GENERAL"]["CONSTANTS_PATH"]

all_fcst_fields = [
    "t2m",
]  # ["tp", "cp", "sp", "tisr", "cape", "tclw", "tcwv", "u700", "v700"]
fcst_hours = np.array([str(x).zfill(2) for x in np.arange(0, 24, 3)])


def denormalise(x):
    """
    Undo log-transform of rainfall.  Also cap at 500 (feel free to adjust according to application!)
    """
    return np.minimum(10**x - 1, 500.0)


# Not needed in our application
def get_dates(year):
    """
    Return dates where we have truth data
    """
    from glob import glob

    file_paths = os.path.join(TRUTH_PATH, str(year), "*.nc")
    files = glob(file_paths)
    dates = []
    for f in files:
        dates.append(f[:-3].split("_")[-1])
    return sorted(dates)


def load_truth_and_mask(hour, year, log_precip=False, aggregate=1):
    hour = str(hour).zfill(2)
    year_start = f"01-01-{year}"
    year_end = f"31-12-{year}"
    data_path = os.path.join(TRUTH_PATH, f"t2m_cerra_{hour}_residuals.nc")
    data = xr.open_dataset(data_path)
    assert int(hour) + aggregate < 25
    y = np.array(data[f"t2m_cerra_{hour}.nc"].sel(time=slice(year_start, year_end)))
    data.close()
    # The remapping of the NIMROD radar left a few negative numbers, so remove those
    # y[y < 0.0] = 0.0  # N/A for our purpose

    # mask: False for valid truth data, True for invalid truth data
    # (compatible with the NumPy masked array functionality)
    # if all data is valid:
    mask = np.full(y.shape, False, dtype=bool)

    if log_precip:
        return np.log10(1 + y), mask
    else:
        return y, mask


# testy, testmask = load_truth_and_mask(12)
# print(testy.shape)
# print(testmask.shape)
# quit()


def logprec(y, log_precip=True):
    if log_precip:
        return np.log10(1 + y)
    else:
        return y


def load_hires_constants(batch_size=1):
    oro_path = os.path.join(CONSTANTS_PATH, "cerra_lsm.nc")
    df = xr.load_dataset(oro_path)
    # LSM is already 0:1
    lsm_ = np.array(df["lsm"])

    # Orography.  Clip below, to remove spectral artifacts, and normalise by max
    z = df["orog"].data
    z[z < 5] = 5
    z = z / z.max()

    print("Time: 1, latitude: 160, longitude: 240")
    print(f"{z.shape = }, {lsm_.shape = }")
    df.close()
    # outshape: (batch_size, latitude, longitude, 2) (2 => 2 fields)
    return np.repeat(np.stack([z, lsm_], -1), batch_size, axis=0)


# z_lsm = load_hires_constants()
# print(z_lsm.shape)
# z = np.squeeze(z_lsm[:, :, :, 0])
# lsm = np.squeeze(z_lsm[:, :, :, 1])

# import matplotlib.pyplot as plt

# plt.imshow(z)
# plt.title("z")
# plt.colorbar()
# plt.savefig("test0.png")
# plt.close()
# plt.imshow(lsm)
# plt.colorbar()
# plt.title("lsm")
# plt.savefig("test1.png")
# quit()


def load_fcst_truth_batch(
    batch_dates,
    fcst_fields=all_fcst_fields,
    log_precip=False,
    hour=0,
    norm=False,
    years=None,
):
    batch_x = []  # forecast
    batch_y = []  # truth
    batch_mask = []  # mask

    if type(hour) == str:
        if hour == "random":
            hours = fcst_hours[
                np.random.randint(len(fcst_hours), size=[len(batch_dates)])
            ]
        else:
            assert False, f"Not configured for {hour}"
    elif np.issubdtype(type(hour), np.integer):
        hours = len(batch_dates) * [hour]
    else:
        hours = hour

    for i, year in enumerate(batch_dates):
        h = hours[i]
        batch_x.append(
            load_fcst_stack(fcst_fields, h, log_precip=log_precip, norm=norm, year=year)
        )
        truth, mask = load_truth_and_mask(h, year, log_precip=log_precip)
        batch_y.append(truth)
        batch_mask.append(mask)

    return np.array(batch_x), np.array(batch_y), np.array(batch_mask)


def load_fcst(ifield, hour, log_precip=False, norm=False, year=None):
    # Get the time required (compensating for IFS forecast saving precip at the end of the timestep)
    # time = datetime(
    #     year=int(date[:4]), month=int(date[4:6]), day=int(date[6:8]), hour=hour
    # ) + timedelta(hours=1)

    # # Get the correct forecast starttime
    # if time.hour < 6:
    #     tmpdate = time - timedelta(days=1)
    #     loaddate = datetime(
    #         year=tmpdate.year, month=tmpdate.month, day=tmpdate.day, hour=18
    #     )
    #     loadtime = "12"
    # elif 6 <= time.hour < 18:
    #     tmpdate = time
    #     loaddate = datetime(
    #         year=tmpdate.year, month=tmpdate.month, day=tmpdate.day, hour=6
    #     )
    #     loadtime = "00"
    # elif 18 <= time.hour < 24:
    #     tmpdate = time
    #     loaddate = datetime(
    #         year=tmpdate.year, month=tmpdate.month, day=tmpdate.day, hour=18
    #     )
    #     loadtime = "12"
    # else:
    #     assert False, "Not acceptable time"
    # dt = time - loaddate
    # time_index = int(dt.total_seconds() // 3600)
    # assert time_index >= 1, "Cannot use first hour of retrival"
    # loaddata_str = loaddate.strftime("%Y%m%d") + "_" + loadtime
    # field = ifield
    # if field in ["u700", "v700"]:
    #     fleheader = "winds"
    #     field = field[:1]
    # elif field in ["cdir", "tcrw"]:
    #     fleheader = "missing"
    # else:
    #     fleheader = "sfc"

    # ds_path = os.path.join(FCST_PATH, f"{fleheader}_{loaddata_str}.nc")
    hour = str(hour).zfill(2)
    ds_path = os.path.join(FCST_PATH, f"t2m_era5_{hour}_residuals.nc")
    ds = xr.open_dataset(ds_path)
    # data = ds[field]
    data = ds[f"t2m_era5_{hour}.nc"]
    # field = ifield
    # if field in ["tp", "cp", "cdir", "tisr"]:  # for accumulated fields
    #     data = data[time_index, :, :] - data[time_index - 1, :, :]
    # else:
    #     data = data[time_index, :, :]
    if not year:
        y = data.values
        data.close()
        ds.close()
        # output shape: (time: 12418, latitude: 32, longitude: 48)
    else:
        if isinstance(year, list):
            year_start = year[0]
            year_end = year[-1]
        else:
            year_start = year
            year_end = year
        print(f"sel data from {year_start = } to {year_end = }")
        datsel = data.sel(time=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
        y = datsel.values
        data.close()
        ds.close()
        datsel.close()
        # output shape: (time: 365/366, latitude: 32, longitude: 48)
    # only precipitation relevant
    if ifield in ["tp", "cp", "pr", "prl", "prc"]:
        # print('pass')
        y[y < 0] = 0.0
        y = 1000 * y
    if log_precip and ifield in ["tp", "cp", "pr", "prc", "prl"]:
        # precip is measured in metres, so multiply up
        return np.log10(1 + y)  # *1000)
    elif norm:
        return (y - fcst_norm[ifield][0]) / fcst_norm[ifield][1]
    else:
        return y


# test = load_fcst("t2m", 12, norm=False)
# test1 = test[50, :, :]
# test2 = test[11111, :, :]

# import matplotlib.pyplot as plt

# plt.imshow(test1)
# plt.title("t2m t=50")
# plt.colorbar()
# plt.savefig("t2m_field_t50.png")
# plt.close()
# plt.imshow(test2)
# plt.colorbar()
# plt.title("t2m t=11111")
# plt.savefig("t2m_field_t11111.png")
# quit()


def load_fcst_stack(fields, hour, log_precip=False, norm=False, year=None):
    field_arrays = []
    for f in fields:
        field_arrays.append(
            load_fcst(f, hour, log_precip=log_precip, norm=norm, year=year)
        )
    return np.stack(field_arrays, -1)


def get_fcst_stats(field, year=2018):
    # import datetime

    # # create date objects
    # begin_year = datetime.date(year, 1, 1)
    # end_year = datetime.date(year, 12, 31)
    # one_day = datetime.timedelta(days=1)
    # next_day = begin_year

    # mi = 0
    # mx = 0
    # mn = 0
    # sd = 0
    # nsamples = 0
    # # for day in range(0, 366):  # includes potential leap year
    # #     if next_day > end_year:
    # #         break
    # for hour in fcst_hours:
    #     print(hour)
    #     dta = load_fcst(field, hour)

    #     print(dta.shape)
    #     try:
    #         quit()
    #         mi = min(mi, dta.min())
    #         mx = max(mx, dta.max())
    #         mn += dta.mean()
    #         sd += dta.std() ** 2
    #         nsamples += 1
    #     except:  # noqa
    #         quit()
    #         print(f"Problem loading {next_day.strftime('%Y%m%d')}, {hour}")
    #     next_day += one_day
    # mn /= nsamples
    # sd = (sd / nsamples) ** 0.5
    dta = load_fcst(field, "12", year=year)
    mi = dta.min()
    mx = dta.max()
    mn = dta.mean()
    sd = dta.std()
    return mi, mx, mn, sd


# test = get_fcst_stats("t2m")
# print(test)
# quit()


def gen_fcst_norm(year=2018):
    """
    One-off function, used to generate normalisation constants, which are used to normalise the various input fields for training/inference.

    Depending on the field, we may subtract the mean and divide by the std. dev., or just divide by the max observed value.
    """

    import pickle

    stats_dic = {}
    for f in all_fcst_fields:
        stats = get_fcst_stats(f, year)
        if f == "sp":
            stats_dic[f] = [stats[2], stats[3]]
        elif f == "u700" or f == "v700":
            stats_dic[f] = [0, max(-stats[0], stats[1])]
        else:
            stats_dic[f] = [0, stats[1]]
    fcstnorm_path = os.path.join(CONSTANTS_PATH, f"FCSTNorm{year}.pkl")
    with open(fcstnorm_path, "wb") as f:
        pickle.dump(stats_dic, f, 0)
    return


### already generated in "/data/TRAINING/RESIDUALS/STATIC/"
# for y in range(1985, 2018 + 1):
#     print(f"Generating forecast norms for year={y}")
#     gen_fcst_norm(year=y)


def load_fcst_norm(year=2018):
    import pickle

    fcstnorm_path = os.path.join(CONSTANTS_PATH, f"FCSTNorm{year}.pkl")
    with open(fcstnorm_path, "rb") as f:
        return pickle.load(f)


try:
    fcst_norm = load_fcst_norm(2018)
except:  # noqa
    fcst_norm = None
