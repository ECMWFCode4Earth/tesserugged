"""contains routines for loading data and calculating+plotting spatial power spectral density"""
from pathlib import Path

import numpy as np
import xarray as xr


def load_train_dataarray(
    lead_time: str, model: str, residual: bool = True
) -> xr.DataArray:
    """load training data for specified lead_time and model"""
    if residual:
        train_res = Path("/data/TRAINING/RESIDUALS")
    else:
        raise NotImplementedError("Only residual=True is implemented currently!")
    allowed_models = ["CERRA", "ERA5"]
    allowed_lead_times = ["00", "03", "06", "09", "12", "15", "18", "21"]
    assert model in allowed_models, f"model '{model}' has to be one of {allowed_models}"
    assert (
        lead_time in allowed_lead_times
    ), f"lead time '{lead_time}' has to be one of {allowed_lead_times}"
    path_ = Path(train_res, model, f"t2m_{model.lower()}_{lead_time}_residuals.nc")
    return xr.load_dataset(path_)[f"t2m_{model.lower()}_{lead_time}.nc"]


def calc_plot_spatial_power_spectral_density(xda: xr.DataArray):
    """Calculates the power spectral density over space for the given input xr.DataArray"""
    existing_dims = xda.dims
    check_only_spatial_dims = list(x for x in existing_dims if x not in ("x", "y"))
    assert (
        not check_only_spatial_dims
    ), f"Input xr.DataArray contains invalid dimensions: {check_only_spatial_dims}"
    # raw_data = xda.values.ravel()
    # if np.isnan(raw_data).any():
    #     raw_data[np.isnan(raw_data)] = 0

    n1 = xda.shape[0]
    n2 = xda.shape[1]
    delta_y = np.abs(xda.y.diff(dim="y")[0].values)
    delta_x = np.abs(xda.x.diff(dim="x")[0].values)
    f1 = np.fft.rfftfreq(n1, d=1 / delta_y)
    f2 = np.fft.rfftfreq(n2, d=1 / delta_x)

    fft1 = np.abs(np.fft.rfft(xda, n1, axis=0))
    fft1 = fft1.real * fft1.real + fft1.imag * fft1.imag
    fft1 = fft1.sum(axis=1) / fft1.shape[1]

    fft2 = np.fft.rfft(xda, n2, axis=1)
    fft2 = fft2.real * fft2.real + fft2.imag * fft2.imag
    fft2 = fft2.sum(axis=0) / fft2.shape[0]

    import matplotlib.pyplot as plt

    plt.plot(f1[1:], fft1[1:], label="fft(dim1)")
    plt.plot(f2[1:], fft2[1:], label="fft(dim2)")
    plt.yscale("log")
    plt.legend()
    plt.ylabel("Power Spectrum")
    plt.xlabel("Spatial Frequency [cycles/degree]")
    plt.title("Spatial Power Spectrum")
    plt.savefig("test.png")
    return None


# def plot_power_spectrum():

if __name__ == "__main__":
    train_era5 = load_train_dataarray(lead_time="00", model="ERA5")
    train_cerra = load_train_dataarray(lead_time="00", model="CERRA")
    # print(train_era5.shape)
    print(train_cerra)
    # print(train_cerra)

    # calc_plot_spatial_power_spectral_density(xda=train_era5.isel(time=0))
