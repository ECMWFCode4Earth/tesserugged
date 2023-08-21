# Downscaling cGAN requirements:

Use conda (even better, mamba); install everything from conda-forge
- Python 3
- TensorFlow 2
    - we use v2.7; other versions should require no/minimal changes
    - GPU strongly recommended (use a GPU-enabled TensorFlow build, compatable with the installed CUDA version)
- numba[^1]
- matplotlib
- seaborn
- cartopy
- jupyter
- xarray
- pandas
- netcdf4
- h5netcdf
- scikit-learn
- cfgrib
- dask
- tqdm
- properscoring
- climlab
- iris

May also require cudatoolkit

[^1]: If numba is not available, we suggest you replace `from properscoring import crps_ensemble` to `from crps import crps_ensemble` in `evaluation.py` and `run_benchmarks.py`. This is because properscoring will fall back to an inefficient and memory-heavy CRPS implementation in this case.

# High-level data overview

You should have two main datasets:
1. Forecast data (low-resolution)
2. Truth data, for example radar (high-resolution).  This may contain missing data.

All images in each dataset should be the same size, and there should be a constant resolution scaling factor between them.  Enter this downscaling factor in `downscaling_factor.yaml`, along with a list of `steps` that multiply to the overall factor.  In the original paper, we use 10x, with steps of 5 and 2.  See `models.py` for exactly how these are used in the architecture.

In the paper, we also used a third, static, dataset:

3. "Constant" data - orography and land-sea mask, at the same resolution as the truth data.

If you do not want to use similar constant data, you will need to adjust the code and model architecture slightly.

Ideally these datasets are as 'clean' as possible.  We recommend you generate these with offline scripts that perform all regridding, cropping, interpolation, etc., so that the files can be loaded in and read directly with as little further processing as possible.  We saved these as netCDF files, which we read in using xarray.

We assume that it is possible to perform inference using full-size images, but that the images are too large to use whole during training.

For this reason, part of our dataflow involves generating training data separately (small portions of the full image), and storing these in .tfrecords files.  We think this has two advantages:
1. The training data is in an 'optimised' format; only the data needed for training is loaded in, rather than needing to open several files and extract a single timestep from each.
2. The training data can be split into different bins, and data can be drawn from these bins in a specific ratio.  We use this to artificially increase the prevalence of rainy data.

# Setting up the downscaling cGAN with your own data

Start by adjusting the paths in `data_paths.yaml` to point at your own datasets.  You can set up multiple configurations, e.g., if you are running your code on different machines that have data stored in different paths.

Copy the provided `local_config-example.yaml` to `local_config.yaml` and set the options appropriately.  The file `read_config.py` controls what these options do, and you may wish to add more options of your own.

The main file you will have to change is `data.py`.  The functions in there control how the forecast and "truth" (radar) data are loaded, along with the definition of the "invalid data" mask.  You will want to rewrite substantial parts of these functions, according to the data that you plan to use.  As written, these functions are centred around the radar data.  Essentially, for a particular date and time, the radar data for that time is loaded.  A corresponding forecast is found (the logic for this is in `load_fcst()`), and that data is loaded.  You may wish to flip this logic around to be more forecast-centric.

A particular trap:
- Beware the difference between instanteneous fields (use `field[hour]`) and accumulated fields (use `field[hour] - field[hour-1]`).  Naturally, beware of off-by-one errors, etc., too.

These functions are ultimately called by the `DataGenerator` class in `data_generator.py`.  This class represents a dataset of full-size images, which may be used for inference, or to create training data from.

At this point, you may wish to visualise the data returned by the `DataGenerator` to check that it matches what you expect!

You will want to manually run the function `gen_fcst_norm` in `data.py`, either from a Python 'notebook', or by writing a small wrapper script.  This function saves certain field statistics (mean, standard dev., max, etc.) that are used for normalising inputs to the neural networks during training.  The NN performance should not be sensitive to the exact values of these, so it will be fine to run for just 1 year (among the training data years)

Next, you will want to manually run the function `write_data` in `tfrecords_generator.py` for each year of training data.  This generates training data by subsampling the full-size images.  This function has various hard-coded constants, including the low-resolution image size (assumed square!  do edit the code if this is not true...), the size of the subsampled images, and the number of samples taken from each full image.  Make sure you adjust these to your own purpose.

The training data is separated into several bins/classes, according to what proportion of the image has rainfall.  You may wish to edit how this is performed!

# Training and evaluating a network

1. Training the model

First, set up your model parameters in the configuration (.yaml) file.
An example is provided in the main directory. We recommend copying this 
file to somewhere on your local machine before training.

Run the following to start the training:

`python main.py --config path/to/config_file.yaml`

There are a number of options you can use at this point. These will 
evaluate your model after it has finished training:

- `--evaluate` to run checkpoint evaluation (CRPS, rank calculations, RMSE, RALSD, etc.)
- `--plot_ranks` will plot rank histograms (requires `--evaluate`)
	   
If you choose to run `--evaluate`, you must also specify if you want
to do this for all model checkpoints or just a selection. Do this using 

- `--eval_full`	  (all model checkpoints)
- `--eval_short`	  (recommended; the final 1/3rd of model checkpoints)
- `--eval_blitz`	  (the final 4 model checkpoints)

Two things to note:
- These three options work well with the 100 checkpoints that we 
have been working with. If this changes, you may want to update
them accordingly.
- Calculating everything, for all model iterations, will take a long 
time. Possibly weeks. You have been warned.

As an example, to train a model and evaluate the last few model
checkpoints, you could run:

`python main.py --config path/to/config_file.yaml --evaluate --eval_blitz --plot_ranks`

2. If you've already trained your model, and you just want to run some 
evaluation, use the --no_train flag, for example:

`python main.py --config path/to/config_file.yaml --no_train --evaluate --eval_full`

3. To generate plots of the output from a trained model, use `predict.py`
This requires a path to the directory where the weights of the model are 
stored, and will read in the setup parameters directly. Use the following
arguments:

Necessary arguments:
- `--log_folder`    path to model directory
- `--model_number`    model checkpoint you want to use to perform predictions

Optional arguments:
- `--predict_year`            year of data to predict on
- `--num_samples`             number of different input images to predict on
- `--pred_ensemble_size`      number of predictions to draw from ensemble
		    N.B. if you run mode == 'det' we have hardcoded this
		    to 1 for obvious reasons.

There are also the following optional arguments:
- `--plot_rapsd`              to plot the power spectrum curves

For example:

`python predict.py --log_folder /path/to/model --model_number 0006400 
--num_samples 5 --pred_ensemble_size 3 --plot_rapsd`

4. If you want to do some evaluation on specific checkpoints only, you can
use the scripts

- `run_eval.py`, to run CRPS, rank calculations, RMSE, RALSD, etc.  Note this uses 100 ensemble members, rather than the 10 used for intermediate checkpoint evaluation via `main.py`.  This script is intended to be used with a very small number of checkpoints, e.g., a single 'best' checkpoint.
- `run_roc.py`, to run ROC and precision-recall curve calculations
- `run_fss.py`, to run fractions skill score calculations

You'll have to open them and hardcode file paths and model numbers at the
top. I also made a plot_comparisons.py at one point but I've deleted it so
now you have to use predict.py. Sorry.

If some of the radar data is invalid, and therefore masked out, the diagnostics take this into account. Pointwise metrics such as MAE and RMSE will not include pixels where the radar data doesn't exist. When metrics are calculated on spatially-pooled fields, pixels are treated as invalid whenever any input pixels are invalid. The fractions skill score is treated differently: any invalid pixels are set to zero precipitation in the truth, and the corresponding pixels are set to zero in the GAN ensemble forecasts. This is for simplicity, since we treat the FSS code as a third-party library routine.

5. If you want to change model architectures, add them to the `models.py` and `setupmodel.py` files. You can then use the `config.yaml` file to call a different model architecture. We suggest that small tweaks to the existing architecture continue to use `models.generator`/`models.discriminator`, but with a new "architecture name" `arch`, and the changes are made conditional on the architecture name. For completely different architectures, write an entirely new architecture function such as `generator2`, and use this in the `setupmodel.py` dictionaries.

6. `run_benchmarks.py` will generate scores for CRPS, RMSE, MAE and RAPSD for particular simple comparison 'benchmark' models. We supply two simple models by default:

- nearest neighbour upsampling, which takes the low-resolution input forecast and repeats each pixel several times in each direction
- zero forecast, a forecast of zero rainfall everywhere

You are, of course, welcome to add more sophisticated comparison approaches of your own. In the paper, we used a re-implementation of the ecPoint approach. However, since some manual tweaking is involved, and in order to avoid any confusion with the official ECMWF ecPoint product, we have not included that approach in the public version of this code.

We have set `run_benchmarks.py` up to use the same evaluation setup as the NN models, and with 100 ensemble members. For a proper comparison, make sure you evaluated your neural network model with 100 ensemble members.

For example:
`python run_benchmarks.py --log_folder /path/to/model --predict_year 2020 --num_images 16`

7. We support two problem types, "normal" and "autocoarsen".

- normal: separate input and output datasets, loaded from user-specified files, assumed to represent forecast data and 'truth' data.
- autocoarsen: the input dataset is automatically created by coarsening the 'truth' data. The forecast data is ignored. This is a somewhat easier version of the downscaling problem since it removes the forecast error component of the problem. See `main.py`, `data_generator.py` and `tfrecords_generator.py` for exact implementation details.

8. This is research code so please let us know if something is wrong and
also note that it definitely isn't perfect :)
