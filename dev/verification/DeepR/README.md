for comparability we use the same validation code as the `DeepR` project.

To run the validation, clone https://github.com/ECMWFCode4Earth/DeepR,
create a conda environment using the environment.yml contained in the repo,
or a reduced version only holding the packages needed for the validation scripts.

The `tesserugged` output needs to be wrangled to be used as an input to the 
`DeepR` code using the script `wrangle.sh` within this folder.

Place the config file `configuration_validation_netcdf.yml` into the `resources` 
directory in the `DeepR` project directory. 

Move to `./deepr/validation/netcdf` and start the validation with 
`python validation.py`. 

Be aware that you might need to change paths within `wrangle.sh` and 
`configuration_validation_netcdf.yml`, depending on your setup.