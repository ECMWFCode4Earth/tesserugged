
#!/usr/bin/env bash

##################################################################
# THIS NEEDS TO BE EXECUTED MANUALLY ONCE PER MACHINE TO SET
# SYMLINKS IN /dat CORRECTLY
##################################################################


if [ $HOSTNAME = "matt1.novalocal" ]
then
    echo "You obviously work on the weather cloud. Links are set to /data"
    ln -s /data/TRAINING dat/TRAINING
    ln -s /data/TESTING dat/TESTING
else
    echo "You obviously work on a local machine. Links are set to the SSEA share."
    ln -s /ssea/SSEA/C4E/DATA/TRAINING dat/TRAINING
    ln -s /ssea/SSEA/C4E/DATA/TESTING dat/TESTING
fi