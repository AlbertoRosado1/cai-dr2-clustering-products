#!/bin/bash
#SBATCH --account=desi_g
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm80g

#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=02:00:00
#SBATCH --license=SCRATCH

#SBATCH --job-name=fm_window
#SBATCH --output="log/%x-%j.out"

# Timer initialisation:
SECONDS=0


source /global/homes/e/edmondc/.bash_profile
export HDF5_USE_FILE_LOCKING=TRUE

#srun -n 4 python desipipe_data_png.py --interactive --blinded

#srun -n 4 python desipipe_data_png.py --interactive --blinded --geo --ellsout 0 2 --tracer LRGxELGnotqso 
#srun -n 4 python desipipe_data_png.py --interactive --blinded --ric --ellsout 0 2 --tracer QSO 
#srun -n 4 python desipipe_data_png.py --interactive --blinded --ric --amr --ellsout 0 2 --tracer LRGxELGnotqso 

#srun -n 4 python desipipe_data_png.py --interactive --blinded --ric --amr --ellsout 0 2 --tracer LRG --region NGCnoN SGCnoDES

#srun -n 4 python desipipe_data_png.py --interactive --blinded --geo --ellsout 0 2 --tracer QSO --region NGC SGC 
#srun -n 4 python desipipe_data_png.py --interactive --blinded --ric --amr --ellsout 0 2 --tracer QSO --region NGC SGC 

# srun -n 4 python desipipe_data_png.py --interactive --blinded --tracer LRG --region NGCnoN SGCnoDES
# srun -n 4 python desipipe_data_png.py --interactive --blinded --geo --ellsout 0 2 --tracer LRG --region NGCnoN SGCnoDES
# srun -n 4 python desipipe_data_png.py --interactive --blinded --ric --amr --ellsout 0 2 --tracer LRG --region NGCnoN SGCnoDES

# Not optimal -> the geometrical part is the same whatever the weight used .. -> OK NEED TO BE UPDATED IN THE NEXT !
#srun -n 4 python desipipe_data_png.py --interactive --blinded --geo --ellsout 0 2 --tracer LRG_zcmb --region NGC SGC 
#srun -n 4 python desipipe_data_png.py --interactive --blinded --ric --amr --ellsout 0 2 --tracer LRG_zcmb --region NGC SGC &

srun -n 4 python desipipe_data_png.py --interactive --blinded --geo --ellsout 0 2 --tracer LRG_zcmb --region NGCnoN SGCnoDES
srun -n 4 python desipipe_data_png.py --interactive --blinded --ric --amr --ellsout 0 2 --tracer LRG_zcmb --region NGCnoN SGCnoDES

# &
wait


if (( $SECONDS > 3600 )); then
    let "hours=SECONDS/3600"
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $hours hour(s), $minutes minute(s) and $seconds second(s)"
elif (( $SECONDS > 60 )); then
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $minutes minute(s) and $seconds second(s)"
else
    echo "Completed in $SECONDS seconds"
fi
echo
