#!/bin/bash
#SBATCH --account desi_g
#SBATCH -C gpu&hbm80g
#SBATCH -N 1
#SBATCH --gpus 4
#SBATCH -t 1:10:00
#SBATCH -q regular
#SBATCH -J holi_mocks_recon
#SBATCH -L SCRATCH
#SBATCH -o slurm_outputs/holi_mocks_recon_%A/mock%a.log
#SBATCH --array=0-49

set -e
# Timer initialisation:
SECONDS=0

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=/global/u1/q/qinxunli/dev/desi-clustering:$PYTHONPATH

imock=$SLURM_ARRAY_TASK_ID
VERSION='holi-v3-altmtl'
STATS_DIR=$SCRATCH/holi_v3/altmtl/post_recon/spectrum/

CODE="python -m clustering_statistics.compute_stats"
echo $STATS_DIR

JOB_FLAGS="-N 1 -n 4"
COMMON_FLAGS="--stats recon_mesh2_spectrum recon_particle2_correlation --region NGC SGC --imock $imock --version $VERSION --stats_dir $STATS_DIR --expand_randoms data-dr2-v2 --weight default-FKP --combine"

LRG_FLAGS="--tracer LRG --zrange 0.4 0.6 0.6 0.8 0.8 1.1"
ELG_FLAGS="--tracer ELG_LOPnotqso --zrange 0.8 1.1 1.1 1.6"
QSO_FLAGS="--tracer QSO --zrange 0.8 2.1"

srun $JOB_FLAGS $CODE $LRG_FLAGS $COMMON_FLAGS
srun $JOB_FLAGS $CODE $ELG_FLAGS $COMMON_FLAGS
srun $JOB_FLAGS $CODE $QSO_FLAGS $COMMON_FLAGS

echo " "
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
