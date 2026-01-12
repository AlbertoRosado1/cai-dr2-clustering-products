#!/bin/bash
# salloc -N 1 -C gpu -t 00:10:00 --gpus 4 --qos interactive --account desi_g
# salloc -N 1 -C "gpu&hbm80g" -t 00:10:00 --gpus 4 --qos interactive --account desi_g
# source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test
# bash run_test.sh

set -e
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
# DA2/LSS/loa-v1/LSScats/v2

CAT_DIR=/dvs_ro/cfs/cdirs/desi/survey/catalogs//Y1/LSS/iron/LSScats/v1.5/
MEAS_DIR=$SCRATCH/cai-dr2-benchmarks/data_v1.5/

JOB_FLAGS="-N 1 -n 4"
COMMON_FLAGS="--stat mesh2_spectrum --region NGC SGC --cat_dir $CAT_DIR --meas_dir $MEAS_DIR --combine"

ELG_FLAGS="--tracer ELG_LOPnotqso --zrange 0.8 1.1 1.1 1.6 --weight_type default_FKP"
LRG_FLAGS="--tracer LRG --zrange 0.4 0.6 0.6 0.8 0.8 1.1 --weight_type default_FKP"
QSO_FLAGS="--tracer QSO --zrange 0.8 2.1 --weight_type default_FKP"

srun $JOB_FLAGS python $CODE $ELG_FLAGS $COMMON_FLAGS
srun $JOB_FLAGS python $CODE $LRG_FLAGS $COMMON_FLAGS
srun $JOB_FLAGS python $CODE $QSO_FLAGS $COMMON_FLAGS