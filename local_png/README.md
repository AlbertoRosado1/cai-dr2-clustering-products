# Tools for PNG Key Project:

**Based on the following works:**
* Chaussidon+24: https://arxiv.org/abs/2411.17623
* Rosado-Marin+26: https://arxiv.org/abs/2604.05213

## TODO: 
* add how the measurements on the mocks are perfomed (which script and how to do it)
* same with the data 
* Add Data path + Mocks paths
* Add description of the tools in this repo and how it works.

----
## Measurements:

### Data:
Blinded Data: `/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/local_png/base/desi-data/loa-v1/v2/fNL/blinded/`


### Mocks:
Holi Mocks: `/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/local_png/base/holi-v3-altmtl`

For additional measurements on mocks: 
 * Script: `https://github.com/cosmodesi/desi-clustering/blob/main/clustering_statistics/job_scripts/desipipe_holi_mocks.py`. 
 * There is a block (lines 149-156) you can uncomment that has the 'local_png' options. 
 * To run with reshuffle, you just set `onthefly='reshuffle'`. 
 * This script can be run as an interactive job or as a Slurm job, by changing the mode option.
 * You can change `max_mocks_per_batch` to ~50 if you only measure the Pks, that way you can process 50 mocks per job.

----
## Fitting local PNG:



----
## Download the measurements on your machine: 

To collect files on my local computer (`/Users/edmond/Work/data/desi/`): 

 * Data:
    * `cd /Users/edmond/Work/data/desi/dr2/summary_statistics/local_png/base/desi-data/loa-v1/v2/fNL/blinded`
    * For CFS official measurements:
        * `rsync -av edmondc@perlmutter-p1.nersc.gov:/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/local_png/base/desi-data/loa-v1/v2/fNL/blinded/ . ` 
    * For your local measurements (on your scratch): 
        * `rsync -av edmondc@perlmutter-p1.nersc.gov:/pscratch/sd/e/edmondc/DR2_local_png/measurements/loa-v1/v2/fNL/blinded/ . ` 
        
 * Holi Mocks:
    * `cd /Users/edmond/Work/data/desi/dr2/summary_statistics/local_png/base/holi-v3-altmtl`
    * `rsync -av edmondc@perlmutter-p1.nersc.gov:/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/local_png/base/holi-v3-altmtl/ . `