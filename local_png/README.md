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

Mocks are: `/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/local_png/base/holi-v3-altmtl/`

But if you want to make other measurements, the script I am using is this one here `https://github.com/cosmodesi/desi-clustering/blob/main/clustering_statistics/job_scripts/desipipe_holi_mocks.py`. There is a block (lines 149-156) you can uncomment that has the 'local_png' options. To run with reshuffle, you just set `onthefly='reshuffle'`. This script can be run as an interactive job or as a Slurm job, by changing the mode option.
You also might want to change the max_mocks_per_batch to ~50 if you only measure the Pks, that way you can process 50 mocks per job.

----
## Fitting local PNG: