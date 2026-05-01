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
## Data Management: 

I (Edmond) prefers to work locally for the inference as we are on an easy fit situation.

### Download the measurements on my machine: 

To collect files on my local computer (`/Users/edmond/Work/data/desi-clustering/`): 

 * Data:
    * `cd /Users/edmond/Work/data/desi-clustering/dr2/summary_statistics/local_png/base/desi-data/loa-v1/v2/fNL/blinded`
    * From CFS:
        * `rsync -av edmondc@perlmutter-p1.nersc.gov:/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/local_png/base/desi-data/loa-v1/v2/fNL/blinded/ . ` 
    * From my local measurements (ie on my scratch): 
        * `rsync -av edmondc@perlmutter-p1.nersc.gov:/pscratch/sd/e/edmondc/desi-clustering/dr2/summary_statistics/local_png/base/desi-data/loa-v1/v2/fNL/blinded/ . ` 
        
 * Holi Mocks:
    * `cd /Users/edmond/Work/data/desi-clustering/dr2/summary_statistics/local_png/base/holi-v3-altmtl`
    * `rsync -av edmondc@perlmutter-p1.nersc.gov:/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/local_png/base/holi-v3-altmtl/ . `

### Upload data on NERSC:

This is a bit trickier because as a simple user I don't have the "write authorization" on `/global/cfs/cdirs/desi/science/cai/`. The idea is to upload everything in my Scratch (that is a miror of the CFS): `/pscratch/sd/e/edmondc/desi-clustering/` and then copy them to the CFS directory once connected as a `desica` user (follow this: https://desi.lbl.gov/trac/wiki/CrossAnalysisInfrastructureWG/NERSC).

First upload from my machine to SCRATCH:
```bash
rsync -av /Users/edmond/Work/data/desi-clustering/dr2/summary_statistics/local_png/base/desi-data/ edmondc@perlmutter-p1.nersc.gov:/pscratch/sd/e/edmondc/desi-clustering/dr2/summary_statistics/local_png/base/desi-data
rsync -av /Users/edmond/Work/data/desi-clustering/dr2/profiles/ edmondc@perlmutter-p1.nersc.gov:/pscratch/sd/e/edmondc/desi-clustering/dr2/profiles
```

### Transfert data from $SCRATCH to desica CFS: 

To connect as desica: 
```bash
sshproxy -u edmondc -c desica
ssh -i $HOME/.ssh/desica desica@perlmutter.nersc.gov
```

Then rsync the Scratch to CFS:
```bash
rsync -av /pscratch/sd/e/edmondc/desi-clustering/dr2/summary_statistics/local_png/base/desi-data/ /global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/local_png/base/desi-data

rsync -av /pscratch/sd/e/edmondc/desi-clustering/dr2/profiles/local_png/ /global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/profiles/local_png/

rsync -av /pscratch/sd/e/edmondc/desi-clustering/dr2/chains/local_png/ /global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/chains/local_png/
```