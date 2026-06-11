# Tools for PNG Key Project:

**Based on the following works:**
* Chaussidon+24: https://arxiv.org/abs/2411.17623
* Rosado-Marin+26: https://arxiv.org/abs/2604.05213

* Supporting papers: 
    * xxx
    * xxx
    * xxx

## TODO: 
* Add description of the tools in this repo and how it works.

----
## Measurements:

### Data:
* Blinded Data: `/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/local_png/base/desi-data/loa-v1/v2/fNL/blinded/`


### Mocks:

* Holi Mocks: `/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/local_png/base/holi-v3-altmtl`
    * Reference: xxx.
    * Script: `https://github.com/cosmodesi/desi-clustering/blob/main/clustering_statistics/job_scripts/desipipe_holi_mocks.py`. 
    * There is a block (lines 149-156) you can uncomment that has the 'local_png' options (You can change `max_mocks_per_batch` to ~50 if only Pk measurements). 
    * To run with reshuffle, you just set `onthefly='reshuffle'`. 
    * Note: I computed the ELGnotqso (and cross-correlation) with wsys-imlin with `job_scripts/desipipe_holi_mocks_edmond.py` 
        * Saved in `$PSCRATCH/cai-dr2-benchmarks/`
        * Copy those on CFS:
```bash
ssh -i $HOME/.ssh/desica desica@perlmutter.nersc.gov
rsync -av /pscratch/sd/e/edmondc/cai-dr2-benchmarks/local_png/base/holi-v3-altmtl/ /global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/local_png/base/holi-v3-altmtl
```

* PNG-Mocks: `/global/cfs/cdirs/desicollab/users/adrigut/PNGxHOD/dev_mocks/catalogs/DA2/v2.0/` (cutsky complete DR3 mocks)
   * Reference: Adame+26 (in CWR)
   * Documentation: `https://desi.lbl.gov/trac/wiki/CosmoSimsWG/DESI-PNG`
   * Script: `https://github.com/cosmodesi/desi-clustering/blob/edmond-dev/clustering_statistics/job_scripts/desipipe_pngunit-xl_mocks.py`


----
## Fitting local PNG:



----
## Data Management: 

I (Edmond) prefers to work locally for the inference as we are on an easy fit situation. Note that the following give you also where the data are on NERSC!

### Download the measurements on my machine: 

To collect files on my local computer (`/Users/edmond/Work/data/desi-clustering/`): 

 * Data:
 ```bash
cd /Users/edmond/Work/data/desi-clustering/dr2/summary_statistics/local_png/base/desi-data/loa-v1/v2/fNL/blinded
# From CFS:
rsync -av edmondc@perlmutter-p1.nersc.gov:/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/local_png/base/desi-data/loa-v1/v2/fNL/blinded/ .  
# From my local measurements (ie on my scratch): 
rsync -av edmondc@perlmutter-p1.nersc.gov:/pscratch/sd/e/edmondc/desi-clustering/dr2/summary_statistics/local_png/base/desi-data/loa-v1/v2/fNL/blinded/ . 
```

 * Holi Mocks:
 ```bash
cd /Users/edmond/Work/data/desi-clustering/dr2/summary_statistics/local_png/base/holi-v3-altmtl
rsync -av edmondc@perlmutter-p1.nersc.gov:/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/local_png/base/holi-v3-altmtl/ . 
rsync -av edmondc@perlmutter-p1.nersc.gov:/pscratch/sd/e/edmondc/desi-clustering/dr2/summary_statistics/local_png/base/holi-v3-altmtl/ .
```

* Glam Mocks:
```bash
cd /Users/edmond/Work/data/desi-clustering/dr2/summary_statistics/local_png/base/glam-uchuu-v2-altmtl
rsync -av edmondc@perlmutter-p1.nersc.gov:/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/local_png/base/glam-uchuu-v2-altmtl/ .
```

* PNG-Mocks: 
```bash
cd /Users/edmond/Work/data/desi-clustering/dr2/summary_statistics/local_png/base/pngunit-xl
rsync -av edmondc@perlmutter-p1.nersc.gov:/pscratch/sd/n/nsailer/measurements/local_png/base/pngunit-xl .

cd /Users/edmond/Work/data/desi-clustering/dr2/profiles/local_png/base/pngunit-xl
rsync -av edmondc@perlmutter-p1.nersc.gov:/pscratch/sd/n/nsailer/profiles/local_png/base/pngunit-xl .
```

### Upload data on NERSC:

This is a bit trickier because as a simple user I don't have the "write authorization" on `/global/cfs/cdirs/desi/science/cai/`. The idea is to upload everything in my Scratch (that is a mirror of the CFS): `/pscratch/sd/e/edmondc/desi-clustering/` and then copy them to the CFS directory once connected as a `desica` user (follow this: https://desi.lbl.gov/trac/wiki/CrossAnalysisInfrastructureWG/NERSC).

First upload from my machine to SCRATCH:
```bash
rsync -av /Users/edmond/Work/data/desi-clustering/dr2/summary_statistics/local_png/base/desi-data/ edmondc@perlmutter-p1.nersc.gov:/pscratch/sd/e/edmondc/desi-clustering/dr2/summary_statistics/local_png/base/desi-data
rsync -av /Users/edmond/Work/data/desi-clustering/dr2/summary_statistics/local_png/base/holi-v3-altmtl/ edmondc@perlmutter-p1.nersc.gov:/pscratch/sd/e/edmondc/desi-clustering/dr2/summary_statistics/local_png/base/holi-v3-altmtl
rsync -av /Users/edmond/Work/data/desi-clustering/dr2/summary_statistics/local_png/base/pngunit-xl/ edmondc@perlmutter-p1.nersc.gov:/pscratch/sd/e/edmondc/desi-clustering/dr2/summary_statistics/local_png/base/pngunit-xl
rsync -av /Users/edmond/Work/data/desi-clustering/dr2/profiles/ edmondc@perlmutter-p1.nersc.gov:/pscratch/sd/e/edmondc/desi-clustering/dr2/profiles
rsync -av /Users/edmond/Work/data/desi-clustering/dr2/chains/ edmondc@perlmutter-p1.nersc.gov:/pscratch/sd/e/edmondc/desi-clustering/dr2/chains
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
rsync -av /pscratch/sd/e/edmondc/desi-clustering/dr2/summary_statistics/local_png/base/pngunit-xl/ /global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/local_png/base/pngunit-xl

rsync -av /pscratch/sd/e/edmondc/desi-clustering/dr2/profiles/local_png/ /global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/profiles/local_png/
rsync -av /pscratch/sd/e/edmondc/desi-clustering/dr2/chains/local_png/ /global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/chains/local_png/
```