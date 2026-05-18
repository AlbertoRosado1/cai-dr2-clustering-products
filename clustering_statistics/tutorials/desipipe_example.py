"""
Script to create and spawn desipipe tasks to compute clustering measurements.
To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python desipipe_example.py             # create the list of tasks
desipipe tasks -q example_job          # check the list of tasks
desipipe spawn -q example_job --spawn  # spawn the jobs
desipipe queues -q example_job         # check the queue
```
To run on an interactive node
```bash
salloc -N 1 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
srun -n 4 python desipipe_example.py
```
"""
import os
from pathlib import Path
import functools

import numpy as np
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

from clustering_statistics import tools

setup_logging()

queue = Queue('example_job') # job name
queue.clear(kill=False) # do not kill running jobs if this script is executed.

output, error = 'slurm_outputs/example_job/slurm-%j.out', 'slurm_outputs/example_job/slurm-%j.err' # define where the slurm outputs will be saved at
kwargs = {}
environ = Environment('nersc-cosmodesi') # this uses source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
tm = TaskManager(queue=queue, environ=environ) # provide queue and environment to desipipe's task manager
tm = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='02:00:00', mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu')) # regular gpu node
tm80 = tm.clone(provider=dict(provider='nersc', time='02:00:00', mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g')) # large memory gpu node

# define possible region combinations
combine_region_sources = {
    'GCcomb': ['NGC', 'SGC'],
    'NS': ['N', 'S'],
    'GCcomb_noN': ['NGCnoN', 'SGC'],
    'GCcomb_noDES': ['NGC', 'SGCnoDES'],
}

def run_stats(tracer='LRG', project='', version='holi-v3-altmtl', onthefly=None, imocks=[150], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], weight='default-FKP', analysis='full_shape', regions=['NGC','SGC'], ibatch=None, postprocess=None, zranges=None, **kwargs):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    import sys
    import functools
    from pathlib import Path
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    try: jax.distributed.initialize()
    except RuntimeError: print('Distributed environment already initialized')
    else: print('Initializing distributed environment')
    from clustering_statistics import tools, setup_logging, compute_stats_from_options, fill_fiducial_options, postprocess_stats_from_options
    setup_logging()
    
    cache = {}
    if zranges is None:
        raise ValueError('Please provide zranges.')
    for imock in imocks:
        for region in regions:
            mesh2_spectrum = {'cut': True if 'shape' in analysis else None, 
                              'auw': True if 'altmtl' in version and onthefly is None and 'shape' in analysis else None}
            window_mesh2_spectrum = {'cut': True if 'shape' in analysis else None}
            
            options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight=weight, imock=imock), 
                           mesh2_spectrum=mesh2_spectrum, window_mesh2_spectrum=window_mesh2_spectrum,
                           window_mesh3_spectrum={'ibatch': ibatch} if isinstance(ibatch, tuple) else {'computed_batches': ibatch})
            options = fill_fiducial_options(options, analysis=analysis)
            
            for itracer in options['catalog']:
                options['catalog'][itracer]['zranges'] = zranges # override fiducial zranges 
                options['catalog'][itracer]['expand']  = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=itracer, nran=options['catalog'][itracer]['nran']), 'from_data': ['Z', 'WEIGHT_SYS', 'FRAC_TLOBS_TILES']}
                if onthefly == 'complete':
                    options['catalog'][itracer]['complete'] = {}
                elif onthefly == 'reshuffle':
                    merged_dir = tools.base_stats_dir / 'merged_catalogs' / version
                    options['catalog'][itracer]['reshuffle'] = {'merged_data_fn': tools.get_catalog_fn(kind='data', cat_dir=merged_dir, **(options['catalog'][itracer] | dict(region='ALL')))}          
            
            get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir, project=project, extra=onthefly if onthefly else '')
            compute_stats_from_options(stats, analysis=analysis, get_stats_fn=get_stats_fn, cache=cache, **options)

    # postprocess will combine results from different regions
    if postprocess:
        postprocess_options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight, imock=imocks[0]), imocks=imocks, 
                                   combine_regions={'stats': stats}, mesh2_spectrum=mesh2_spectrum, window_mesh2_spectrum=window_mesh2_spectrum)
        postprocess_stats_from_options(postprocess, analysis=analysis, get_stats_fn=get_stats_fn, **postprocess_options)

def postprocess_stats(tracer='LRG', analysis='full_shape', project='', version='holi-v3-altmtl', onthefly=None, imocks=[150], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], weight='default-FKP', postprocess=['combine_regions'], zranges=None, regions = ['GCcomb'], **kwargs):
    from clustering_statistics import postprocess_stats_from_options
    if len(imocks) == 0:
        return
    if zranges is None:
        zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
    for region in regions:
        options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight=weight, imock=imocks[0]), imocks=imocks, combine_regions={'stats': stats}, 
                    mesh2_spectrum={'cut': True, 'auw': True}, window_mesh2_spectrum={'cut': True})
        options.update(combine_regions={'stats': stats, 'regions': combine_region_sources.get(region, ['NGC', 'SGC'])})
        stats_dir_kws = dict(stats_dir=stats_dir, project=project)
        if onthefly == 'complete':
            get_stats_fn = functools.partial(tools.get_stats_fn, extra='complete', **stats_dir_kws)
        elif onthefly == 'reshuffle':
            get_stats_fn = functools.partial(tools.get_stats_fn, extra='reshuffle', **stats_dir_kws)
        else:
            get_stats_fn = functools.partial(tools.get_stats_fn, **stats_dir_kws)
        postprocess_stats_from_options(postprocess, analysis=analysis, get_stats_fn=get_stats_fn, **options)


if __name__ == '__main__':

    stats, postprocess = [], []
    # Check for  existing measurements, such that we only run what is missing. Setting this to False,
    # will compute measurments whether they exist or not (overwriting existing measurements).
    check_for_existing_measurements = True 
    postprocess_only = False # If True, no measurements are performed and only postprocessing of existing measurements is handled.

    mode = 'slurm' # to run as job
    # mode = 'interactive' # to run on interactive node
    version  = 'holi-v3-altmtl'
    imocks2run = np.arange(1)
    stats_dir  = Path(os.getenv('SCRATCH')) / 'clustering_measurements' 
    stats    = ['mesh2_spectrum', 'mesh3_spectrum', 'particle2_correlation'] # which stats to measure
    weight   = 'default-FKP'
    regions  = ['NGC','SGC']
    tracers  = ['LRG', 'ELG_LOPnotqso', 'QSO']
    postprocess = ['combine_regions']
    max_mocks_per_batch = 10 # max number of mocks to process per job

    # run with fiducial full_shape setup
    analysis = 'full_shape'
    project  = f'{analysis}/base'
    postregions = ['GCcomb']

    # run data_splits with full_shape setup 
    # analysis = 'full_shape'
    # project = f'{analysis}/data_splits'
    # regions = ['NGC', 'SGC', 'N', 'NGCnoN', 'S', 'SGCnoDES'] # galactic and imaging regions
    # regions = regions + ['ACT_DR6', 'PLANCK_PR4'] + [f'GAL0{i}' for i in [40, 60]] # lensing regions
    # postregions = ['GCcomb', 'NS', 'GCcomb_noN', 'GCcomb_noDES']

    # run fiducial local_png
    # analysis = 'local_png'
    # project  = f'{analysis}/base'
    # weight   = 'default-fkp-oqe'
    # tracers  = ['LRG', 'ELGnotqso', 'QSO', ('LRG','QSO'), ('LRG','ELGnotqso'), ('ELGnotqso','QSO')] # notice we can measure cross-correlations between tracers if tracer is a tuple of tracers.
    # postregions = ['GCcomb']

    # `onthefly` is used to specify whether or not to reshuffle the randoms or generate 'complete' mocks on-the-fly.
    onthefly = None # other options are 'reshuffle' or 'complete'
    
    for tracer in tracers:
        if 'png' in analysis:
            # do not compute measurements for overlapping redshifts, unless you wish to, in that case remove the slice.
            zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)[:1]
        else:
            zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
        if check_for_existing_measurements:
            # This block of code handles the file checks, and determines which measurements are missing, so you can compute them.
            exists, missing = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_catalog_fn, tracer=tracer[0] if isinstance(tracer, (list, tuple)) else tracer,
                                                                                           region='NGC', version=version), test_if_readable=False, imock=imocks2run)[:2]
            catalog_imocks = exists[1]['imock']
            rerun_by_region = {region: [] for region in regions}
            for zrange in zranges:
                for kind in stats:
                    for region in regions:
                        stats_kws = dict(basis='sugiyama-diagonal', kind=kind, stats_dir=Path(str(stats_dir).replace('global','dvs_ro')),
                                         tracer=tracer, region=region, weight=weight, zrange=zrange, version=version, project=project,
                                         extra=onthefly if onthefly else '')
                        rexists, missing, unreadable = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_stats_fn, **stats_kws), test_if_readable=True, imock=imocks2run)
                        rerun_by_region[region] += [imock for imock in catalog_imocks if (imock in unreadable[1]['imock']) or (imock not in rexists[1]['imock'])]
            rerun_by_region = {region: sorted(set(rerun)) for region, rerun in rerun_by_region.items()}
            imocks = sorted(set(imock for rerun in rerun_by_region.values() for imock in rerun))
        else:
            imocks = imocks2run
            rerun_by_region = {region: imocks for region in regions}
       
        def get_run_stats():
            # return the appropiate task manager for the job
            _tm = tm80
            if tracer in ['LRG']:
                _tm = tm
            return run_stats if mode == 'interactive' else _tm.python_app(run_stats)

        # group all the arguments required by `run_stats`.
        run_stats_kws = dict(tracer=tracer, stats_dir=stats_dir, project=project, version=version, stats=stats, analysis=analysis, onthefly=onthefly, zranges=zranges, regions=regions, weight=weight, postprocess=postprocess)
        if not postprocess_only:
            for region, region_imocks in rerun_by_region.items():
                batch_imocks = np.array_split(region_imocks, max(len(region_imocks) // max_mocks_per_batch, 1)) if len(region_imocks) else []
                for _imocks in batch_imocks:
                    get_run_stats()(imocks=_imocks, **(run_stats_kws | dict(regions=[region])))

        else:
            # this handles the combination of measurements by region
            if check_for_existing_measurements:
                postprocess_rerun = []
                for zrange in zranges:
                    for kind in stats:
                        for region in postregions:
                            stats_kws = dict(basis='sugiyama-diagonal', kind=kind, stats_dir=Path(str(stats_dir).replace('global','dvs_ro')),
                                             tracer=tracer, region=region, weight=weight, zrange=zrange, version=version, project=project,
                                             extra=onthefly if onthefly else '')
                            rexists, missing, unreadable = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_stats_fn, **stats_kws), test_if_readable=True, imock=imocks2run)
                            postprocess_rerun += [imock for imock in imocks2run if (imock in unreadable[1]['imock']) or (imock not in rexists[1]['imock'])]
                imocks = sorted(set(postprocess_rerun))
            else:
                imocks = imocks2run
            postprocess_stats(imocks=imocks, **(run_stats_kws | dict(regions=postregions)))
