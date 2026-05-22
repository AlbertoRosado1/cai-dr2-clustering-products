"""
Script to create and spawn desipipe tasks to compute 2pcfs on glam-uchuu bgs mocks (on reserved nodes).
To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python reservation_bgs_glam_xi.py         # create the list of tasks
desipipe tasks  -q bgs_glam_xi         # check the list of tasks
desipipe spawn  -q bgs_glam_xi --spawn # spawn the jobs
desipipe queues -q bgs_glam_xi         # check the queue
```
"""
import os
from pathlib import Path
import functools

import numpy as np
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

from clustering_statistics import tools

setup_logging()

reservation_name = '_CAP_bgs_glam_xi'
reservation_time = '24:00:00'

queue = Queue('bgs_glam_xi')
queue.clear(kill=False)

output, error = 'slurm_outputs/bgs_glam_xi/slurm-%j.out', 'slurm_outputs/bgs_glam_xi/slurm-%j.err'
kwargs = {}
# environ = Environment('nersc-cosmodesi')
environ = Environment('nersc-cosmodesi', command='export PYTHONPATH=$HOME/LSScode/dr2-clustering-analysis/:$PYTHONPATH')
tm = TaskManager(queue=queue, environ=environ)
tm = tm.clone(scheduler=dict(max_workers=30), provider=dict(provider='nersc', time=reservation_time, mpiprocs_per_worker=4, output=output, error=error,
                                                           stop_after=1, constraint='gpu', live_jobids=False, kwargs={'reservation': reservation_name}))
tm80 = tm.clone(provider=dict(provider='nersc', time=reservation_time, mpiprocs_per_worker=4, output=output, error=error,
                              stop_after=1, constraint='gpu&hbm80g', kwargs={'reservation': reservation_name}))

combine_region_sources = {
    'GCcomb': ['NGC', 'SGC'],
    'NS': ['N', 'S'],
    'GCcomb_noN': ['NGCnoN', 'SGC'],
    'GCcomb_noDES': ['NGC', 'SGCnoDES'],
}

def run_stats(tracer='LRG', project='', version='glam-uchuu-v2-altmtl', onthefly=None, imocks=[150], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], weight='default-FKP', analysis='full_shape', regions=['NGC','SGC'], ibatch=None, postprocess=None, zranges=None, **kwargs):
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

    # postprocess
    if postprocess:
        postprocess_options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight, imock=imocks[0]), imocks=imocks, 
                                   combine_regions={'stats': stats}, mesh2_spectrum=mesh2_spectrum, window_mesh2_spectrum=window_mesh2_spectrum)
        postprocess_stats_from_options(postprocess, analysis=analysis, get_stats_fn=get_stats_fn, **postprocess_options)


if __name__ == '__main__':

    stats, postprocess = [], []
    # version  = 'glam-uchuu-v2-altmtl'
    version  = 'glam-uchuu-bgs-altmtl'
    check_for_existing_measurements = True # Important because it also checks if the clustering catalogs exist.

    # run on interactive node for testing changes
    # mode = 'interactive'
    # imocks2run = 151 + np.arange(1)
    # stats_dir  = Path(os.getenv('SCRATCH')) / 'cai-dr2-benchmarks' 
    
    # to run job
    mode = 'slurm'
    imocks2run = np.arange(1500)
    if version == 'glam-uchuu-v2-altmtl':
        imocks2run = np.loadtxt('../../helper_scripts/glam-uchuu-v2-altmtl_dark-time_imocks_for_covariance.txt',dtype=int)
    stats_dir  = tools.base_stats_dir

    # run fiducial full_shape
    stats = ['particle2_correlation']
    postprocess = ['combine_regions']
    analysis = 'full_shape'
    project  = f'{analysis}/base'
    weight   = 'default-FKP'
    regions  = ['NGC','SGC']
    tracers = ['BGS_BRIGHT-21.35']
    max_mocks_per_batch_qso = max_mocks_per_batch_others = 49
    postregions = ['GCcomb']

    onthefly = None

    for tracer in tracers:
        if tracer == 'QSO':
             max_mocks_per_batch = max_mocks_per_batch_qso # allow mocks to be processed since QSOs only have one zbin
        else:
             max_mocks_per_batch = max_mocks_per_batch_others
        if 'png' in analysis:
            # do not compute measurements for overlapping redshifts
            zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)[:1]
        else:
            zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
        if check_for_existing_measurements:
            exists, missing = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_catalog_fn, tracer=tracer[0] if isinstance(tracer, (list, tuple)) else tracer,
                                                                                           region='NGC', version=version), test_if_readable=False, imock=imocks2run)[:2]
            catalog_imocks = exists[1]['imock']
            rerun = []
            for zrange in zranges[-1:]: # only check last zrange to speed up this step.
                for kind in stats:

                    stats_kws = dict(basis='sugiyama-diagonal', kind=kind, stats_dir=Path(str(stats_dir).replace('global','dvs_ro')),
                                     tracer=tracer, region=regions[-1], weight=weight, zrange=zrange, version=version, project=project,
                                     extra=onthefly if onthefly else '')
                    rexists, missing, unreadable = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_stats_fn, **stats_kws), test_if_readable=True, imock=imocks2run)
                    rerun += [imock for imock in catalog_imocks if (imock in unreadable[1]['imock']) or (imock not in rexists[1]['imock'])]
            imocks = sorted(set(rerun))
        else:
            imocks = imocks2run

        def get_run_stats():
            _tm = tm # reservation is for regular GPU nodes
            # _tm = tm80 # reservation is for large memory GPU nodes
            # if tracer in ['LRG','BGS_BRIGHT-21.35']:
            #     _tm = tm
            return run_stats if mode == 'interactive' else _tm.python_app(run_stats)

        run_stats_kws = dict(tracer=tracer, stats_dir=stats_dir, project=project, version=version, stats=stats, analysis=analysis, onthefly=onthefly, zranges=zranges, regions=regions, weight=weight, postprocess=postprocess)
        batch_imocks = np.array_split(imocks, max(len(imocks) // max_mocks_per_batch, 1)) if len(imocks) else []
        for _imocks in batch_imocks:
            get_run_stats()(imocks=_imocks, **run_stats_kws)