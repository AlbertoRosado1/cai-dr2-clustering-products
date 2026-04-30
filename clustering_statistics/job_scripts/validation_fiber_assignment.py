#!/usr/bin/env python
"""
Script to create and spawn desipipe tasks to compute clustering measurements on abacus mocks.
To create and spawn the tasks on NERSC, use the following commands:
```bash
salloc -N 1 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python desipipe_abacus_mocks.py          # create the list of tasks
desipipe tasks -q abacus_mocks           # check the list of tasks
desipipe spawn -q abacus_mocks --spawn   # spawn the jobs
desipipe queues -q abacus_mocks          # check the queue
```
"""
import os
from pathlib import Path
import functools

import numpy as np
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

from clustering_statistics import tools

setup_logging()

queue = Queue('abacus_mocks')
queue.clear(kill=False)

output, error = 'slurm_outputs/abacus_mocks/slurm-%j.out', 'slurm_outputs/abacus_mocks/slurm-%j.err'
kwargs = {}
environ = Environment('nersc-cosmodesi')
tm = TaskManager(queue=queue, environ=environ)
tm = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='02:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu'))
tm80 = tm.clone(provider=dict(provider='nersc', time='02:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))
tmw = tm.clone(scheduler=dict(max_workers=1), provider=dict(provider='nersc', time='00:10:00',
                mpiprocs_per_worker=2250, nodes_per_worker=25, output=output, error=error, stop_after=1, constraint='cpu'))


def run_stats(tracer='LRG', project='', version='abacus-hf-dr2-v2-altmtl', onthefly=None, imocks=[150], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], weight='default-FKP', analysis='full_shape', regions=['NGC','SGC'], ibatch=None, postprocess=None, zranges=None, **kwargs):
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
            mesh2_spectrum = {'cut': True if 'full_shape' in analysis else None, 
                              'auw': True if 'altmtl' in version and onthefly is None and 'full_shape' in analysis else None}
            window_mesh2_spectrum = {'cut': True if 'full_shape' in analysis else None}
            
            options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight=weight, imock=imock), 
                           mesh2_spectrum=mesh2_spectrum, window_mesh2_spectrum=window_mesh2_spectrum,
                           window_mesh3_spectrum={'ibatch': ibatch} if isinstance(ibatch, tuple) else {'computed_batches': ibatch})
            options = fill_fiducial_options(options, analysis=analysis)
            
            for itracer in options['catalog']:
                options['catalog'][itracer]['zranges'] = zranges # override fiducial zranges 
                options['catalog'][itracer]['expand'] = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=itracer, nran=options['catalog'][itracer]['nran'])}
                if onthefly is not None and onthefly.startswith('complete'):
                    options['catalog'][itracer]['complete'] = {'downsample_nobj': 'downsample' in onthefly}
                elif onthefly == 'reshuffle':
                    options['catalog'][itracer]['reshuffle'] = {'merged_data_fn': tools.get_catalog_fn(kind='data', **(options['catalog'][itracer] | dict(region='ALL')))}                
            
            get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir, project=project, extra=onthefly if onthefly else '')
            compute_stats_from_options(stats, analysis=analysis, get_stats_fn=get_stats_fn, cache=cache, **options)

    # postprocess
    if postprocess:
        postprocess_options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight, imock=imocks[0]), imocks=imocks, 
                                   combine_regions={'stats': stats}, mesh2_spectrum=mesh2_spectrum, window_mesh2_spectrum=window_mesh2_spectrum)
        postprocess_stats_from_options(postprocess, analysis=analysis, get_stats_fn=get_stats_fn, **postprocess_options)


def postprocess_stats(tracer='LRG', analysis='full_shape', project='', version='abacus-hf-dr2-v2-altmtl', onthefly=None, imocks=[150], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum', 'mesh3_spectrum'], weight='default-FKP', postprocess=['combine_regions'], zranges=None, **kwargs):
    from clustering_statistics import postprocess_stats_from_options
    if zranges is None:
        zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
    options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight, imock=imocks[0]), imocks=imocks, combine_regions={'stats': stats}, mesh2_spectrum={'cut': True, 'auw': True}, window_mesh2_spectrum={'cut': True})
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
    version  = 'abacus-hf-dr2-v2-altmtl'
    # version = 'abacus-2ndgen-dr2-complete'
    # version = 'abacus-2ndgen-dr2-altmtl'
    check_for_existing_measurements = False

    # to run job
    mode = 'interactive'
    imocks = np.arange(25)
    stats_dir = tools.base_stats_dir

    # run fiducial full_shape
    tracers = ['LRG', 'ELG_LOPnotqso']

    # run data_splits for lensing group with full_shape setup 
    stats = ['window_mesh2_spectrum', 'mesh3_spectrum'][:1]
    postprocess = ['combine_regions'][:0]
    analysis = 'full_shape'
    project = f'{analysis}/fiber_assignment_systematics'
    weight = 'default-FKP'
    regions = ['NGC', 'SGC']
    max_mocks_per_batch = 5 

    onthefly = None
    #onthefly = 'complete'
    #onthefly = 'complete-downsample'
    
    for tracer in tracers:
        if 'png' in analysis:
            # do not compute measurements for overlapping redshifts
            zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)[:1]
        else:
            zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)[-1:]
       
        def get_run_stats():
            _tm = tm80
            if tracer in ['LRG']:
                _tm = tm
            if any('window_mesh3' in stat for stat in stats):
                _tm = tmw
            return run_stats if mode == 'interactive' else _tm.python_app(run_stats)

        run_stats_kws = dict(tracer=tracer, stats_dir=stats_dir, project=project, version=version, analysis=analysis, onthefly=onthefly, zranges=zranges, regions=regions, weight=weight, postprocess=postprocess)
        if True:
            if any('window' in stat for stat in stats):
                _imocks = [0]
                nbatches = 1
                tasks = []
                for ibatch in range(nbatches):
                    task = get_run_stats()(imocks=_imocks, ibatch=(ibatch, nbatches), stats=stats, **run_stats_kws)
                    tasks.append(task)
                if nbatches >= 1:
                    # Add dependence on other tasks
                    get_run_stats()(imocks=_imocks, ibatch=nbatches, tasks=tasks, stats=stats, **run_stats_kws)
            elif any('covariance' in stat for stat in stats):
                get_run_stats()(imocks=[0], stats=stats, **run_stats_kws)
            elif stats:
                batch_imocks = np.array_split(imocks, max(len(imocks) // max_mocks_per_batch, 1)) if len(imocks) else []
                for _imocks in batch_imocks:
                    get_run_stats()(imocks=_imocks, stats=stats, **run_stats_kws)
        if postprocess:
            postprocess_stats(imocks=imocks, **run_stats_kws)