"""
salloc -N 1 -C "gpu&hbm80g" -t 02:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
or source /global/homes/s/shengyu/env.sh 2pt_env
srun -n 4 python desipipe_data_splits.py
desipipe tasks -q data_splits  # check the list of tasks
desipipe spawn -q data_splits --spawn  # spawn the jobs
desipipe queues -q data_splits  # check the queue
"""

import os, sys
import itertools
import numpy as np
import functools
from pathlib import Path
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

import sys
sys.path.append('..')
from clustering_statistics import tools
setup_logging()

from mpi4py import MPI
mpicomm = MPI.COMM_WORLD

queue = Queue('data_splits')
queue.clear(kill=False)

output, error = './slurm_outputs/data_splits/slurm-%j.out', './slurm_outputs/data_splits/slurm-%j.err'
kwargs = {}
environ = Environment('nersc-cosmodesi')
tm = TaskManager(queue=queue, environ=environ)
tm = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='01:30:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu'))
tm80 = tm.clone(provider=dict(provider='nersc', time='02:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))
tmw = tm.clone(scheduler=dict(max_workers=1), provider=dict(provider='nersc', time='00:10:00',
                mpiprocs_per_worker=2250, nodes_per_worker=25, output=output, error=error, stop_after=1, constraint='cpu'))

combine_region_sources = {
    'GCcomb': ['NGC', 'SGC'],
    'NS': ['N', 'S'],
    'GCcomb_noN': ['NGCnoN', 'SGC'],
    'GCcomb_noDES': ['NGC', 'SGCnoDES'],
}

def run_stats(version='data-dr2-v2', tracer='LRG', regions=['NGC', 'SGC'], weight='default-FKP', stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', project='', stats=['mesh2_spectrum'], ibatch=None, **kwargs):
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
    from clustering_statistics import tools, setup_logging, compute_stats_from_options, fill_fiducial_options
    setup_logging()
    cache = {}
    zranges = tools.propose_fiducial('zranges', tracer)
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir, project=project)
    for region in regions:
        options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight=weight), mesh2_spectrum={'cut': True, 'auw': True}, window_mesh2_spectrum={'cut': True}, window_mesh3_spectrum={'ibatch': ibatch} if isinstance(ibatch, tuple) else {'computed_batches': ibatch})
        options = fill_fiducial_options(options)
        compute_stats_from_options(stats, get_stats_fn=get_stats_fn, cache=cache, **options)


def postprocess_stats(version='data-dr2-v2', tracer='LRG', regions=['GCcomb'], weight='default-FKP', stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', postprocess=['combine_regions'], **kwargs):
    from clustering_statistics import postprocess_stats_from_options
    zranges = tools.propose_fiducial('zranges', tracer)
    for region in regions:
        get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir)
        options = dict(
            catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight=weight),
            combine_regions={
                'stats': ['mesh2_spectrum', 'mesh3_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum', 'window_mesh3_spectrum'],
                'regions': combine_region_sources.get(region, ['NGC', 'SGC']),
            },
            mesh2_spectrum={'cut': True, 'auw': True},
            window_mesh2_spectrum={'cut': True},
        )
        postprocess_stats_from_options(postprocess, get_stats_fn=get_stats_fn, **options)


########################################################################################################################################################################################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--zrange', nargs='+', type=str, default=(0.1, 0.4), help='Redshift bins')
    # parser.add_argument('--regions', nargs='+', type=str, default=['NGC'], help='Sky regions to include.')  
    # parser.add_argument('--subver', default=None, choices=['zcmb', None], help='sub version for data catalogs')
    parser.add_argument('--tracers', nargs='+', type=str, default=['LRG', 'ELG_LOPnotqso', 'QSO'], choices=['BGS_BRIGHT-21.35', 'LRG', 'ELG_LOPnotqso', 'QSO'], help='Tracers')
    parser.add_argument('--versions', nargs='+', type=str,  default=['data-dr2-v2'], choices=['data-dr2-v2'], help='Catalog versions to use.')
    parser.add_argument('--weight_types', nargs='+', type=str, default=['default-FKP'],
                        help='Weighting schemes to use: default, default-FKP, default_thetacut, default_auw, bitwise, bitwise-FKP, bitwise_auw')
    parser.add_argument('--todo', nargs='+', type=str, default=['mesh3_spectrum'],
                        choices=['auw', 'mesh2_spectrum', 'mesh3_spectrum', 'window_mesh3_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum', 'count2_correlation', 'blinded_mesh2_spectrum'], help='Which processing steps to run.')
    args = parser.parse_args()
    mode = 'interactive'
    stats = args.todo
    check_for_existing_measurements = True

    stats_dir = tools.base_stats_dir
    analysis = 'full_shape'
    project = f'{analysis}/data_splits/blinded_data'

    regions = ['NGC', 'SGC', 'N', 'NGCnoN', 'S', 'SGCnoDES'] #galactic and imaging regions
    regions = regions + ['ACT_DR6', 'PLANCK_PR4'] + [f'GAL0{i}' for i in [40, 60]] #lensing regions
    postprocess = ['combine_regions']
    postregions = ['GCcomb', 'NS', 'GCcomb_noN', 'GCcomb_noDES']

    for version, tracer, weight_type in itertools.product(args.versions, args.tracers, args.weight_types):
        regions_to_run = list(regions)
        if check_for_existing_measurements and stats:
            rerun_regions = []
            zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
            for region in regions:
                keep_region = False
                for zrange in zranges:
                    for kind in stats:
                        stats_kws = dict(kind=kind, stats_dir=Path(str(stats_dir).replace('global', 'dvs_ro')), project=project, 
                                         tracer=tracer, region=region, weight=weight_type, zrange=zrange, version=version)
                        rexists, missing, unreadable = tools.checks_if_exists_and_readable(
                            get_fn=lambda _: tools.get_stats_fn(**stats_kws),
                            test_if_readable=True,
                            _=[0],
                        )
                        if (not rexists[0]) or unreadable[0]:
                            keep_region = True
                            break
                    if keep_region:
                        break
                if keep_region:
                    rerun_regions.append(region)
            regions_to_run = rerun_regions

        def get_run_stats():
            _tm = tm80
            if tracer in ['LRG']:
                _tm = tm
            if any('window_mesh3' in stat for stat in stats):
                _tm = tmw
            return run_stats if mode == 'interactive' else _tm.python_app(run_stats)
        if regions_to_run:
            get_run_stats()(version=version, tracer=tracer, regions=regions_to_run, stats_dir=stats_dir, stats=stats, project=project, weight=weight_type)
        if postprocess:
            postprocess_stats(version=version, tracer=tracer, regions=postregions, stats_dir=stats_dir, project=project, weight=weight_type, postprocess=postprocess)
