"""
Script to create and spawn desipipe tasks to compute merged catalogs.
To create and spawn the tasks on NERSC, use the following commands:
```bash
salloc -N 1 -C cpu -t 04:00:00 --qos interactive 
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python desipipe_merged_catalogs.py          # create the list of tasks
desipipe tasks  -q merged_catalogs          # check the list of tasks
desipipe spawn  -q merged_catalogs --spawn  # spawn the jobs
desipipe queues -q merged_catalogs          # check the queue
```
"""
import os
import sys
from pathlib import Path
import functools
from time import time

import numpy as np
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

from clustering_statistics import tools


setup_logging()

queue = Queue('merged_catalogs')
queue.clear(kill=False)

output, error = 'slurm_outputs/merged_catalogs/slurm-%j.out', 'slurm_outputs/merged_catalogs/slurm-%j.err'
kwargs = {}
# tmp_dir = Path(os.getenv('SCRATCH'), 'tmp')
# tmp_dir.mkdir(exist_ok=True)
environ = Environment('nersc-cosmodesi')
tm = TaskManager(queue=queue, environ=environ)
tm = tm.clone(scheduler=dict(max_workers=30),
              provider=dict(provider='nersc', time='00:30:00', mpiprocs_per_worker=1, nodes_per_worker=0.2,
                            output=output, error=error, stop_after=1, constraint='cpu'))


def merge_data_catalogs(output_fn, input_fns, merge_catalogs=tools.merge_data_catalogs, read_catalog=tools._read_catalog, factor=1):
    from clustering_statistics.tools import setup_logging
    setup_logging()
    merge_catalogs(output_fn, input_fns, read_catalog=read_catalog, factor=factor)


def merge_randoms_catalogs(output_fn, input_fns, parent_randoms_fn=None, merge_catalogs=tools.merge_randoms_catalogs,
                           read_catalog=tools._read_catalog, expand_randoms=tools.expand_randoms, input_data_fns=None, factor=1):
    import functools
    from clustering_statistics.tools import setup_logging
    setup_logging()
    expand_randoms = functools.partial(expand_randoms, from_randoms=['RA', 'DEC'], from_data=['FRAC_TLOBS_TILES'])
    merge_catalogs(output_fn, input_fns, parent_randoms_fn=parent_randoms_fn, read_catalog=read_catalog, expand_randoms=expand_randoms, input_data_fns=input_data_fns, factor=factor)


if __name__ == '__main__':

    #mode = 'slurm'
    mode = 'interactive'

    # version = 'glam-uchuu-v2-altmtl'
    version = 'holi-v3-altmtl'
    out_dir  = tools.base_stats_dir / 'merged_catalogs' / version
    # out_dir = Path(os.getenv('SCRATCH')) / 'cai-dr2-benchmarks' / 'merged_catalogs' / version

    # kinds = ['data', 'randoms']
    kinds = ['data']
    tracers = ['LRG', 'ELG_LOPnotqso', 'ELGnotqso', 'QSO']
    # tracers = ['ELGnotqso']
    regions = ['NGC', 'SGC']
    # imocks = 150 + np.arange(50) # number of mocks to merge
    imocks = np.arange(100) # number of mocks to merge
    # if version == 'holi-v3-altmtl':
    #     # do not perform measurements on dubious mocks
    #     bad_imocks = np.loadtxt('../helper_scripts/dubious_holi-v3-altmtl.txt',dtype=int)
    #     imocks = imocks[~np.isin(imocks,bad_imocks)]
    nran_list = np.arange(18) # randoms to process
    fraction_to_keep = 0.1 # keep only ~10% of the catalogs
    
    def get_taskmanager(fun):
        _tm = tm
        return fun if mode == 'interactive' else _tm.python_app(fun)
        
    for kind in kinds:
        for tracer in tracers:
            for region in regions:
                catalog_kws = dict(version=version, tracer=tracer, region=region)

                if 'data' in kind:
                    # Merge data mock catalogs
                    input_data_fns,_ = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_catalog_fn, kind=kind, **catalog_kws),
                                                                           test_if_readable=False, imock=imocks)[0]
                    factor = len(input_data_fns) * fraction_to_keep
                    output_data_fn = tools.get_catalog_fn(kind=kind, cat_dir=out_dir, **catalog_kws)
                    get_taskmanager(merge_data_catalogs)(output_data_fn, input_data_fns, factor=factor)

                if 'randoms' in kind:
                    # Merge randoms catalogs
                    def get_single_fn(kind='randoms', nran=0, **kw):
                        # Return single random filename
                        return tools.get_catalog_fn(kind=kind, **kw, nran=[nran])[0]

                    exists, missing, unreadable = tools.checks_if_exists_and_readable(get_fn=functools.partial(get_single_fn, kind='randoms', cat_dir=out_dir, **catalog_kws),
                                                                                      nran=nran_list)
                    rerun = [inran for inran in nran_list if (inran in unreadable[1]['nran']) or (inran not in exists[1]['nran'])]
                    for iran in rerun:
                        parent_randoms_fn = get_single_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=iran)
                        input_randoms_fns, kw_fns = tools.checks_if_exists_and_readable(get_fn=functools.partial(get_single_fn, kind='randoms', nran=iran, **catalog_kws),
                                                                                  test_if_readable=False, imock=imocks)[0]
                        input_data_fns = [tools.get_catalog_fn(kind='data', **(catalog_kws | dict(region='ALL', imock=imock))) for imock in kw_fns['imock']]
                        factor = len(input_data_fns) * fraction_to_keep
                        output_randoms_fn = get_single_fn(kind=kind, cat_dir=out_dir, nran=iran, **catalog_kws)
                        get_taskmanager(merge_randoms_catalogs)(output_randoms_fn, input_randoms_fns, parent_randoms_fn=parent_randoms_fn, input_data_fns=input_data_fns, factor=factor)
