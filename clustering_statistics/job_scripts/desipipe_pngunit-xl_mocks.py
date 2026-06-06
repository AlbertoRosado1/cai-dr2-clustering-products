"""
Usage on NERSC:
    salloc -N 1 -C "gpu&hbm80g" -t 02:00:00 --gpus 4 --qos interactive --account desi_g
    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
    python desipipe_pngunit-xl_mocks.py          # create the tasks
    desipipe spawn -q pngunit_xl_mocks --spawn   # spawn the jobs (runs in this allocation)
    desipipe queues -q pngunit_xl_mocks          # check the queue
    desipipe tasks  -q pngunit_xl_mocks          # inspect task states / errors
"""

from desipipe import Queue, Environment, TaskManager, setup_logging
from clustering_statistics import tools
setup_logging()
BDIR = '/global/cfs/cdirs/desicollab/users/adrigut/PNGxHOD/dev_mocks/catalogs/DA2/v2.0/PNGUNITXL'
FNLS = ['fnl0', 'fnl20', 'fnlm20']
TRACERS = ['LRG', 'QSO']
CROSS = ('LRG', 'QSO')     
REGIONS = ['NGC', 'SGC']
NRAN_MAX = 18  # randoms numbered 0..17 for PNGUNIT-XL
VERSION = 'pngunit-xl'    
ANALYSIS = 'local_png'
WEIGHT  = 'default-fkp-oqe'
PROJECT = f'{ANALYSIS}/base'

def run_stats(tracer, bdir=BDIR, fnls=FNLS, regions=REGIONS, version=VERSION,
              project=PROJECT, analysis=ANALYSIS, weight=WEIGHT, nran_max=NRAN_MAX):
    """Measure P(k) for each (fNL, region), then combine NGC+SGC -> GCcomb.
    `tracer` may be a string (auto) or a 2-tuple (cross, e.g. ('LRG','QSO')).
    Self-contained: everything here runs on the compute nodes."""
    import os
    import functools
    from pathlib import Path
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    try: jax.distributed.initialize()
    except (RuntimeError, ValueError): pass
    from clustering_statistics import tools, setup_logging, compute_stats_from_options, postprocess_stats_from_options
    setup_logging()
    bdir = Path(bdir)
    # individual tracers involved (1 for auto, 2 for cross)
    tracers_t = tuple(tracer) if isinstance(tracer, (tuple, list)) else (tracer,)
    zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)[:1]
    # fiducial directory
    stats_dir = tools.base_stats_dir
    # if you don't have write access to cfs, save to local scratch instead, for example:
    stats_dir = Path(os.environ['SCRATCH']) / 'measurements' 
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir, project=project)
    #
    cache = {}
    # per-tracer FKP_P0 (only used to invert WEIGHT_FKP -> NX) and nran, from the fiducial.
    _filled = tools.fill_fiducial_options(dict(catalog=dict(tracer=tracer, zrange=zranges)), analysis=analysis)
    p0   = {t: _filled['catalog'][t].get('FKP_P0') for t in tracers_t}
    nran = {t: min(_filled['catalog'][t].get('nran'), nran_max) for t in tracers_t}
    nran_scalar = min(nran.values())   # passed in the catalog options (same file count per tracer)

    def prepare_catalog(catalog, kind='data', _base=tools.prepare_catalog, **kw):
        # inject NX into the RAW randoms before the default prep (which would drop WEIGHT_FKP).
        # NX==0 (the only thing the downstream mask tests) is p0-independent, so the exact p0
        # is immaterial here; we still use the per-tracer value for correctness.
        if 'randoms' in kind:
            tr = kw.get('tracer')
            p0t = p0.get(tr, next(iter(p0.values())))
            cats = catalog if isinstance(catalog, (list, tuple)) else [catalog]
            for c in cats:
                if 'NX' not in c.columns() and 'WEIGHT_FKP' in c.columns():
                    c['NX'] = (1.0 / c['WEIGHT_FKP'] - 1.0) / p0t
        return _base(catalog, kind=kind, **kw)

    def make_get_catalog_fn(fnl, region):
        # tracer-aware: compute_stats passes the per-tracer `tracer` (and `nran`) in kwargs,
        # so the cross reads each tracer's region-specific files.
        def get_catalog_fn(kind='data', **kwargs):
            tr = kwargs.get('tracer', tracers_t[0])
            n = kwargs.get('nran', nran.get(tr, nran_scalar))
            if 'data' in kind:
                return bdir / 'complete' / fnl / f'{tr}_complete_{region}_clustering.dat.fits'
            if 'randoms' in kind:
                return [bdir / 'randoms' / f'{tr}_complete_{region}_{i}_clustering.ran.fits'
                        for i in range(n)]
            raise NotImplementedError(f'unsupported catalog kind {kind!r}')
        return get_catalog_fn

    for fnl in fnls:
        for region in regions:
            compute_stats_from_options(
                ['mesh2_spectrum'], analysis=analysis, cache=cache,
                get_catalog_fn=make_get_catalog_fn(fnl, region),
                get_stats_fn=functools.partial(get_stats_fn, extra=fnl),
                prepare_catalog=prepare_catalog,
                catalog=dict(version=version, tracer=tracer, region=region,
                             zrange=zranges, nran=nran_scalar, weight=weight, FKP_P0=None),
            )
        # combine NGC + SGC -> GCcomb 
        postprocess_stats_from_options(
            ['combine_regions'], analysis=analysis, get_stats_fn=get_stats_fn, extra=fnl,
            catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight),
            combine_regions={'stats': ['mesh2_spectrum'], 'regions': regions},
        )

if __name__ == '__main__':
    queue = Queue('pngunit_xl_mocks')
    queue.clear(kill=False)
    output, error = 'slurm_outputs/pngunit_xl_mocks/slurm-%j.out', 'slurm_outputs/pngunit_xl_mocks/slurm-%j.err'
    environ = Environment('nersc-cosmodesi')
    tm = TaskManager(queue=queue, environ=environ)
    # Batch-submission variant (submits separate sbatch jobs); kept for reference.
    # tm = tm.clone(scheduler=dict(max_workers=len(TRACERS) + 1),
    #               provider=dict(provider='nersc', time='01:00:00', mpiprocs_per_worker=4,
    #                             nodes_per_worker=1, output=output, error=error, constraint='gpu&hbm80g'))
    # Local provider: run the workers inside the current interactive allocation via srun.
    tm = tm.clone(
        scheduler=dict(max_workers=1),
        provider=dict(provider='local',
              mpiprocs_per_worker=4,
              mpiexec='srun -n {mpiprocs:d} --gpus-per-task=1 --gpu-bind=none {cmd}'),
    )
    app = tm.python_app(run_stats)
    for tracer in TRACERS:
        app(tracer=tracer)
    app(tracer=CROSS)   