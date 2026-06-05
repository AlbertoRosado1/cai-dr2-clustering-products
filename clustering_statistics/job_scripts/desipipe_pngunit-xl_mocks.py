"""
Create & spawn desipipe tasks to measure the power spectrum P(k) multipoles of the
PNG-UNIT-XL mocks (fNL = 0, +20, -20) for LRG and QSO, in NGC and SGC, then combine
into GCcomb. Only `mesh2_spectrum` is measured -- no window functions.

Mimics job_scripts/desipipe_holi_mocks.py (local_png setup: meshsize 700 / cellsize 20,
ells=(0, 2), 'default-fkp-oqe' weights, dk=0.001 k-binning). Catalog paths are resolved
by a small custom get_catalog_fn, because the PNG-UNIT-XL data and randoms live in
different directories and carry a 'complete' infix:
    <bdir>/complete/<fnl>/{tracer}_complete_{region}_clustering.dat.fits   (data)
    <bdir>/randoms/{tracer}_complete_{region}_{i}_clustering.ran.fits      (randoms, i=0..17)

Usage on NERSC:
    salloc -N 1 -C "gpu&hbm80g" -t 02:00:00 --gpus 4 --qos interactive --account desi_g
    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
    python desipipe_pngunit-xl_mocks.py          # create the tasks
    desipipe spawn -q pngunit_xl_mocks --spawn   # spawn the jobs
"""
from desipipe import Queue, Environment, TaskManager, setup_logging

from clustering_statistics import tools

setup_logging()

BDIR = '/global/cfs/cdirs/desicollab/users/adrigut/PNGxHOD/dev_mocks/catalogs/DA2/v2.0/PNGUNITXL'
FNLS = ['fnl0', 'fnl20', 'fnlm20']
TRACERS = ['LRG', 'QSO']
REGIONS = ['NGC', 'SGC']
NRAN = 18                  # randoms numbered 0..17
VERSION = 'pngunit-xl'     # label only used to build the output filename/dir
ANALYSIS = 'local_png'
PROJECT = f'{ANALYSIS}/base'


def run_stats(tracer, bdir=BDIR, fnls=FNLS, regions=REGIONS, nran=NRAN,
              version=VERSION, project=PROJECT, analysis=ANALYSIS):
    """Measure P(k) for each (fNL, region), then combine NGC+SGC -> GCcomb.
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
    from clustering_statistics import (tools, setup_logging, compute_stats_from_options,
                                       postprocess_stats_from_options)
    setup_logging()

    bdir = Path(bdir)
    zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)[:1]
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=tools.base_stats_dir, project=project)
    cache = {}

    def make_get_catalog_fn(fnl, region):
        # Files are region-specific; compute_stats reads region='ALL' and then masks to
        # `region` (a no-op here), so the passed region is ignored.
        def get_catalog_fn(kind='data', nran=nran, **kwargs):
            if 'data' in kind:
                return bdir / 'complete' / fnl / f'{tracer}_complete_{region}_clustering.dat.fits'
            if 'randoms' in kind:
                return [bdir / 'randoms' / f'{tracer}_complete_{region}_{i}_clustering.ran.fits'
                        for i in range(nran)]
            raise NotImplementedError(f'unsupported catalog kind {kind!r}')
        return get_catalog_fn

    for fnl in fnls:
        for region in regions:
            compute_stats_from_options(
                ['mesh2_spectrum'], analysis=analysis, cache=cache,
                get_catalog_fn=make_get_catalog_fn(fnl, region),
                get_stats_fn=functools.partial(get_stats_fn, extra=fnl),
                catalog=dict(version=version, tracer=tracer, region=region, zrange=zranges, nran=nran),
            )
        # combine NGC + SGC -> GCcomb (sums numerators and normalizations)
        postprocess_stats_from_options(
            ['combine_regions'], analysis=analysis, get_stats_fn=get_stats_fn, extra=fnl,
            catalog=dict(version=version, tracer=tracer, zrange=zranges),
            combine_regions={'stats': ['mesh2_spectrum'], 'regions': regions},
        )


if __name__ == '__main__':
    queue = Queue('pngunit_xl_mocks')
    queue.clear(kill=False)
    output, error = 'slurm_outputs/pngunit_xl_mocks/slurm-%j.out', 'slurm_outputs/pngunit_xl_mocks/slurm-%j.err'
    # Uses the desi-clustering bundled with cosmodesi. To run a dev checkout instead, add e.g.
    # command='export PYTHONPATH=/path/to/your/desi-clustering:$PYTHONPATH' to Environment(...).
    environ = Environment('nersc-cosmodesi')
    tm = TaskManager(queue=queue, environ=environ)
    tm = tm.clone(scheduler=dict(max_workers=len(TRACERS)),
                  provider=dict(provider='nersc', time='01:00:00', mpiprocs_per_worker=4,
                                nodes_per_worker=1, output=output, error=error, constraint='gpu&hbm80g'))
    app = tm.python_app(run_stats)
    for tracer in TRACERS:
        app(tracer=tracer)
