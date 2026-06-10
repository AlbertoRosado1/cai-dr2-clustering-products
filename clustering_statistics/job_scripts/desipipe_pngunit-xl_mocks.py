"""
Power spectra for the PNGUNIT-XL mocks (local PNG key project), with optional
shuffled-randoms RIC (radial integral constraint). Toggle it with SHUFFLE_RANDOMS
below. This script computes P(k) only -- no window functions.

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

# --- shuffled-randoms (radial integral constraint) ---
SHUFFLE_RANDOMS = True  # toggle: resample random redshifts from the mock-data n(z)
SHUFFLE_SEED    = 42    # base RNG seed (reproducible for a fixed number of MPI ranks)


def run_stats(tracer, bdir=BDIR, fnls=FNLS, regions=REGIONS, version=VERSION,
              project=PROJECT, analysis=ANALYSIS, weight=WEIGHT, nran_max=NRAN_MAX,
              shuffle_randoms=SHUFFLE_RANDOMS, shuffle_seed=SHUFFLE_SEED):
    """Per fNL: measure P(k) for each region, then combine NGC+SGC -> GCcomb.
    If `shuffle_randoms`, the random redshifts are resampled from the data within
    each tracer's photometric sub-regions (radial integral constraint) before the
    spectrum is measured.
    `tracer` may be a string (auto) or a 2-tuple (cross, e.g. ('LRG','QSO')).
    Self-contained: everything here runs on the compute nodes."""
    import os
    import functools
    from pathlib import Path
    import numpy as np
    from mpi4py import MPI
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
    #
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

    def make_get_catalog_fn(fnl):
        # tracer- AND region-aware: compute_stats passes the per-tracer `tracer` (and `nran`)
        # in kwargs, and reads region='ALL' (the full footprint) -- which the shuffling
        # wrapper also uses to draw redshifts from the whole-footprint data. For 'ALL' we
        # return the NGC and SGC files together; for a single region we return just that
        # region's files.
        def get_catalog_fn(kind='data', **kwargs):
            tr = kwargs.get('tracer', tracers_t[0])
            reg = kwargs.get('region')
            n = kwargs.get('nran', nran.get(tr, nran_scalar))
            if reg in ('NGC', 'SGC'):
                regs = [reg]
            elif reg == 'ALL':
                regs = ['NGC', 'SGC']
            else:
                raise NotImplementedError(f'unsupported region {reg!r}')
            if 'data' in kind:
                return [bdir / 'complete' / fnl / f'{tr}_complete_{r}_clustering.dat.fits' for r in regs]
            if 'randoms' in kind:
                return [bdir / 'randoms' / f'{tr}_complete_{r}_{i}_clustering.ran.fits'
                        for r in regs for i in range(n)]
            raise NotImplementedError(f'unsupported catalog kind {kind!r}')
        return get_catalog_fn

    def make_shuffling_read_catalog(seed, regions_by_tracer, _base=tools.read_catalog):
        # Wrap tools.read_catalog so that, when randoms are read, their redshifts are
        # resampled from the mock data (the "shuffled randoms" radial integral constraint).
        def _sample_data_to_randoms(randoms, data, rng, sub_regions):
            # Reassign each random's REDSHIFT by drawing with replacement from the mock-data
            # galaxies in the same `select_region` sub-region, so the randoms' radial
            # selection matches the region-dependent mock-data n(z). 
            if 'WEIGHT_FKP' in randoms and 'WEIGHT_FKP' not in data:
                raise ValueError('Randoms carry WEIGHT_FKP but the mock data does not; cannot '
                                 'keep the FKP weight consistent with the resampled redshift.')
            do_fkp = ('WEIGHT_FKP' in randoms) and ('WEIGHT_FKP' in data)
            randoms['Z'] = -randoms.ones()  # placeholder, overwritten below
            assigned = np.zeros(randoms['RA'].size, dtype='?')
            for region in sub_regions:
                mask_data = tools.select_region(data['RA'], data['DEC'], region=region)
                mask_randoms = tools.select_region(randoms['RA'], randoms['DEC'], region=region)
                n_data, n_randoms = int(mask_data.sum()), int(mask_randoms.sum())
                if n_randoms == 0 or n_data == 0:
                    continue
                index = rng.choice(n_data, size=n_randoms)
                randoms['Z'][mask_randoms] = data['Z'][mask_data][index]
                if do_fkp:
                    randoms['WEIGHT_FKP'][mask_randoms] = data['WEIGHT_FKP'][mask_data][index]
                assigned |= mask_randoms
            if not assigned.all():  # catch any randoms outside the listed sub-regions
                mask_randoms = ~assigned
                index = rng.choice(data['Z'].size, size=int(mask_randoms.sum()))
                randoms['Z'][mask_randoms] = data['Z'][index]
                if do_fkp:
                    randoms['WEIGHT_FKP'][mask_randoms] = data['WEIGHT_FKP'][index]
            if np.any(randoms['Z'] == -1.):
                raise ValueError('Placeholder z = -1. remains after reshuffling the randoms.')

        def read_catalog(kind='data', **kw):
            catalog = _base(kind=kind, **kw)
            if kind != 'randoms':
                return catalog
            # this tracer's photometric sub-regions (from propose_photoregions)
            tr = kw.get('tracer')
            sub_regions = regions_by_tracer.get(tr, next(iter(regions_by_tracer.values())))
            # Full data on every rank, drawn over the whole footprint (region='ALL').
            data_kw = {k: v for k, v in kw.items() if k not in ('concatenate', 'mpicomm', 'kind')}
            data = _base(kind='data', concatenate=True, mpicomm=MPI.COMM_SELF, **data_kw)
            cats = catalog if isinstance(catalog, (list, tuple)) else [catalog]
            for ifn, c in enumerate(cats):
                rank = c.mpicomm.rank if hasattr(c, 'mpicomm') else 0
                # per-(file, rank) seed: independent draws across ranks, reproducible
                # for a fixed number of MPI ranks.
                rng = np.random.RandomState(seed=seed + ifn + 7919 * rank)
                _sample_data_to_randoms(c, data, rng, sub_regions)
            return catalog
        return read_catalog

    read_catalog = tools.read_catalog
    if shuffle_randoms:
        # per-tracer photometric sub-regions from the fiducial `propose_photoregions`
        # (LRG -> ['N', 'S'], QSO -> ['N', 'SnoDES', 'DES']); same source the data
        # pipeline uses for the forward-model RIC (`ric_regions`).
        shuffle_regions_by_tracer = {t: tools.propose_fiducial('window_mesh2_spectrum_fm', tracer=t)['ric_regions']
                                     for t in tracers_t}
        read_catalog = make_shuffling_read_catalog(shuffle_seed, shuffle_regions_by_tracer)

    for fnl in fnls:
        cache = {}
        # fNL goes into the directory (not `extra`): see module docstring. It is folded into the
        # catalog `version`, which get_stats_fn appends *after* `project`, so the path reads
        # .../<project>/<version>/<fnl>/... -> .../local_png/base/pngunit-xl/fnl0/...
        fnl_version = f'{version}/{fnl}'
        get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir, project=project)
        get_catalog_fn = make_get_catalog_fn(fnl)

        # --- power spectrum P(k), FKP/OQE weights (randoms shuffled when SHUFFLE_RANDOMS) ---
        for region in regions:
            compute_stats_from_options(
                ['mesh2_spectrum'], analysis=analysis, cache=cache,
                get_catalog_fn=get_catalog_fn,
                get_stats_fn=get_stats_fn,
                read_catalog=read_catalog,  # shuffles random redshifts when SHUFFLE_RANDOMS
                prepare_catalog=prepare_catalog,
                catalog=dict(version=fnl_version, tracer=tracer, region=region,
                             zrange=zranges, nran=nran_scalar, weight=weight, FKP_P0=None),
            )
        # combine NGC + SGC -> GCcomb
        postprocess_stats_from_options(
            ['combine_regions'], analysis=analysis, get_stats_fn=get_stats_fn,
            catalog=dict(version=fnl_version, tracer=tracer, zrange=zranges, weight=weight),
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
    #               provider=dict(provider='nersc', time='02:00:00', mpiprocs_per_worker=4,
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