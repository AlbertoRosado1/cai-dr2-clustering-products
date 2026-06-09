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

# --- which products to compute (toggle independently) ---
WITH_WINDOW = False        # exact (wide-angle) window matrix, FKP/OQE weights (`window_mesh2_spectrum`)
WITH_RIC    = False        # RIC correction to the window via the forward model (auto-enables WITH_WINDOW)

# --- forward-model (RIC) window configuration ---
ELLSOUT = None             # multipoles for the FM window; None -> use those of the spectrum (0, 2)
N_REALIZATIONS = 10        # number of gaussian realizations for the forward model
SEEDS = [85, 95, 75, 65, 91, 37, 46, 87, 19, 38]  # one seed per realization (matches fiducial defaults)
FM_NRAN = 1                # randoms files for the FM read (full footprint -> memory heavy)
FM_BATCH_SIZE = 3          # parallel window computations in the FM (lower if OOM)


def run_stats(tracer, bdir=BDIR, fnls=FNLS, regions=REGIONS, version=VERSION,
              project=PROJECT, analysis=ANALYSIS, weight=WEIGHT, nran_max=NRAN_MAX,
              with_window=WITH_WINDOW, with_ric=WITH_RIC, ellsout=ELLSOUT,
              n_realizations=N_REALIZATIONS, seeds=SEEDS, fm_nran=FM_NRAN,
              fm_batch_size=FM_BATCH_SIZE):
    """Per fNL: measure P(k) for each region (and, if `with_window`, the exact
    wide-angle window), combine NGC+SGC -> GCcomb, then (if `with_ric`) compute
    the forward-model window and add the RIC correction (combine seeds, then
    combine regions). `with_ric` auto-enables the exact window since the RIC
    combine reads it from disk.
    `tracer` may be a string (auto) or a 2-tuple (cross, e.g. ('LRG','QSO')).
    Self-contained: everything here runs on the compute nodes."""
    import os
    import functools
    import itertools
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
        # in kwargs, and forces region='ALL' when reading the full footprint for the
        # forward-model window. For 'ALL' we return the NGC and SGC files together; for a
        # single region we return just that region's files.
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

    for fnl in fnls:
        cache = {}
        # fNL goes into the directory (not `extra`): see module docstring. It is folded into the
        # catalog `version`, which get_stats_fn appends *after* `project`, so the path reads
        # .../<project>/<version>/<fnl>/... -> .../local_png/base/pngunit-xl/fnl0/...
        fnl_version = f'{version}/{fnl}'
        get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir, project=project)
        get_catalog_fn = make_get_catalog_fn(fnl)

        # --- Stage 1: power spectrum (+ exact wide-angle window), FKP/OQE weights ---
        # The RIC forward model reads the analytical exact window from disk, so compute the
        # window whenever the window OR the RIC correction is requested.
        compute_window = with_window or with_ric
        stage1_stats = ['mesh2_spectrum'] + (['window_mesh2_spectrum'] if compute_window else [])
        for region in regions:
            compute_stats_from_options(
                stage1_stats, analysis=analysis, cache=cache,
                get_catalog_fn=get_catalog_fn,
                get_stats_fn=get_stats_fn,
                prepare_catalog=prepare_catalog,
                catalog=dict(version=fnl_version, tracer=tracer, region=region,
                             zrange=zranges, nran=nran_scalar, weight=weight, FKP_P0=None),
            )
        # combine NGC + SGC -> GCcomb (spectrum and, if computed, exact window)
        postprocess_stats_from_options(
            ['combine_regions'], analysis=analysis, get_stats_fn=get_stats_fn,
            catalog=dict(version=fnl_version, tracer=tracer, zrange=zranges, weight=weight),
            combine_regions={'stats': stage1_stats, 'regions': regions},
        )

        if not with_ric:
            continue

        # --- Stage 2: RIC correction via the forward-model window ---
        # The FM reads the FULL footprint over the tracer's total z-range; both come from the
        # fiducial (handles the tuple-valued cross case for LRGxQSO automatically). ric_regions,
        # ric_nbins, etc. are filled from the fiducial; we only override the run controls below.
        total_region, total_zrange = tools.propose_fiducial('window_mesh2_spectrum_fm', tracer=tracer)['total_region_zrange']
        spectrum_regions_zranges = list(itertools.product(regions, zranges))
        fm_cache = {}
        fm_options = dict(
            geo=True, ric=True, amr=False, ellsout=ellsout,
            n_realizations=n_realizations, seeds=seeds, batch_size=fm_batch_size,
            spectrum_regions_zranges=spectrum_regions_zranges,
        )
        # writes per-seed 'geometry' and 'RIC' windows for each (region, zrange)
        compute_stats_from_options(
            ['window_mesh2_spectrum_fm'], analysis=analysis, cache=fm_cache,
            get_catalog_fn=get_catalog_fn,
            get_stats_fn=get_stats_fn,
            prepare_catalog=prepare_catalog,
            catalog=dict(version=fnl_version, tracer=tracer, region=total_region,
                         zrange=total_zrange, nran=fm_nran, weight=weight, FKP_P0=None,
                         keep_columns=['RA', 'DEC', 'Z', 'POSITION', 'NX', 'TARGETID', 'WEIGHT_FKP']),
            window_mesh2_spectrum_fm=fm_options,
        )

        # combine seeds: RIC-corrected window = analytical exact window + mean(RIC - geometry), per region
        for region in regions:
            postprocess_stats_from_options(
                ['combine_window_mesh2_spectrum'], analysis=analysis, get_stats_fn=get_stats_fn,
                catalog=dict(version=fnl_version, tracer=tracer, region=region, zrange=zranges, weight=weight),
                window_mesh2_spectrum_fm=dict(ellsout=ellsout, n_realizations=n_realizations, seeds=seeds),
                combine_window_mesh2_spectrum={'effect': 'RIC'},
            )
        # combine NGC + SGC -> GCcomb for the RIC-corrected window (tagged extra='RIC')
        postprocess_stats_from_options(
            ['combine_regions'], analysis=analysis, get_stats_fn=get_stats_fn, extra='RIC',
            catalog=dict(version=fnl_version, tracer=tracer, zrange=zranges, weight=weight),
            combine_regions={'stats': ['window_mesh2_spectrum'], 'regions': regions},
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