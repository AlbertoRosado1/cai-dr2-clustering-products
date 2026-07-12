#!/usr/bin/env python
"""
Compute power spectrum, bispectrum and windows of the Abacus-HF DR2 v2 altmtl mocks,
contaminated on-the-fly with the data imaging systematic pattern.

The mean imaging systematic weight (imweight = WEIGHT_SYS, WEIGHT_IMLIN or
WEIGHT_IMLIN_DES) is computed in each healpix pixel (nside=256, nest=True) from the
DA2 fNL ELG catalogs (the same catalogs as in test_angular_systematics_ELG.py), and
stamped on the mock data as WEIGHT_CONT = 1 / map, with WEIGHT_CONTCONST the mean
WEIGHT_CONT in each N, SnoDES, DES region. The data individual weight is
WEIGHT * WEIGHT_CONT / WEIGHT_SYS (weight='default-FKP-wsys-CONT', or '-wsys-CONTCONST').

The WEIGHT_CONT / WEIGHT_CONTCONST columns are propagated to the randoms from the
matched data object: TARGETID_DATA match (read_catalog 'expand' mechanism) for the
standard altmtl clustering catalogs, or along the redshift assignment of
reshuffle_randoms for on-the-fly complete / altmtl catalogs (``complete`` option, as
onthefly='complete' / 'altmtl' in validation_fiber_assignment.py). The randoms / data
are then renormalized in each N, SnoDES, DES region.

Run with e.g.:
```bash
salloc -N 1 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
srun -n 4 python compute_contaminated_stats.py
```
"""
import os
import functools
import logging
from pathlib import Path

import numpy as np
import healpy as hp
from mpi4py import MPI

from clustering_statistics import tools, setup_logging, compute_stats_from_options, fill_fiducial_options
from clustering_statistics.tools import Catalog, desi_dir


logger = logging.getLogger('contaminated_stats')


version = 'abacus-hf-dr2-v2-altmtl'
data_cat_dir = desi_dir / 'survey/catalogs/DA2/LSS/loa-v1/LSScats/v2/fNL'
stats_dir = Path(os.getenv('SCRATCH', '.')) / 'measurements'
project = 'full_shape/angular_systematics'
norm_regions = ('N', 'SnoDES', 'DES')
nside = 256


def get_stats_fn(*args, extra='', imweight='', onthefly=None, **kwargs):
    extra = '_'.join(txt for txt in [extra, imweight, onthefly] if txt)
    return tools.get_stats_fn(*args, extra=extra, **kwargs)


def compute_weight_sys_map(tracer='ELG_LOPnotqso', imweight='WEIGHT_SYS', nside=nside):
    """
    Return the full-sky healpix map (nside, nest=True) of the mean ``imweight``
    ('WEIGHT_SYS', 'WEIGHT_IMLIN' or 'WEIGHT_IMLIN_DES') of the fNL data catalogs
    (NGC + SGC); pixels with no data are set to 1.
    """
    fns = [data_cat_dir / f'{tracer}_{region}_clustering.dat.fits' for region in ['NGC', 'SGC']]
    data = tools._read_catalog(fns, mpicomm=MPI.COMM_SELF)
    pix = hp.ang2pix(nside, data['RA'], data['DEC'], nest=True, lonlat=True)
    npix = hp.nside2npix(nside)
    counts = np.bincount(pix, minlength=npix)
    wsum = np.bincount(pix, weights=data[imweight], minlength=npix)
    wsys_map = np.divide(wsum, counts, out=np.ones(npix, dtype='f8'), where=counts > 0)
    logger.info(f'{imweight} map from {counts.sum():d} objects in {(counts > 0).sum():d} pixels; '
                f'mean = {data[imweight].mean():.4f}, pixel range = [{wsys_map[counts > 0].min():.4f}, {wsys_map[counts > 0].max():.4f}]')
    return wsys_map


_wsys_maps = {}

def get_weight_sys_map(tracer='ELG_LOPnotqso', imweight='WEIGHT_SYS', mpicomm=MPI.COMM_WORLD):
    """Compute the healpix map on rank 0 and broadcast it; cache per (tracer, imweight)."""
    key = (tracer, imweight)
    if key not in _wsys_maps:
        wsys_map = None
        if mpicomm.rank == 0:
            wsys_map = compute_weight_sys_map(tracer=tracer, imweight=imweight)
        _wsys_maps[key] = mpicomm.bcast(wsys_map, root=0)
    return _wsys_maps[key]


def add_cont_columns(catalog, wsys_map):
    """Stamp WEIGHT_CONT (from the healpix map) and WEIGHT_CONTCONST (mean per N, SnoDES, DES region) on ``catalog``."""
    pix = hp.ang2pix(nside, np.asarray(catalog['RA']), np.asarray(catalog['DEC']), nest=True, lonlat=True)
    catalog['WEIGHT_CONT'] = 1. / wsys_map[pix]
    """
    catalog['WEIGHT_CONTCONST'] = catalog.ones()
    for photo_region in norm_regions:
        mask = tools.select_region(catalog['RA'], catalog['DEC'], region=photo_region)
        csize = catalog.mpicomm.allreduce(mask.sum())
        if csize > 0:
            catalog['WEIGHT_CONTCONST'][mask] = catalog['WEIGHT_CONT'][mask].csum() / csize
    """
    return catalog


def run_stats(stats=('mesh2_spectrum', 'mesh3_spectrum'), imweight='WEIGHT_SYS', contweight='CONT',
              region='NGC', imocks=(0,), tracer='ELG_LOPnotqso', zranges=((1.1, 1.6),), analysis='full_shape',
              complete=None):
    """
    Compute ``stats`` for the mocks contaminated with given ``imweight``
    ('WEIGHT_SYS', 'WEIGHT_IMLIN' or 'WEIGHT_IMLIN_DES'), applying
    WEIGHT * ``contweight`` column / WEIGHT_SYS ('CONT' or 'CONTCONST') to data and randoms.
    If ``complete`` is a dict (e.g. {} for complete, {'altmtl': True} for altmtl on-the-fly),
    catalogs are created on-the-fly.
    """
    zranges = [tuple(zrange) for zrange in zranges]
    cache = {}
    wsys_map = get_weight_sys_map(tracer=tracer, imweight=imweight)

    def read_catalog(kind=None, expand=None, reshuffle=None, **kwargs):
        if kind == 'randoms' and complete is None and isinstance(expand, dict) and not isinstance(expand.get('data_fn', None), Catalog):
            # Contaminate the data catalog used for the TARGETID_DATA match to the randoms
            fns = tools.get_catalog_fn(kind='data', **(kwargs | dict(region='ALL')))
            expand['data_fn'] = add_cont_columns(tools._read_catalog(fns, mpicomm=MPI.COMM_SELF), wsys_map)
        catalog = tools.read_catalog(kind=kind, expand=expand, reshuffle=reshuffle, **kwargs)
        if kind == 'data':
            add_cont_columns(catalog, wsys_map)
            # Contaminate the on-the-fly data; reshuffle_randoms then propagates the columns
            # to the randoms along the redshift assignment (reshuffle['from_data'])
            if isinstance(reshuffle, dict) and isinstance(reshuffle.get('data_fn', None), Catalog):
                add_cont_columns(reshuffle['data_fn'], wsys_map)
        return catalog

    state = {}

    def prepare_catalog(catalogs, kind=None, **kwargs):
        catalogs = tools.prepare_catalog(catalogs, kind=kind, **kwargs)
        if kind == 'data':
            state['data'] = catalogs
        elif kind == 'randoms':
            for randoms in catalogs:
                tools.renormalize_randoms_over_data(randoms, state['data'], regions=norm_regions)
        return catalogs

    onthefly = None
    if complete is not None:
        onthefly = 'altmtl' if complete.get('altmtl', False) else 'complete'
    _get_stats_fn = functools.partial(get_stats_fn, stats_dir=stats_dir, project=project, imweight=imweight, onthefly=onthefly)

    for imock in imocks:
        options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region,
                                    weight=f'default-FKP-wsys-{contweight}', imock=imock),
                       mesh2_spectrum={'auw': False, 'cut': False}, mesh3_spectrum={'auw': False},
                       window_mesh2_spectrum={'cut': False}, window_mesh3_spectrum={})
        options = fill_fiducial_options(options, analysis=analysis)
        for itracer in options['catalog']:
            options['catalog'][itracer]['zranges'] = zranges  # override fiducial zranges
            options['catalog'][itracer]['expand'] = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=itracer, nran=options['catalog'][itracer]['nran']),
                                                     'from_data': ['Z', 'WEIGHT_SYS', 'WEIGHT_CONT']}
            if complete is not None:
                options['catalog'][itracer]['complete'] = dict(complete)
                options['catalog'][itracer]['reshuffle'] = {'from_data': ['WEIGHT_CONT']}
        compute_stats_from_options(list(stats), analysis=analysis, get_stats_fn=_get_stats_fn,
                                   read_catalog=read_catalog, prepare_catalog=prepare_catalog, cache=cache, **options)


if __name__ == '__main__':
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
    import jax
    jax.config.update('jax_enable_x64', True)
    try: jax.distributed.initialize()
    except RuntimeError: pass
    setup_logging()

    imweights = ['WEIGHT_SYS']
    contweights = ['CONT', 'CONTCONST'][:1]
    complete = None
    #complete = {}  # on-the-fly complete catalogs
    #complete = {'altmtl': True}  # on-the-fly altmtl catalogs
    tracer = 'QSO'
    zranges = [(0.8, 2.1)]
    for imweight in imweights:
        for contweight in contweights:
            for region in ['NGC', 'SGC']:
                run_stats(stats=['mesh2_spectrum', 'mesh3_spectrum'], imweight=imweight, contweight=contweight, region=region, tracer=tracer, zranges=zranges, imocks=range(5), complete=complete)
                #run_stats(stats=['window_mesh2_spectrum', 'window_mesh3_spectrum'], imweight=imweight, contweight=contweight, region=region, imocks=[0], complete=complete)
