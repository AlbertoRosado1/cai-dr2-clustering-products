import os
import logging
from pathlib import Path

import numpy as np

from mockfactory import Catalog, setup_logging, sky_to_cartesian
import lsstypes as types


logger = logging.getLogger('io')


desi_dir = Path('/dvs_ro/cfs/cdirs/desi/')


def load_footprint():
    #global footprint
    from regressis import footprint
    footprint = footprint.DR9Footprint(256, mask_lmc=False, clear_south=True, mask_around_des=False, cut_desi=False)
    return footprint


def join_tracers(tracers):
    if not isinstance(tracers, str):
        return 'x'.join(tracers)
    return tracers


def get_simple_tracer(tracer):
    if 'BGS' in tracer:
        return 'BGS'
    elif 'LRG+ELG' in tracer:
        return 'LRG+ELG'
    elif 'LRG' in tracer:
        return 'LRG'
    elif 'ELG' in tracer:
        return 'ELG'
    elif 'QSO' in tracer:
        return 'QSO'
    else:
        raise NotImplementedError(f'tracer {tracer} is unknown')


def select_region(ra, dec, region=None):
    import healpy as hp
    # print('select', region)
    if region in [None, 'ALL', 'GCcomb']:
        return np.ones_like(ra, dtype='?')
    mask_ngc = (ra > 100 - dec)
    mask_ngc &= (ra < 280 + dec)
    mask_n = mask_ngc & (dec > 32.375)
    mask_s = (~mask_n) & (dec > -25.)
    if region == 'NGC':
        return mask_ngc
    if region == 'SGC':
        return ~mask_ngc
    if region == 'N':
        return mask_n
    if region == 'S':
        return mask_s
    if region == 'SNGC':
        return mask_ngc & mask_s
    if region == 'SSGC':
        return (~mask_ngc) & mask_s
    if region == 'NGCnoN':
        return mask_ngc & (~mask_n)
    # if region == 'GCcomb_noNorth':
    #     return ~mask_n
    north, south, des = load_footprint().get_imaging_surveys()
    mask_des = des[hp.ang2pix(256, ra, dec, nest=True, lonlat=True)]
    if region == 'DES':
        return mask_des
    if region == 'SnoDES':
        return mask_s & (~mask_des)
    if region == 'SSGCnoDES':
        return (~mask_ngc) & mask_s & (~mask_des)
    if region == 'SGCnoDES':
        return (~mask_ngc) & (~mask_des)
    # if region == 'GCcomb_noDES':
    #     return ~mask_des
    raise ValueError('unknown region {}'.format(region))


def propose_fiducial(kind, tracer):
    cellsize = 7.8
    base = {'correlation': {}, 'mesh2_spectrum': {}, 'mesh3_spectrum': {}}
    propose_fiducial = {
        'BGS': base | {'zranges': [(0.1, 0.4)], 'mattrs': dict(boxsize=4000., cellsize=cellsize), 'nran': 2, 'recon': dict(bias=1.5, smoothing_radius=15.)},
        'LRG+ELG': base | {'zranges': [(0.8, 1.1)], 'mattrs': dict(boxsize=9000., cellsize=cellsize), 'nran': 13, 'recon': dict(bias=1.6, smoothing_radius=15.)},
        'LRG': base | {'zranges': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)], 'mattrs': dict(boxsize=7000., cellsize=cellsize), 'nran': 9, 'recon': dict(bias=2.0, smoothing_radius=15.)},
        'ELG': base | {'zranges': [(0.8, 1.1), (1.1, 1.6)], 'mattrs': dict(boxsize=9000., cellsize=cellsize), 'nran': 13, 'recon': dict(bias=1.2, smoothing_radius=15.)},
        'QSO': base | {'zranges': [(0.8, 2.1)], 'mattrs': dict(boxsize=10000., cellsize=cellsize), 'nran': 4, 'recon': dict(bias=2.1, smoothing_radius=30.)}
    }
    return propose_fiducial[get_simple_tracer(tracer)][kind]


def get_catalog_dir(survey='Y1', verspec='iron', version='v1.2', base_dir=desi_dir / 'survey/catalogs'):
    base_dir = Path(base_dir)
    return base_dir / survey / 'LSS' / verspec / 'LSScats' / version


def get_catalog_fn(version=None, cat_dir=None, kind='data', tracer='LRG',
                   region='NGC', weight_type='default_FKP', nran=10, imock=0, ext='h5', **kwargs):
    if region in ['N', 'NGC', 'NGCnoN']: region = 'NGC'
    elif region in ['SGC', 'SGCnoDES']: region = 'SGC'
    elif 'full' not in kind:
        if region in ['S', 'ALL']: regions = ['NGC', 'SGC']
        else: raise NotImplementedError(f'{region} is unknown')
        return [get_catalog_fn(version=version, cat_dir=cat_dir, kind=kind, tracer=tracer,
                               region=region, weight_type=weight_type, nran=nran, imock=imock, ext=ext, **kwargs) for region in regions]

    if cat_dir is None:  # pre-registered paths
        if version == 'data-dr1-v1.5':
            cat_dir = desi_dir / f'Y1/LSS/iron/LSScats'
            if 'bitwise' in weight_type:
                cat_dir = cat_dir / 'v1.5pip'
            else:
                cat_dir = cat_dir / 'v1.5'
        elif version == 'data-dr2-v2':
            cat_dir = desi_dir / f'DA2/LSS/loa-v1/LSScats/v2'
            if 'bitwise' in weight_type:
                data_dir = cat_dir / 'PIP'
            else:
                data_dir = cat_dir / 'nonKP'
            if kind == 'data':
                return data_dir / f'{tracer}_{region}_clustering.dat.fits'
            if kind == 'randoms':
                return [data_dir / f'{tracer}_{region}_{iran:d}_clustering.ran.fits' for iran in range(nran)]
            if kind == 'full_data':
                return cat_dir / f'{tracer}_full_HPmapcut.dat.fits'
            if kind == 'full_randoms':
                return [cat_dir / f'{tracer}_{iran:d}_full_HPmapcut.ran.fits' for iran in range(nran)]
        elif version == 'holi-v1-complete':
            cat_dir = desi_dir / f'mocks/cai/LSS/DA2/mocks/holi_v1/altmtl{imock:d}/loa-v1/mock{imock:d}/LSScats'
            if kind == 'data':
                return cat_dir / f'{tracer}_complete_{region}_clustering.dat.h5'
            if kind == 'randoms':
                return [cat_dir / f'{tracer}_complete_{region}_{iran:d}_clustering.ran.h5' for iran in range(nran)]
        elif version == 'holi-v1-altmtl':
            cat_dir = desi_dir / f'mocks/cai/LSS/DA2/mocks/holi_v1/altmtl{imock:d}/loa-v1/mock{imock:d}/LSScats'
            ext = 'fits' if 'full' in kind else 'h5'
    cat_dir = Path(cat_dir)
    if kind == 'data':
        return cat_dir / f'{tracer}_{region}_clustering.dat.{ext}'
    if kind == 'randoms':
        return [cat_dir / f'{tracer}_{region}_{iran:d}_clustering.ran.{ext}' for iran in range(nran)]
    if kind == 'full_data':
        return cat_dir / f'{tracer}_full_HPmapcut.dat.{ext}'
    if kind == 'full_randoms':
        return [cat_dir / f'{tracer}_{iran:d}_full_HPmapcut.ran.{ext}' for iran in range(nran)]



def get_measurement_fn(meas_dir=Path(os.getenv('SCRATCH')) / 'measurements', kind='mesh2_spectrum', version='data-dr2-v2', recon=None,
                       tracer='LRG', region='NGC', zrange=(0.8, 1.1), auw=None, cut=None, weight_type='default', imock=None, extra='', ext='h5', **kwargs):
    if imock == '*':
        fns = [get_measurement_fn(meas_dir=meas_dir, kind=kind, version=version, recon=recon, tracer=tracer, region=region, zrange=zrange, auw=auw, cut=cut, weight_type=weight_type, imock=imock, ext=ext, **kwargs) for imock in range(1000)]
        return [fn for fn in fns if os.path.exists(fn)]
    if cut: cut = '_thetacut'
    else: cut = ''
    if auw: auw = '_auw'
    else: auw = ''
    meas_dir = meas_dir / version
    if recon:
        meas_dir = meas_dir / recon
    if imock is None: imock = ''
    else: imock = '_{imock:d}'
    if extra: extra = '_{extra}'
    tracer = join_tracers(tracer)
    basename = f'{kind}_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{region}_{weight_type}{auw}{cut}{extra}{imock}.{ext}'
    return meas_dir / basename


def apply_wntmp(ntile, ntmp_table, method='ntmp'):
    frac_missing_pw, frac_zero_prob = ntmp_table
    if method == 'ntmp':
        toret = 1 - frac_missing_pw[ntile]
    elif method == 'ntzp':
        toret = 1 - frac_zero_prob[ntile]
    else:
        raise NotImplementedError(f'unknown method {method}')
    #ref = apply_wntmp_bak(ntile, frac_missing_pw, frac_zero_prob, ntile_range=[0,15], randoms=True)[0]
    #assert np.allclose(toret, ref)
    return toret

# Create a lookup table for set bits per byte
_popcount_lookuptable = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)


def popcount(*arrays):
    """
    Return number of 1 bits in each value of input array.
    Inspired from https://github.com/numpy/numpy/issues/16325.
    """
    # if not np.issubdtype(array.dtype, np.unsignedinteger):
    #     raise ValueError('input array must be an unsigned int dtype')
    toret = _popcount_lookuptable[arrays[0].view((np.uint8, (arrays[0].dtype.itemsize,)))].sum(axis=-1)
    for array in arrays[1:]: toret += popcount(array)
    return toret


def _compute_ntmp(bitweights, loc_assigned, ntile):
    """
    nbits = 64 * np.shape(bitweights)[1]
    recurr = prob_obs * nbits
    """
    # Input: list of bitweights
    nbits = 8 * sum(weight.dtype.itemsize for weight in bitweights)
    recurr = popcount(*bitweights)
    wiip = (nbits + 1) / (recurr + 1)
    zero_prob = (recurr == 0) & (~loc_assigned)

    #print(np.sum(zerop_msk))
    sum_ntile = np.bincount(ntile)
    sum_zero_prob = np.bincount(ntile, weights=zero_prob)
    sum_loc_assigned = np.bincount(ntile, weights=loc_assigned)
    sum_wiip = np.bincount(ntile, weights=loc_assigned * wiip)
    mask_zero_ntile = sum_ntile == 0
    frac_zero_prob = np.divide(sum_zero_prob, sum_ntile, out=np.ones_like(sum_wiip), where=~mask_zero_ntile)
    frac_missing_pw = np.divide(sum_ntile - sum_wiip, sum_ntile, out=np.ones_like(sum_wiip), where=~mask_zero_ntile)
    return frac_missing_pw, frac_zero_prob


def _read_catalog(fn, mpicomm=None):
    one_fn = fn[0] if isinstance(fn, (tuple, list)) else fn
    kw = {}
    if str(one_fn).endswith('.h5'): kw['group'] = 'LSS'
    catalog = Catalog.read(fn, mpicomm=mpicomm, **kw)
    if str(one_fn).endswith('.fits'): catalog.get(catalog.columns())  # Faster to read all columns at once
    return catalog


def compute_ntmp(full_data_fn):
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    ntmp = None
    if mpicomm.rank == 0:
        catalog = _read_catalog(full_data_fn, mpicomm=MPI.COMM_SELF)
        ntmp = _compute_ntmp(_format_bitweights(catalog['BITWEIGHTS']), catalog['LOCATION_ASSIGNED'], catalog['NTILE'])
    return mpicomm.bcast(ntmp, root=0)


def _compute_wntile(ntile, weight):
    sum_ntile = np.bincount(ntile)
    sum_weight = np.bincount(ntile, weights=weight)
    mask_zero_ntile = sum_ntile == 0
    return np.divide(sum_weight, sum_ntile, out=np.ones_like(sum_weight), where=~mask_zero_ntile)


def compute_wntile(clustering_data_fn):
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    wntile = None
    if mpicomm.rank == 0:
        catalog = _read_catalog(clustering_data_fn, mpicomm=MPI.COMM_SELF)
        wntile = _compute_wntile(catalog['NTILE'], catalog['WEIGHT'] / catalog['WEIGHT_COMP'])
    return mpicomm.bcast(wntile, root=0)


def apply_wntile(ntile, wntile_table):
    return wntile_table[ntile]


def _format_bitweights(bitweights):
    if bitweights.ndim == 2: return list(bitweights.T)
    return [bitweights]


def read_clustering_rdzw(*fns, kind=None, zrange=None, region=None, weight_type='default', ntmp=None, concatenate=True, **kwargs):
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD

    catalogs = [None] * len(fns)
    for ifn, fn in enumerate(fns):
        irank = ifn % mpicomm.size
        catalogs[ifn] = (irank, None)
        if mpicomm.rank == irank:  # Faster to read catalogs from one rank
            catalog = _read_catalog(catalog, mpicomm=MPI.COMM_SELF)
            columns = ['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_SYS', 'WEIGHT_ZFAIL', 'WEIGHT_COMP', 'WEIGHT_FKP', 'BITWEIGHTS', 'FRAC_TLOBS_TILES', 'NTILE']
            columns = [col for col in columns if col in catalog.columns()]
            catalog = catalog[columns]
            if zrange is not None:
                mask = (catalog['Z'] >= zrange[0]) & (catalog['Z'] <= zrange[1])
                catalog = catalog[mask]
            if 'bitwise' in weight_type:
                mask = (catalog['FRAC_TLOBS_TILES'] != 0)
                catalog = catalog[mask]
            if region is not None:
                mask = select_region(catalog['RA'], catalog['DEC'], region)
                catalog = catalog[mask]
            catalogs[ifn] = (irank, catalog)

    rdzw = []
    for irank, catalog in catalogs:
        if mpicomm.size > 1:
            catalog = Catalog.scatter(catalog, mpicomm=mpicomm, mpiroot=irank)
        individual_weight = catalog['WEIGHT']
        bitwise_weights = []
        if 'bitwise' in weight_type:
            if kind == 'data':
                individual_weight = catalog['WEIGHT'] / catalog['WEIGHT_COMP']
                bitwise_weights = _format_bitweights(catalog['BITWEIGHTS'])
            elif kind == 'randoms' and ntmp is not None:
                individual_weight = catalog['WEIGHT'] * apply_wntmp(catalog['NTILE'], ntmp)
        if 'FKP' in weight_type.upper():
            individual_weight *= catalog['WEIGHT_FKP']
        _rdzw = [catalog['RA'], catalog['DEC'], catalog['Z'], individual_weight] + bitwise_weights
        for i in range(4): _rdzw[i] = _rdzw[i].astype('f8')
        rdzw.append(_rdzw)
    if concatenate:
        rdzw = [np.concatenate([arrays[i] for arrays in rdzw], axis=0) for i in range(len(rdzw[0]))]
    else:
        return rdzw


def read_full_rdw(*fns, kind='parent', region=None, weight_type='default', ntmp=None, wntile=None, **kwargs):

    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD

    catalogs = [None] * len(fns)
    for ifn, fn in enumerate(fns):
        irank = ifn % mpicomm.size
        catalogs[ifn] = (irank, None)
        if mpicomm.rank == irank:  # Faster to read catalogs from one rank
            catalog = Catalog.read(fn, mpicomm=MPI.COMM_SELF)
            catalog.get(catalog.columns())  # Faster to read all columns at once
            columns = ['RA', 'DEC', 'LOCATION_ASSIGNED', 'BITWEIGHTS', 'NTILE', 'WEIGHT_NTILE', 'FRACZ_TILELOCID', 'FRAC_TLOBS_TILES']
            columns = [col for col in columns if col in catalog.columns()]
            catalog = catalog[columns]
            if 'fibered' in kind:
                mask = catalog['LOCATION_ASSIGNED']
                catalog = catalog[mask]
            if region is not None:
                mask = select_region(catalog['RA'], catalog['DEC'], region)
                catalog = catalog[mask]
            catalogs[ifn] = (irank, catalog)

    rdzw = []
    for irank, catalog in catalogs:
        if mpicomm.size > 1:
            catalog = Catalog.scatter(catalog, mpicomm=mpicomm, mpiroot=irank)
        individual_weight = catalog['WEIGHT_NTILE']
        if wntile is not None:
            individual_weight = apply_wntile(catalog['NTILE'], wntile)
            assert np.allclose(individual_weight, catalog['WEIGHT_NTILE'])
        bitwise_weights = []
        if 'fibered' in kind and 'data' in kind:
            if ntmp is not None:
                individual_weight /= apply_wntmp(catalog['NTILE'], ntmp)
            bitwise_weights = _format_bitweights(catalog['BITWEIGHTS'])
            if 'bitwise' not in weight_type:
                individual_weight /= (catalog['FRACZ_TILELOCID'] * catalog['FRAC_TLOBS_TILES'])
                bitwise_weights = []
        rdzw.append([catalog['RA'], catalog['DEC'], individual_weight] + bitwise_weights)
    rdzw = [np.concatenate([arrays[i] for arrays in rdzw], axis=0) for i in range(len(rdzw[0]))]
    for i in range(3): rdzw[i] = rdzw[i].astype('f8')
    return rdzw[:2], rdzw[2:]


def get_positions_weights_from_rdzw(rdzw):
    from cosmoprimo.fiducial import TabulatedDESI, DESI
    fiducial = TabulatedDESI()  # faster than DESI/class (which takes ~30 s for 10 random catalogs)

    def _get_clustering_positions_weights(ra, dec, z, weights):
        weights = np.asarray(weights, dtype='f8')
        dist = fiducial.comoving_radial_distance(z)
        positions = sky_to_cartesian(dist, ra, dec, dtype='f8')
        return positions, weights

    if isinstance(rdzw[0], (tuple, list)):  # list of (RA, DEC, Z, W)
        return [_get_clustering_positions_weights(*rdzw) for rdzw in rdzw]
    else:
        return _get_clustering_positions_weights(*rdzw)


def write_summary_statistics(fn, stats):
    import jax
    if jax.process_index() == 0:
        logger.info(f'Writing to {fn}')
        stats.write(fn)


def possible_combine_regions(regions):
    """Return potential combinations."""
    regions = sorted(regions)
    region_combs = {'GCcomb': ['NGC', 'SGC'],
                    'NS': ['N', 'S'],
                    'GCcomb_noN': ['NGCnoN', 'SGC'],
                    'GCcomb_noDES': ['NGC', 'SGCnoDES']}
    combs = {}
    for _region_comb, _regions in region_combs.items():
        if all(region in _regions for region in regions):
            combs[_region_comb] = _regions
    return combs