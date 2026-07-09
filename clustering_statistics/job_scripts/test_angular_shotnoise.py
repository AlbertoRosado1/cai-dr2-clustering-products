#!/usr/bin/env python
"""
Test the impact on the power spectrum shot noise of angular (imaging systematic) weights
applied to both data and randoms.

First step: compute the power spectrum of the DA2 fNL ELG catalogs with imaging systematic
weights WEIGHT_IMLIN and WEIGHT_IMLIN_DES applied to the data; WEIGHT_IMLIN_DES is also
reassigned to the randoms via TARGETID_DATA match with the data, and folded into the
randoms WEIGHT.

Run with e.g.:
```bash
salloc -N 1 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
srun -n 4 python test_angular_shotnoise.py
```
"""
import os
import logging
from pathlib import Path

import numpy as np
from mpi4py import MPI

from clustering_statistics import tools
from clustering_statistics.tools import Catalog, desi_dir, default_mpicomm, setup_logging
from clustering_statistics.spectrum2_tools import compute_mesh2_spectrum


logger = logging.getLogger('angular_shotnoise')


cat_dir = desi_dir / 'survey/catalogs/DA2/LSS/loa-v1/LSScats/v2/fNL'
stats_dir = Path(os.getenv('SCRATCH', '.')) / 'measurements' / 'angular_shotnoise'


def get_catalog_fn(kind='data', iran=0):
    """Return fNL clustering catalog filename."""
    if kind == 'data':
        return cat_dir / 'ELG_LOPnotqso_SGC_clustering.dat.fits'
    return cat_dir / f'ELGnotqso_SGC_{iran:d}_clustering.ran.fits'


def match_data_column(randoms, data, column):
    """Return data ``column`` values matched to ``randoms`` via TARGETID_DATA to TARGETID match."""
    sorted_index = np.argsort(data['TARGETID'])
    index_in_sorted = np.searchsorted(data['TARGETID'][sorted_index], randoms['TARGETID_DATA'])
    index = sorted_index[np.clip(index_in_sorted, 0, len(sorted_index) - 1)]
    if not np.all(data['TARGETID'][index] == randoms['TARGETID_DATA']):
        raise ValueError(f'some randoms TARGETID_DATA are not found in data TARGETID (matching {column})')
    return data[column][index]


@default_mpicomm
def compute_spectrum_imlin(nran=15, zrange=(0.8, 1.6), region=None,
                           data_weights='WEIGHT_IMLIN_DES',
                           randoms_weights='WEIGHT_IMLIN_DES',
                           FKP_P0=2e4, ells=(0, 2), edges=None,
                           mattrs=None, mpicomm=None):
    r"""
    Compute the power spectrum of the fNL ELG catalogs with imaging systematic weights.

    Data individual weight is WEIGHT times the ``data_weights`` columns.
    The ``randoms_weights`` columns are reassigned from the data to the randoms via
    TARGETID_DATA match, and folded into the randoms WEIGHT.

    Parameters
    ----------
    nran : int
        Number of random catalogs to concatenate.
    zrange : tuple
        Redshift range.
    region : str, optional
        Sky region passed to :func:`tools.mask_catalog` (e.g. 'DES', 'SnoDES');
        catalogs are already SGC.
    data_weights : tuple
        Data columns to multiply into the data individual weight, on top of WEIGHT.
    randoms_weights : tuple
        Data columns to reassign to the randoms (TARGETID_DATA match) and fold into WEIGHT.
    FKP_P0 : float, optional
        If not ``None``, multiply individual weights by 1 / (1 + NX * FKP_P0).
    ells : tuple
        Multipole moments to compute.
    edges : dict, optional
        :math:`k`-binning; default step of :math:`0.001 h/\mathrm{Mpc}`.
    mattrs : dict, optional
        Mesh attributes; default is the PNG fiducial (meshsize=700, cellsize=20).

    Returns
    -------
    spectrum : Mesh2SpectrumPoles
    """
    if mattrs is None: mattrs = dict(meshsize=700, cellsize=20.)

    data = randoms = None
    if mpicomm.rank == 0:  # faster to read and process catalogs from one rank
        data = tools._read_catalog(get_catalog_fn(kind='data'), mpicomm=MPI.COMM_SELF)
        randoms = tools._read_catalog([get_catalog_fn(kind='randoms', iran=iran) for iran in range(nran)], mpicomm=MPI.COMM_SELF)

        for catalog in (data, randoms):
            for column in catalog:
                if np.issubdtype(catalog[column].dtype, np.floating):
                    catalog[column] = catalog[column].astype('f8')  # native endianness for jax

        # Reassign data weights to the randoms via TARGETID_DATA match, folding into randoms WEIGHT
        for column in randoms_weights:
            randoms['WEIGHT'] = randoms['WEIGHT'] * match_data_column(randoms, data, column)

        individual_weight = data['WEIGHT'].copy()
        individual_weight = data[data_weights] / data['WEIGHT_SYS']
        data['INDWEIGHT'] = individual_weight
        randoms['INDWEIGHT'] = randoms['WEIGHT'].copy()

        if FKP_P0 is not None:
            logger.info(f'Multiplying individual weights by FKP weight computed with FKP_P0 = {FKP_P0}')
            for catalog in (data, randoms):
                catalog['INDWEIGHT'] = catalog['INDWEIGHT'] / (1. + catalog['NX'] * FKP_P0)

        keep_columns = ['RA', 'DEC', 'Z', 'NX', 'TARGETID', 'INDWEIGHT']
        data = data[keep_columns]
        randoms = randoms[keep_columns]

    if mpicomm.size > 1:
        data = Catalog.scatter(data, mpicomm=mpicomm, mpiroot=0)
        randoms = Catalog.scatter(randoms, mpicomm=mpicomm, mpiroot=0)

    def get_data_randoms():
        toret = {}
        for kind, catalog in [('data', data), ('randoms', randoms)]:
            catalog = tools.mask_catalog(catalog, kind, zrange=zrange, region=region)
            catalog = tools.set_positions_from_rdz(catalog)
            toret[kind] = catalog[['POSITION', 'INDWEIGHT', 'TARGETID']]
        return toret

    return compute_mesh2_spectrum(get_data_randoms, ells=ells, edges=edges, mattrs=mattrs)


if __name__ == '__main__':

    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
    import jax
    jax.config.update('jax_enable_x64', True)
    try: jax.distributed.initialize()
    except RuntimeError: pass
    setup_logging()

    spectrum = compute_spectrum_imlin()
    tools.mkdir(stats_dir)
    tools.write_stats(stats_dir / 'mesh2_spectrum_ELG_LOPnotqso_SGC_imlin.h5', spectrum, mpicomm=MPI.COMM_WORLD)
