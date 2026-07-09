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
from clustering_statistics.tools import renormalize_randoms_over_data, Catalog, desi_dir, default_mpicomm, setup_logging
from clustering_statistics.spectrum2_tools import compute_mesh2_spectrum


logger = logging.getLogger('angular_shotnoise')


cat_dir = desi_dir / 'survey/catalogs/DA2/LSS/loa-v1/LSScats/v2/fNL'
print(cat_dir)
stats_dir = Path(os.getenv('SCRATCH', '.')) / 'measurements' / 'angular_shotnoise'


def get_catalog_fn(kind='data', tracer='ELG_LOPnotqso', region='NGC', iran=0):
    """Return fNL clustering catalog filename."""
    if kind == 'data':
        return cat_dir / f'{tracer}_{region}_clustering.dat.fits'
    return cat_dir / f'{tracer}_{region}_{iran:d}_clustering.ran.fits'


@default_mpicomm
def compute_spectrum_imlin(nran=5, zrange=(1.1, 1.6), tracer='ELG_LOPnotqso', region=None,
                           imweights=['WEIGHT_IMLIN'],
                           ells=(0, 2, 4), edges=None,
                           mattrs=None, mpicomm=None):
    r"""
    Compute the power spectrum of the fNL ELG catalogs with imaging systematic weights.

    Data individual weight is WEIGHT times the ``data_weights`` columns.
    The ``randoms_weights`` columns are reassigned from the data to the randoms via
    TARGETID_DATA match, and folded into the randoms WEIGHT.

    Returns
    -------
    spectrum : Mesh2SpectrumPoles
    """
    if mattrs is None: mattrs = dict(meshsize=960, cellsize=7.5)

    data = randoms = None
    regions = ['NGC', 'SGC']

    def match_data_column(randoms, data):
        sorted_index = np.argsort(data['TARGETID'])
        index_in_sorted = np.searchsorted(data['TARGETID'][sorted_index], randoms['TARGETID_DATA'])
        index = sorted_index[np.clip(index_in_sorted, 0, len(sorted_index) - 1)]
        return index

    if mpicomm.rank == 0:  # faster to read and process catalogs from one rank
        data = tools._read_catalog([get_catalog_fn(kind='data', tracer=tracer, region=_region) for _region in regions], mpicomm=MPI.COMM_SELF)
        randoms = tools._read_catalog([get_catalog_fn(kind='randoms', tracer=tracer, region=_region, iran=iran) for _region in regions for iran in range(nran)], mpicomm=MPI.COMM_SELF)

        for catalog in (data, randoms):
            for column in catalog:
                if np.issubdtype(catalog[column].dtype, np.floating):
                    catalog[column] = catalog[column].astype('f8')  # native endianness for jax

        randoms['INDEX'] = match_data_column(randoms, data)

    spectra = []
    for imweight in imweights:
        if mpicomm.rank == 0:
            data['INDWEIGHT'] = data['WEIGHT'] * data[imweight] / data['WEIGHT_SYS'] * data['WEIGHT_FKP']
            randoms['INDWEIGHT'] = randoms['WEIGHT'] * data[imweight][randoms['INDEX']] / randoms['WEIGHT_SYS'] * randoms['WEIGHT_FKP']

        catalogs = {'data': data.deepcopy() if data is not None else None,
                    'randoms': randoms.deepcopy() if randoms is not None else None}
        if mpicomm.size > 1:
            catalogs['data'] = Catalog.scatter(data, mpicomm=mpicomm, mpiroot=0)
            catalogs['randoms'] = Catalog.scatter(randoms, mpicomm=mpicomm, mpiroot=0)
        renormalize_randoms_over_data(catalogs['randoms'], catalogs['data'], tracer=tracer)
        for kind, catalog in catalogs.items():
            catalog = tools.mask_catalog(catalog, kind, zrange=zrange, region=region)
            catalog = tools.set_positions_from_rdz(catalog)
            catalogs[kind] = catalog[['POSITION', 'INDWEIGHT', 'TARGETID']]
        spectra.append(compute_mesh2_spectrum(lambda: catalogs, ells=ells, edges=edges, mattrs=mattrs))

    return spectra


if __name__ == '__main__':
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
    import jax
    jax.config.update('jax_enable_x64', True)
    try: jax.distributed.initialize()
    except RuntimeError: pass
    setup_logging()

    mpicomm = MPI.COMM_WORLD
    tracer = 'ELG_LOPnotqso'
    imweights = ['WEIGHT_SYS', 'WEIGHT_IMLIN', 'WEIGHT_IMLIN_DES']
    for region in ['NGC', 'SGC'][1:]:
        spectra = compute_spectrum_imlin(imweights=imweights, region=region, tracer=tracer, mpicomm=mpicomm)
        for spectrum, imweight in zip(spectra, imweights):
            tools.write_stats(stats_dir / f'mesh2_spectrum_ELG_LOPnotqso_{region}_{imweight}.h5',
                              spectrum, mpicomm=mpicomm)
