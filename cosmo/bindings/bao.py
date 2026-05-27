"""DESI BAO dataset metadata used by cosmology-code bindings."""

import os
from pathlib import Path


DEFAULT_BAO_DATA_PATH = Path('/global/cfs/cdirs/desicollab/science/cpe/y3_bao_cosmo/bao_v1p2/bao/cobaya_data')

BAO_DR2_DATASETS = {
    'desi-bao-all': {
        'likelihood': 'desi_dr2_bao_all',
        'measurements_file': 'desi_gaussian_bao_ALL_GCcomb_mean.txt',
        'cov_file': 'desi_gaussian_bao_ALL_GCcomb_cov.txt',
    },
    'desi-bao-bgs': {
        'likelihood': 'desi_dr2_bao_bgs_z1',
        'measurements_file': 'desi_gaussian_bao_BGS_BRIGHT-21.35_GCcomb_z0.1-0.4_mean.txt',
        'cov_file': 'desi_gaussian_bao_BGS_BRIGHT-21.35_GCcomb_z0.1-0.4_cov.txt',
    },
    'desi-bao-lrg-z1': {
        'likelihood': 'desi_dr2_bao_lrg_z1',
        'measurements_file': 'desi_gaussian_bao_LRG_GCcomb_z0.4-0.6_mean.txt',
        'cov_file': 'desi_gaussian_bao_LRG_GCcomb_z0.4-0.6_cov.txt',
    },
    'desi-bao-lrg-z2': {
        'likelihood': 'desi_dr2_bao_lrg_z2',
        'measurements_file': 'desi_gaussian_bao_LRG_GCcomb_z0.6-0.8_mean.txt',
        'cov_file': 'desi_gaussian_bao_LRG_GCcomb_z0.6-0.8_cov.txt',
    },
    'desi-bao-lrgpluselg': {
        'likelihood': 'desi_dr2_bao_lrgpluselg_z1',
        'measurements_file': 'desi_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_mean.txt',
        'cov_file': 'desi_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_cov.txt',
    },
    'desi-bao-elg': {
        'likelihood': 'desi_dr2_bao_elg_z2',
        'measurements_file': 'desi_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_mean.txt',
        'cov_file': 'desi_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_cov.txt',
    },
    'desi-bao-qso': {
        'likelihood': 'desi_dr2_bao_qso_z1',
        'measurements_file': 'desi_gaussian_bao_QSO_GCcomb_z0.8-2.1_mean.txt',
        'cov_file': 'desi_gaussian_bao_QSO_GCcomb_z0.8-2.1_cov.txt',
    },
    'desi-bao-lya': {
        'likelihood': 'desi_dr2_bao_lya',
        'measurements_file': 'desi_gaussian_bao_Lya_GCcomb_mean.txt',
        'cov_file': 'desi_gaussian_bao_Lya_GCcomb_cov.txt',
    },
}


def make_list(value):
    """Return ``value`` as a list, treating strings as scalars."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def normalize_dataset(dataset, datasets=BAO_DR2_DATASETS):
    """Normalize one or more dataset labels to a list of strings."""
    names = make_list(dataset)
    unknown = [name for name in names if name not in datasets]
    if unknown:
        raise ValueError('Unknown BAO dataset(s): {}. Known datasets are {}.'.format(
            ', '.join(unknown), ', '.join(sorted(datasets))))
    return names


def get_bao_data_path(path=None):
    """Return the directory containing BAO mean/covariance files."""
    if path is None:
        path = os.getenv('DESI_CLUSTERING_COSMO_BAO_DATA_PATH', None)
    if path is None:
        path = DEFAULT_BAO_DATA_PATH
    return Path(path)


def get_dataset_metadata(dataset, datasets=BAO_DR2_DATASETS):
    """Return metadata dictionaries for one or more BAO datasets."""
    return [datasets[name] for name in normalize_dataset(dataset, datasets=datasets)]
