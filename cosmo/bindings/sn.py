"""Supernova likelihood metadata."""

SN_LIKELIHOODS = {
    'desdovekie': {'family': 'sn', 'parameterization': 'background', 'cobaya': 'sn.desdovekie'},
    'pantheon': {'family': 'sn', 'parameterization': 'background', 'cobaya': 'sn.pantheon'},
    'pantheonplus': {'family': 'sn', 'parameterization': 'background', 'cobaya': 'sn.pantheonplus'},
    'union3': {'family': 'sn', 'parameterization': 'background', 'cobaya': 'sn.union3'},
    'desy5sn': {'family': 'sn', 'parameterization': 'background', 'cobaya': 'sn.desy5'},
    'desy5sn-zmin0.0': {'family': 'sn', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.sn.zmin.DESY5Zmin': {'dataset_file': 'DESY5/config.dataset', 'aliases': ['DESY5'], 'use_abs_mag': False, 'speed': 100, 'zmin': 0.0}}},
    'desy5sn-zmin0.05': {'family': 'sn', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.sn.zmin.DESY5Zmin': {'dataset_file': 'DESY5/config.dataset', 'aliases': ['DESY5'], 'use_abs_mag': False, 'speed': 100, 'zmin': 0.05}}},
    'desy5sn-zmin0.1': {'family': 'sn', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.sn.zmin.DESY5Zmin': {'dataset_file': 'DESY5/config.dataset', 'aliases': ['DESY5'], 'use_abs_mag': False, 'speed': 100, 'zmin': 0.1}}},
    'desy5sn-zmin0.2': {'family': 'sn', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.sn.zmin.DESY5Zmin': {'dataset_file': 'DESY5/config.dataset', 'aliases': ['DESY5'], 'use_abs_mag': False, 'speed': 100, 'zmin': 0.2}}},
    'pantheonplus-zmin0.1': {'family': 'sn', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.sn.zmin.PantheonPlusZmin': {'dataset_file': 'PantheonPlus/config.dataset', 'aliases': ['PantheonPlus'], 'use_abs_mag': False, 'speed': 100, 'zmin': 0.1}}},
    'union3-zmin0.1': {'family': 'sn', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.sn.zmin.Union3Zmin': {'dataset_file': 'Union3/full_long.dataset', 'aliases': ['Union3'], 'use_abs_mag': False, 'speed': 100, 'zmin': 0.1}}},
}
