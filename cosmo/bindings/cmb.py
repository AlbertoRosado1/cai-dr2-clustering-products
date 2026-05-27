"""CMB, compressed CMB, and CMB-lensing likelihood metadata."""


def _spt_candl_dataset_file():
    """Return the installed SPT-3G candl dataset index path."""
    from pathlib import Path
    import spt_candl_data
    return str(Path(spt_candl_data.__file__).parent / 'SPT3G_D1_TnE_v0' / 'SPT3G_D1_TnE_index.yaml')


def _candl_likelihood_class():
    """Return candl's Cobaya likelihood class."""
    from candl.interface import CandlCobayaLikelihood
    return CandlCobayaLikelihood


def _cmb_spa_cobaya():
    """Return the CMB-SPA composite likelihood block."""
    return {
        'planck_2018_lowl.TT': None,
        'planck_2018_lowl.EE_sroll2': None,
        'act_dr6_cmbonly.PlanckActCut': {
            'dataset_params': {
                'use_cl': 'tt te ee',
                'lmin_cuts': '0 0 0',
                'lmax_cuts': '1000 600 600',
            },
            'params': {
                'A_planck': {
                    'value': 'lambda A_act: A_act',
                    'latex': r'A_{\rm planck}',
                    'proposal': 0.003,
                },
            },
        },
        'act_dr6_cmbonly.ACTDR6CMBonly': {
            'stop_at_error': True,
            'ell_cuts': {'EE': [600, 8500], 'TE': [600, 8500], 'TT': [600, 8500]},
            'input_file': 'dr6_data_cmbonly.fits',
            'lmax_theory': 9500,
        },
        'candl_like': {
            'external': _candl_likelihood_class(),
            'data_set_file': _spt_candl_dataset_file(),
            'clear_internal_priors': True,
            'variant': 'lite',
            'feedback': True,
            'wrapper': None,
            'additional_args': {},
        },
        'act_dr6_spt_lenslike.ACTDR6LensLike': {
            'lens_only': False,
            'variant': 'actplanckspt3g_baseline',
            'lmax': 4000,
            'version': 'v1.2',
        },
    }


CMB_SPA_PRIORS = {
    'cal_dip_prior': 'lambda A_act: stats.norm.logpdf(A_act, loc = 1.0, scale = 0.003)',
    'gaussian_Tcal': 'lambda Tcal: stats.norm.logpdf(Tcal, loc=1.0, scale=0.0036)',
}


CMB_LIKELIHOODS = {
    'CMB-SPA': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': _cmb_spa_cobaya(), 'prior': CMB_SPA_PRIORS},
    'CMB-SPA-tauprior': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': _cmb_spa_cobaya(), 'prior': CMB_SPA_PRIORS, 'tauprior': True},
    # Planck 2018 theta*/rdrag priors.
    'planck2018-thetastar-fixed-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.thetastar_planck2018_fixed_nnu': None}},
    'planck2018-thetastar-varied-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.thetastar_planck2018_varied_nnu': None}},
    'planck2018-thetastar-fixed-marg-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.thetastar_planck2018_fixed_marg_nnu': None}},
    'planck2018-rdrag-fixed-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.rdrag_planck2018_fixed_nnu': None}},

    # Standard external Cobaya CMB likelihood package names.
    'planck2018-lowl-TT-clik': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_lowl.TT_clik'},
    'planck2018-lowl-EE-clik': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_lowl.EE_clik'},
    'planck2018-lowl-TT': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_lowl.TT'},
    'planck2018-lowl-EE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_lowl.EE'},
    'planck2018-lowl-EE-sroll2': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_lowl.EE_sroll2'},
    'planck2018-highl-plik-TT': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_highl_plik.TT'},
    'planck2018-highl-plik-TTTEEE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_highl_plik.TTTEEE'},
    'planck2018-highl-plik-TTTEEE-lite': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_highl_plik.TTTEEE_lite'},
    'planck2018-highl-CamSpec-TT-clik': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_highl_CamSpec.TT'},
    'planck2018-highl-CamSpec-TTTEEE-clik': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_highl_CamSpec.TTTEEE'},
    'planck2018-highl-CamSpec-TT': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_highl_CamSpec.TT_native'},
    'planck2018-highl-CamSpec-TTTEEE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_highl_CamSpec.TTTEEE_native'},
    'planck2018-highl-CamSpec2021-TT': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_highl_CamSpec2021.TT'},
    'planck2018-highl-CamSpec2021-TTTEEE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_highl_CamSpec2021.TTTEEE'},
    'planck-NPIPE-highl-CamSpec-TT': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_NPIPE_highl_CamSpec.TT'},
    'planck-NPIPE-highl-CamSpec-TE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_NPIPE_highl_CamSpec.TE'},
    'planck-NPIPE-highl-CamSpec-EE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_NPIPE_highl_CamSpec.EE'},
    'planck-NPIPE-highl-CamSpec-TTTEEE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.planck_npipe.TTTEEENoCache': None}},
    'planck-NPIPE-highl-CamSpec-TTTEEE-ell-max-600': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.planck_npipe.TTTEEENoCache': {'dataset_file': '/global/cfs/cdirs/desicollab/science/cpe/cmbdata/CamSpec_NPIPE/CamSpec_NPIPE_12_6_cl.dataset', 'dataset_params': {'use_cl': '143x143 217x217 143x217 TE EE', 'use_range': '30-600'}, 'aliases': ['CamSpec_NPIPE_TTTEEE']}}},
    'planck-NPIPE-highl-CamSpec-TTTEEE-cuts-for-act': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.planck_npipe.TTTEEENoCache': {'dataset_file': '/global/cfs/cdirs/desicollab/science/cpe/cmbdata/CamSpec_NPIPE/CamSpec_NPIPE_12_6_cl.dataset', 'dataset_params': {'use_cl': '143x143 217x217 143x217 TE EE', 'use_range': {'143x143': '30-2000', '217x217': '500-2000', '143x217': '500-2000', 'TE': '30-1000', 'EE': '30-1000'}}, 'aliases': ['CamSpec_NPIPE_TTTEEE']}}},
    'planck2018-lensing-clik': {'family': 'cmb_lensing', 'parameterization': 'cmb', 'cobaya': 'planck_2018_lensing.clik'},
    'planck2018-lensing': {'family': 'cmb_lensing', 'parameterization': 'cmb', 'cobaya': 'planck_2018_lensing.native'},
    'planckpr4lensing': {'family': 'cmb_lensing', 'parameterization': 'cmb', 'cobaya': 'planckpr4lensing.PlanckPR4Lensing'},
    'planckpr4lensingmarged': {'family': 'cmb_lensing', 'parameterization': 'cmb', 'cobaya': 'planckpr4lensing.PlanckPR4LensingMarged'},
    'planck2020-lollipop-lowlE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2020_lollipop.lowlE'},
    'planck2020-lollipop-lowlB': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2020_lollipop.lowlB'},
    'planck2020-lollipop-lowlEB': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2020_lollipop.lowlEB'},
    'planck2020-hillipop-TT': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2020_hillipop.TT'},
    'planck2020-hillipop-TE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2020_hillipop.TE'},
    'planck2020-hillipop-EE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2020_hillipop.EE'},
    'planck2020-hillipop-TTTEEE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2020_hillipop.TTTEEE'},
    'wmap-TTTEEE': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': {'wmaplike.WMAPLike': {'temin': 24, 'params': {'A_sz': {'prior': {'min': 0.0, 'max': 2.0}}}}}},
    'act-dr6-lensing': {'family': 'cmb_lensing', 'parameterization': 'cmb', 'cobaya': {'act_dr6_lenslike_v1_2.ACTDR6LensLike': {'lens_only': False, 'variant': 'act_baseline', 'lmax': 4000, 'version': 'v1.2'}}},
    'planck-act-dr6-lensing': {'family': 'cmb_lensing', 'parameterization': 'cmb', 'cobaya': {'act_dr6_lenslike_v1_2.ACTDR6LensLike': {'lens_only': False, 'variant': 'actplanck_baseline', 'lmax': 4000, 'version': 'v1.2'}}},
    'planck2018-lowl-TTTEEE-sroll2-momento': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.momento.TTTEEE_SROLL20': None}},
    'planck2018-lowl-TT-11-29-clik': {'family': 'cmb', 'parameterization': 'cmb', 'cobaya': 'planck_2018_lowl.TT_clik'},

    # Compressed CMB likelihoods.
    'CMB-compressed-theta-ombh2-ombch2-marg-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.CMB_standard_compression_PR4': {'observables': ['thetastar', 'ombh2', 'ombch2'], 'inflate_cov': True}}},
    'CMB-compressed-ombh2-ombch2-marg-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.CMB_standard_compression_PR4': {'observables': ['ombh2', 'ombch2'], 'inflate_cov': True}}},
    'CMB-compressed-theta-ombh2-marg-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.CMB_standard_compression_PR4': {'observables': ['thetastar', 'ombh2'], 'inflate_cov': True}}},
    'CMB-compressed-theta-ombh2-ombch2': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.CMB_standard_compression_PR4': {'observables': ['thetastar', 'ombh2', 'ombch2'], 'inflate_cov': False}}},
    'CMB-compressed-ombh2-ombch2': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.CMB_standard_compression_PR4': {'observables': ['ombh2', 'ombch2'], 'inflate_cov': False}}},
    'CMB-compressed-theta-ombh2': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.CMB_standard_compression_PR4': {'observables': ['thetastar', 'ombh2'], 'inflate_cov': False}}},
    'CMB-compressed-theta': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.CMB_standard_compression_PR4': {'observables': ['thetastar'], 'inflate_cov': False}}},
    'CMB-compressed-ombh2': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.CMB_standard_compression_PR4': {'observables': ['ombh2'], 'inflate_cov': False}}},
    'CMB-compressed-ombch2': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.CMB_standard_compression_PR4': {'observables': ['ombch2'], 'inflate_cov': False}}},
    'CMB-compressed-fake-theta-ombh2-ombch2': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.CMB_standard_compression_PR4_DESI_omch2': {'observables': ['thetastar', 'ombh2', 'ombch2'], 'inflate_cov': False}}},
    'CMB-compressed-R-lA-marg-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.CMB_shift_parameters_compression_PR3': {'observables': ['R', 'lA'], 'inflate_cov': True}}},
    'CMB-compressed-R-lA-ombh2-marg-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.CMB_shift_parameters_compression_PR3': {'observables': ['R', 'lA', 'ombh2'], 'inflate_cov': True}}},
    'CMB-compressed-R-lA-ombh2-ombch2-marg-nnu': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.CMB_shift_parameters_compression_PR3': {'observables': ['R', 'lA', 'ombh2', 'omch2'], 'inflate_cov': True}}},
    'CMB-compressed-R-lA': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.CMB_shift_parameters_compression_PR3': {'observables': ['R', 'lA'], 'inflate_cov': False}}},
    'CMB-compressed-R-lA-ombh2': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.CMB_shift_parameters_compression_PR3': {'observables': ['R', 'lA', 'ombh2'], 'inflate_cov': False}}},
    'CMB-compressed-R-lA-ombh2-ombch2': {'family': 'cmb_compressed', 'parameterization': 'background', 'cobaya': {'cosmo.cobaya_likelihoods.cmb.CMB_shift_parameters_compression_PR3': {'observables': ['R', 'lA', 'ombh2', 'omch2'], 'inflate_cov': False}}},
}
