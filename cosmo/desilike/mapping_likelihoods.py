"""desilike likelihood mapping.

Maps named likelihoods from the cobaya registry (see
``cosmo/cobaya/mapping_likelihoods.py``) to desilike likelihood instances.

Skipped families / individual names:
  - planck2020-* : no desilike equivalent
  - *-momento : no desilike equivalent
  - planckpr4* : no desilike equivalent
  - wmap-* : no desilike equivalent
  - planck2018-lensing* : no desilike lensing equivalent for Planck 2018
  - planck2018-highl-CamSpec-* (PR3) : no PR3 CamSpec class in desilike
  - planck-NPIPE-highl-CamSpec-TE/EE : no per-spectrum class in desilike
  - CMB-compressed-fake-* : uses a non-existent cobaya class, skipped
"""


# ---------------------------------------------------------------------------
# BAO
# ---------------------------------------------------------------------------

# Cobaya BAO name → zbins for DESIDR2BAOLikelihood (None = all tracers)
_BAO_ZBINS = {
    'desi-dr2-bao-all':        None,
    'desi-dr2-bao-bgs':        ['BGS'],
    'desi-dr2-bao-lrg-z1':     ['LRG1'],
    'desi-dr2-bao-lrg-z2':     ['LRG2'],
    'desi-dr2-bao-lrgpluselg': ['LRG3+ELG1'],
    'desi-dr2-bao-elg':        ['ELG2'],
    'desi-dr2-bao-qso':        ['QSO'],
    'desi-dr2-bao-lya':        ['Lya'],
}


# ---------------------------------------------------------------------------
# SN
# ---------------------------------------------------------------------------

# Cobaya SN name → (desilike class name, zrange)
_SN_MAP = {
    'desdovekie':           ('DESY5DovekieSNLikelihood', (None, None)),
    'pantheon':             ('PantheonSNLikelihood',     (None, None)),
    'pantheonplus':         ('PantheonPlusSNLikelihood', (None, None)),
    'union3':               ('Union3SNLikelihood',       (None, None)),
    'desy5sn':              ('DESY5v1SNLikelihood',      (None, None)),
    'desy5sn-zmin0.0':      ('DESY5v1SNLikelihood',      (0.0,  None)),
    'desy5sn-zmin0.05':     ('DESY5v1SNLikelihood',      (0.05, None)),
    'desy5sn-zmin0.1':      ('DESY5v1SNLikelihood',      (0.1,  None)),
    'desy5sn-zmin0.2':      ('DESY5v1SNLikelihood',      (0.2,  None)),
    'pantheonplus-zmin0.1': ('PantheonPlusSNLikelihood', (0.1,  None)),
    'union3-zmin0.1':       ('Union3SNLikelihood',       (0.1,  None)),
}


# ---------------------------------------------------------------------------
# Full-shape (FS)
# ---------------------------------------------------------------------------

# Stat abbreviation → generate_likelihood_options_helper stats list
_FS_STAT_ABBREVS = {
    's2':         ['mesh2_spectrum'],
    's2-s3':      ['mesh2_spectrum', 'mesh3_spectrum'],
    's2-baor':    ['mesh2_spectrum', 'recon_bao'],
    's2-c2r':     ['mesh2_spectrum', 'recon_particle2_correlation'],
    's2-s3-c2r':  ['mesh2_spectrum', 'mesh3_spectrum', 'recon_particle2_correlation'],
    's2-s3-baor': ['mesh2_spectrum', 'mesh3_spectrum', 'recon_bao'],
}

# Individual tracer strings (used by the "all" variants)
_FS_ALL_TRACERS = ['BGS1', 'LRG1', 'LRG2', 'LRG3', 'LRG3xELG1', 'ELG2', 'QSO1']

# DR2 full-shape name → kwargs for generate_likelihood_options_helper.
# Entries without a 'tracer' key represent the "all individual tracers" combination.
_FS_TRACERS = {}
for _stat_abbrev, _stats in _FS_STAT_ABBREVS.items():
    _FS_TRACERS[f'desi-dr2-fs-{_stat_abbrev}-all'] = {'stats': _stats}
    for _short, _tracer in [('bgs', 'BGS1'), ('lrg-z1', 'LRG1'), ('lrg-z2', 'LRG2'),
                             ('lrg-z3', 'LRG3'), ('lrgxelg', 'LRG3xELG1'),
                             ('elg', 'ELG2'), ('qso', 'QSO1')]:
        _FS_TRACERS[f'desi-dr2-fs-{_stat_abbrev}-{_short}'] = {'tracer': _tracer, 'stats': _stats}


# ---------------------------------------------------------------------------
# CMB
# ---------------------------------------------------------------------------

# Names that have no desilike equivalent and should raise NotImplementedError
_CMB_NOT_IMPLEMENTED = {
    # Planck 2018 PR3 CamSpec (no PR3 CamSpec class in desilike)
    'planck2018-highl-CamSpec-TT',
    'planck2018-highl-CamSpec-TTTEEE',
    'planck2018-highl-CamSpec-TT-clik',
    'planck2018-highl-CamSpec-TTTEEE-clik',
    'planck2018-highl-CamSpec2021-TT',
    'planck2018-highl-CamSpec2021-TTTEEE',
    # NPIPE CamSpec per-spectrum (no TE-only / EE-only class)
    'planck-NPIPE-highl-CamSpec-TE',
    'planck-NPIPE-highl-CamSpec-EE',
    # Planck 2018 lensing (user excluded)
    'planck2018-lensing',
    'planck2018-lensing-clik',
}


def get_likelihood(name, cosmo=None, **kwargs):
    """Return the desilike likelihood instance(s) for a named likelihood.

    Parameters
    ----------
    name : str
        Likelihood name as defined in ``cosmo/cobaya/mapping_likelihoods.py``,
        e.g. ``'desi-dr2-bao-all'``, ``'pantheonplus'``, ``'planck2018-lowl-EE-sroll2'``.
    cosmo : BasePrimordialCosmology, optional
        Shared cosmology calculator.  When ``None`` each likelihood constructs
        its own default (``CosmoprimoCosmology(fiducial='DESI')``).

    Returns
    -------
    likelihood or list of likelihoods
        A single desilike likelihood instance, or a list of instances for names
        that decompose into multiple independent likelihoods.

    Raises
    ------
    ValueError
        If *name* is not in the known registry.
    NotImplementedError
        If *name* is known but has no desilike equivalent yet.
    """
    # ------------------------------------------------------------------
    # BAO
    # ------------------------------------------------------------------
    if name in _BAO_ZBINS:
        from desilike.likelihoods.bao import DESIDR2BAOLikelihood
        return DESIDR2BAOLikelihood(zbins=_BAO_ZBINS[name], cosmo=cosmo)

    # ------------------------------------------------------------------
    # BBN
    # ------------------------------------------------------------------
    if name == 'schoneberg2024-bbn':
        from desilike.likelihoods.bbn import Schoneberg2024BBNLikelihood
        return Schoneberg2024BBNLikelihood(cosmo=cosmo)

    # ------------------------------------------------------------------
    # SN
    # ------------------------------------------------------------------
    if name in _SN_MAP:
        cls_name, zrange = _SN_MAP[name]
        import desilike.likelihoods.supernovae as sn_mod
        cls = getattr(sn_mod, cls_name)
        return cls(cosmo=cosmo, zrange=zrange)

    # ------------------------------------------------------------------
    # CMB — names that have no desilike equivalent
    # ------------------------------------------------------------------
    if name in _CMB_NOT_IMPLEMENTED:
        raise NotImplementedError(f'No desilike equivalent for likelihood {name!r}.')

    # ------------------------------------------------------------------
    # CMB — Planck 2018 low-ell (via candl/clipy clik wrappers)
    # ------------------------------------------------------------------
    if name in ('planck2018-lowl-TT', 'planck2018-lowl-TT-clik', 'planck2018-lowl-TT-11-29-clik'):
        from desilike.likelihoods.cmb.candl import PlanckPR3LowlTTLikelihood
        return PlanckPR3LowlTTLikelihood(cosmo=cosmo)

    if name in ('planck2018-lowl-EE', 'planck2018-lowl-EE-clik'):
        from desilike.likelihoods.cmb.candl import PlanckPR3LowlEELikelihood
        return PlanckPR3LowlEELikelihood(cosmo=cosmo)

    if name == 'planck2018-lowl-EE-sroll2':
        from desilike.likelihoods.cmb.candl import PlanckPR3LowlEESroll2Likelihood
        return PlanckPR3LowlEESroll2Likelihood(cosmo=cosmo)

    # ------------------------------------------------------------------
    # CMB — Planck 2018 plik high-ell (via candl/clipy clik wrappers)
    # ------------------------------------------------------------------
    if name == 'planck2018-highl-plik-TT':
        from desilike.likelihoods.cmb.candl import PlanckPR3TTLikelihood
        return PlanckPR3TTLikelihood(cosmo=cosmo)

    if name == 'planck2018-highl-plik-TTTEEE':
        from desilike.likelihoods.cmb.candl import PlanckPR3TTTEEELikelihood
        return PlanckPR3TTTEEELikelihood(cosmo=cosmo)

    if name == 'planck2018-highl-plik-TTTEEE-lite':
        from desilike.likelihoods.cmb.candl import PlanckPR3TTTEEELiteLikelihood
        return PlanckPR3TTTEEELiteLikelihood(cosmo=cosmo)

    # ------------------------------------------------------------------
    # CMB — Planck NPIPE CamSpec high-ell (native Python)
    # ------------------------------------------------------------------
    if name == 'planck-NPIPE-highl-CamSpec-TT':
        from desilike.likelihoods.cmb import TTHighlPlanckNPIPECamspecLikelihood
        return TTHighlPlanckNPIPECamspecLikelihood(cosmo=cosmo)

    if name == 'planck-NPIPE-highl-CamSpec-TTTEEE':
        from desilike.likelihoods.cmb import TTTEEEHighlPlanckNPIPECamspecLikelihood
        return TTTEEEHighlPlanckNPIPECamspecLikelihood(cosmo=cosmo)

    if name == 'planck-NPIPE-highl-CamSpec-TTTEEE-ell-max-600':
        from desilike.likelihoods.cmb import TTTEEEHighlPlanckNPIPECamspecEllMax600Likelihood
        return TTTEEEHighlPlanckNPIPECamspecEllMax600Likelihood(cosmo=cosmo)

    if name == 'planck-NPIPE-highl-CamSpec-TTTEEE-cuts-for-act':
        from desilike.likelihoods.cmb import TTTEEEHighlPlanckNPIPECamspecCutsForACTLikelihood
        return TTTEEEHighlPlanckNPIPECamspecCutsForACTLikelihood(cosmo=cosmo)

    # ------------------------------------------------------------------
    # CMB — CMB-SPA composite (Planck low-ell + ACT DR6 CMB + SPT-3G + lensing)
    # ------------------------------------------------------------------
    if name in ('CMB-SPA', 'CMB-SPA-tauprior'):
        from desilike.likelihoods.cmb.candl import (
            PlanckPR3LowlTTLikelihood, PlanckPR3LowlEESroll2Likelihood,
            PlanckPR3TTTEEELiteLikelihood, ACTDR6TTTEEELikelihood, SPT3GD1TnELikelihood,
        )
        from desilike.likelihoods.cmb import ACTDR6SPTLensingLikelihood
        likelihoods = [
            PlanckPR3LowlTTLikelihood(cosmo=cosmo),
            PlanckPR3LowlEESroll2Likelihood(cosmo=cosmo),
            # Planck plik-lite restricted to ell ≤ 1000 (TT) / ≤ 600 (TE, EE)
            # to avoid overlap with ACT DR6 primary CMB at higher ell.
            PlanckPR3TTTEEELiteLikelihood(cosmo=cosmo,
                data_selection=['TT ell<1001 only', 'TE ell<601 only', 'EE ell<601 only']),
            # ACT DR6 foreground-marginalized CMB-only (same data as candl_data.ACT_DR6_TTTEEE)
            # restricted to ell > 600 to complement the Planck plik-lite cut above.
            ACTDR6TTTEEELikelihood(cosmo=cosmo, data_selection=['ell>600 only']),
            SPT3GD1TnELikelihood(variant='lite', cosmo=cosmo),
            ACTDR6SPTLensingLikelihood(variant='actplanckspt3g_baseline', cosmo=cosmo),
        ]
        if name == 'CMB-SPA-tauprior':
            from desilike.likelihoods.bbn import BaseBBNLikelihood
            # External Gaussian prior on tau_reio used in place of a low-ell EE data constraint.
            likelihoods.append(
                BaseBBNLikelihood(mean=[0.051], covariance=[[0.006 ** 2]],
                                  quantities=['tau_reio'], cosmo=cosmo)
            )
        return likelihoods

    # ------------------------------------------------------------------
    # CMB — Planck 2018 thetastar priors
    # ------------------------------------------------------------------
    if name == 'planck2018-thetastar-fixed-nnu':
        from cosmo.desilike.external_likelihoods.cmb import PlanckPR3ThetaStarFixedNnuLikelihood
        return PlanckPR3ThetaStarFixedNnuLikelihood(cosmo=cosmo)

    if name == 'planck2018-thetastar-varied-nnu':
        from cosmo.desilike.external_likelihoods.cmb import PlanckPR3ThetaStarVariedNnuLikelihood
        return PlanckPR3ThetaStarVariedNnuLikelihood(cosmo=cosmo)

    if name == 'planck2018-thetastar-fixed-marg-nnu':
        from cosmo.desilike.external_likelihoods.cmb import PlanckPR3ThetaStarMargNnuLikelihood
        return PlanckPR3ThetaStarMargNnuLikelihood(cosmo=cosmo)

    # ------------------------------------------------------------------
    # CMB — Planck 2018 rdrag prior
    # ------------------------------------------------------------------
    if name == 'planck2018-rdrag-fixed-nnu':
        from cosmo.desilike.external_likelihoods.cmb import PlanckPR3RdragLikelihood
        return PlanckPR3RdragLikelihood(cosmo=cosmo)

    # ------------------------------------------------------------------
    # CMB — PR4 standard compression (thetastar, ombh2, ombch2)
    # ------------------------------------------------------------------
    _PR4_STANDARD_MAP = {
        'CMB-compressed-theta':                    (['thetastar'],               False),
        'CMB-compressed-ombh2':                    (['ombh2'],                   False),
        'CMB-compressed-ombch2':                   (['ombch2'],                  False),
        'CMB-compressed-theta-ombh2':              (['thetastar', 'ombh2'],      False),
        'CMB-compressed-ombh2-ombch2':             (['ombh2', 'ombch2'],         False),
        'CMB-compressed-theta-ombh2-ombch2':       (['thetastar', 'ombh2', 'ombch2'], False),
        'CMB-compressed-theta-ombh2-marg-nnu':     (['thetastar', 'ombh2'],      True),
        'CMB-compressed-ombh2-ombch2-marg-nnu':    (['ombh2', 'ombch2'],         True),
        'CMB-compressed-theta-ombh2-ombch2-marg-nnu': (['thetastar', 'ombh2', 'ombch2'], True),
    }
    if name in _PR4_STANDARD_MAP:
        from cosmo.desilike.external_likelihoods.cmb import PlanckPR4StandardCompressionLikelihood
        observables, inflate_cov = _PR4_STANDARD_MAP[name]
        return PlanckPR4StandardCompressionLikelihood(observables=observables,
                                                      inflate_cov=inflate_cov, cosmo=cosmo)

    # ------------------------------------------------------------------
    # CMB — PR3 shift-parameter compression (R, lA, ombh2, omch2)
    # ------------------------------------------------------------------
    _PR3_SHIFT_MAP = {
        'CMB-compressed-R-lA':                    (['R', 'lA'],               False),
        'CMB-compressed-R-lA-ombh2':              (['R', 'lA', 'ombh2'],      False),
        'CMB-compressed-R-lA-ombh2-ombch2':       (['R', 'lA', 'ombh2', 'omch2'], False),
        'CMB-compressed-R-lA-marg-nnu':           (['R', 'lA'],               True),
        'CMB-compressed-R-lA-ombh2-marg-nnu':     (['R', 'lA', 'ombh2'],      True),
        'CMB-compressed-R-lA-ombh2-ombch2-marg-nnu': (['R', 'lA', 'ombh2', 'omch2'], True),
    }
    if name in _PR3_SHIFT_MAP:
        from cosmo.desilike.external_likelihoods.cmb import PlanckPR3ShiftParameterCompressionLikelihood
        observables, inflate_cov = _PR3_SHIFT_MAP[name]
        return PlanckPR3ShiftParameterCompressionLikelihood(observables=observables,
                                                            inflate_cov=inflate_cov, cosmo=cosmo)

    # ------------------------------------------------------------------
    # CMB — CMB-SP4A (CamSpec NPIPE lite + ACT DR6 CMBonly + SPT-3G lite)
    # ------------------------------------------------------------------
    if name == 'CMB-SP4A':
        from desilike.likelihoods.cmb.camspec import CamspecNPIPELiteLikelihood
        from desilike.likelihoods.cmb.candl import ACTDR6TTTEEELikelihood, SPT3GD1TnELikelihood
        return [
            CamspecNPIPELiteLikelihood(
                ell_cuts={'TT': [30, 1500], 'TE': [30, 1000], 'EE': [30, 600]},
                cosmo=cosmo),
            ACTDR6TTTEEELikelihood(
                ell_cuts={'TT': [1500, 6500], 'TE': [1000, 6500], 'EE': [600, 6500]},
                cosmo=cosmo),
            SPT3GD1TnELikelihood(variant='lite', cosmo=cosmo),
        ]

    # ------------------------------------------------------------------
    # CMB — ACT DR6 lensing (JAX-native; uses act_dr6_spt_lenslike)
    # ------------------------------------------------------------------
    if name == 'act-dr6-lensing':
        from desilike.likelihoods.cmb import ACTDR6SPTLensingLikelihood
        return ACTDR6SPTLensingLikelihood(variant='act_baseline', cosmo=cosmo)

    if name == 'planck-act-dr6-lensing':
        from desilike.likelihoods.cmb import ACTDR6SPTLensingLikelihood
        return ACTDR6SPTLensingLikelihood(variant='actplanck_baseline', cosmo=cosmo)

    # ------------------------------------------------------------------
    # Full-shape (FS)
    # ------------------------------------------------------------------
    if name in _FS_TRACERS:
        from full_shape.tools import generate_likelihood_options_helper, get_likelihood as _get_fs_likelihood, fill_fiducial_options
        # Merge registered defaults with any caller overrides (except cache_dir, which goes to _get_fs_likelihood)
        cache_dir = kwargs.get('cache_dir', None)
        helper_kwargs = {**_FS_TRACERS[name], **{k: v for k, v in kwargs.items() if k != 'cache_dir'}}
        tracer = helper_kwargs.pop('tracer', None)
        tracers = _FS_ALL_TRACERS if tracer is None else [tracer]
        options = {'likelihoods': [generate_likelihood_options_helper(tracer=t, **helper_kwargs) for t in tracers]}
        options = fill_fiducial_options(options)
        if cache_dir is None:
            for likelihood_options in options['likelihoods']:
                likelihood_options['emulator'] = {'name': ''}
        cosmology_options = cosmo if cosmo is not None else options.get('cosmology')
        return _get_fs_likelihood(options['likelihoods'], cosmology_options=cosmology_options, cache_dir=cache_dir)

    # ------------------------------------------------------------------
    # Unknown name
    # ------------------------------------------------------------------
    try:
        from cosmo.cobaya.mapping_likelihoods import LIKELIHOOD_REGISTRY
        if name in LIKELIHOOD_REGISTRY:
            raise NotImplementedError(f'No desilike equivalent for likelihood {name!r} (yet) --- just ask!.')
        known = sorted(LIKELIHOOD_REGISTRY)
    except ImportError:
        known = list(_BAO_ZBINS) + list(_SN_MAP) + list(_CMB_NOT_IMPLEMENTED)
    raise ValueError(f'Unknown likelihood {name!r}. Known names: {known}.')
