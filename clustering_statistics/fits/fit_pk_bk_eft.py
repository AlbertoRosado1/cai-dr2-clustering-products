"""
fit_pk_bk_eft.py
================
Joint EFT power spectrum + bispectrum fitting for a DESI Y1 cutsky tracer.

This script sets up and runs a joint EFT likelihood fit to the galaxy power
spectrum *and* bispectrum multipoles measured from DESI Y1 cutsky data.  The
power spectrum theory is handled by ``FOLPSv2TracerPowerSpectrumMultipoles``
and the bispectrum theory by ``FOLPSv2TracerBispectrumMultipoles``.  Both
share the same cosmological template and galaxy bias parameters; only the EFT
counterterms and shot-noise amplitudes differ.

A single tracer is fitted at a time (the original script targets LRG2 / SGC).
Multi-tracer joint fits are not currently implemented for the PS+BK case.

Bispectrum window matrices must be loaded from external text files whose
paths are configured in :class:`FitConfig`. The k grid for the bispectrum
window convolution is also specified in the configuration.

Command-line usage
------------------
Run a chain::

    python fit_pk_bk_eft.py --run_chains

Test the likelihood once::

    python fit_pk_bk_eft.py --test

Plot best-fit theory after sampling::

    python fit_pk_bk_eft.py --plot_bestfit

Plot posterior chains::

    python fit_pk_bk_eft.py --plot_chains
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np

os.environ.setdefault('FOLPS_BACKEND', 'jax')

from desilike import setup_logging
from desilike.theories import Cosmoprimo
from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate
from desilike.theories.galaxy_clustering.full_shape import (
    FOLPSv2TracerPowerSpectrumMultipoles,
    FOLPSv2TracerBispectrumMultipoles,
)
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.emulators import Emulator, TaylorEmulatorEngine
from desilike.likelihoods import ObservablesGaussianLikelihood, SumLikelihood
from cosmoprimo.fiducial import DESI

from cutsky_data_tools import (
    TRACER_REDSHIFTS,
    build_pk_bk_data_cutsky,
)

logger = logging.getLogger('fit_pk_bk_eft')


# ===========================================================================
# Configuration
# ===========================================================================

class FitConfig:
    """Central configuration for the joint EFT PS+BK fitting.

    Attributes
    ----------
    tracer : str
        Tracer label: one of ``'BGS'``, ``'LRG1'``, ``'LRG2'``, ``'LRG3'``,
        ``'ELG'``, or ``'QSO'``.
    region : str
        Sky region: ``'SGC'``, ``'NGC'``, or ``'GCcomb'``.
    prior_basis : str
        Nuisance parameter parametrisation: ``'physical_prior_doc'`` or
        ``'standard'``.
    pt_model : str
        Perturbation theory model.  ``'EFT'`` uses FOLPSv2.
    damping : str
        FoG damping profile: ``'lor'``, ``'exp'``, or ``'vdg'``.
    k_max_p : float
        Maximum k [h/Mpc] for the power spectrum data vector.
    k_max_b0 : float
        Maximum k [h/Mpc] for the bispectrum monopole B_000.
    k_max_b2 : float
        Maximum k [h/Mpc] for the bispectrum quadrupole B_202.
    bk_window_dir : str
        Directory containing the bispectrum window matrices (``wcmat_000_*``
        and ``wcmat_202_*`` text files).
    k_window_b : ndarray or None
        Fine k grid [h/Mpc] on which the bispectrum window matrices are
        defined.  Set this to the array matching your window matrix files.
        If ``None``, the default grid from the original analysis is used.
    use_emulator : bool
        Enable 4th-order Taylor emulation of the PS theory.
    A_full : bool
        Use the full A-matrix in FOLPS.
    b3_coev : bool
        Enforce the co-evolution prior on b3.
    width_EFT : float
        Gaussian prior width for EFT counterterms alpha0/2/4.
    width_SN0 : float
        Gaussian prior width for shot-noise monopole sn0.
    width_SN2 : float
        Gaussian prior width for shot-noise quadrupole sn2.
    GR_criteria : float
        Cobaya Gelman–Rubin convergence criterion R-1 < GR_criteria.
    sampler : str
        Sampler backend: ``'cobaya'`` or ``'emcee'``.
    base_dir : str
        Root output directory for chains.
    """

    tracer: str = 'LRG2'
    region: str = 'SGC'

    prior_basis: str = 'physical_prior_doc'
    pt_model: str = 'EFT'
    damping: str = 'lor'
    A_full: bool = False
    b3_coev: bool = False

    k_max_p:  float = 0.201
    k_max_b0: float = 0.20
    k_max_b2: float = 0.03

    # Bispectrum window matrix configuration
    bk_window_dir: str = (
        '/global/cfs/cdirs/desi/users/jaides26/window_function/wc_matrices'
    )
    #: Fine k grid matching the bispectrum window matrix columns.
    #: Keep as ``None`` to use the default grid from the original analysis.
    k_window_b: np.ndarray = None

    use_emulator: bool = True

    width_EFT: float = 12.5
    width_SN0: float = 2.0
    width_SN2: float = 5.0

    sampler: str = 'cobaya'
    GR_criteria: float = 0.05
    restart_chain: bool = False

    base_dir: str = (
        '/global/cfs/cdirs/desicollab/users/prakharb/mock_challenge/'
        'cutsky_mocks/base'
    )

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f'Unknown configuration key: {key!r}')
            setattr(self, key, val)
        # Default fine k grid (matches original analysis window matrix files)
        if self.k_window_b is None:
            self.k_window_b = np.array([
                0.00574982, 0.01023557, 0.01526211, 0.02031268, 0.02516689,
                0.03020486, 0.03517139, 0.04012842, 0.04507161, 0.05004095,
                0.05507535, 0.0600698,  0.06512571, 0.07011277, 0.07502157,
                0.08004289, 0.08504873, 0.09004322, 0.09502533, 0.1000098,
                0.10504149, 0.1100224,  0.11502614, 0.12001702, 0.12502202,
                0.13002994, 0.13502229, 0.14005739, 0.14502593, 0.1500209,
                0.15502526, 0.160035,   0.16505188, 0.17003695, 0.1750324,
                0.18003276, 0.18504025, 0.1900268,  0.19500211, 0.20001755,
                0.20502544, 0.21001923, 0.21502351, 0.22001683, 0.22501584,
                0.23002834, 0.23502663, 0.2400277,  0.24503438, 0.25002842,
                0.25501355, 0.26001043, 0.26502645, 0.27002398, 0.27502142,
                0.28002714, 0.28503308, 0.29001812, 0.29500021, 0.30001302,
                0.3050127,  0.31002231, 0.31501079, 0.32000858, 0.32502457,
                0.33001103, 0.33501358, 0.34001106, 0.34501103, 0.35001633,
                0.35501422, 0.3600208,  0.36501188, 0.37001315, 0.37501782,
                0.3800033,  0.38501567, 0.39002782, 0.39500995, 0.40000344,
            ])

    def chain_name(self) -> str:
        """Return the output chain filename (without extension)."""
        name = (
            f'{self.base_dir}/{self.tracer}_{self.region}'
            f"_{'std' if self.prior_basis == 'standard' else 'phys'}"
            f'_kr{self.k_max_p:.3f}'
            f'_kb0{self.k_max_b0:.3f}_kb2{self.k_max_b2:.3f}'
            f'_{self.pt_model}'
            f"_{'Afull' if self.A_full else 'Ano'}"
            f"{f'_damping_{self.damping}' if self.damping != 'lor' else ''}"
        )
        return name

    def load_bk_window_matrices(self) -> tuple:
        """Load bispectrum window matrices for the configured tracer and region.

        Window matrix files are expected to match the naming pattern::

            wcmat_000_{tracer_str}_{region}_{zstr}_HF_finebin.txt
            wcmat_202_{tracer_str}_{region}_{zstr}_HF_finebin.txt

        Returns
        -------
        wmat_000 : ndarray
            Window matrix for the bispectrum monopole (B_000).
        wmat_202 : ndarray
            Window matrix for the bispectrum quadrupole (B_202).

        Notes
        -----
        The file naming convention used by jaides26's pipeline is, e.g.::

            wcmat_000_LRG_SGC_0.6z0.8_HF_finebin.txt

        Update the ``_TRACER_STR`` and ``_ZSTR`` dictionaries below if your
        files use a different convention.
        """
        # Map tracer labels to the string used in the window matrix filenames
        _TRACER_STR = {
            'BGS':  'BGS',
            'LRG1': 'LRG',
            'LRG2': 'LRG',
            'LRG3': 'LRG',
            'ELG':  'ELG_LOPnotqso',
            'QSO':  'QSO',
        }
        # Map tracer labels to the redshift string used in the filenames
        _ZSTR = {
            'BGS':  '0.1z0.4',
            'LRG1': '0.4z0.6',
            'LRG2': '0.6z0.8',
            'LRG3': '0.8z1.1',
            'ELG':  '1.1z1.6',
            'QSO':  '0.8z2.1',
        }
        tracer_str = _TRACER_STR[self.tracer]
        z_str      = _ZSTR[self.tracer]
        wc_dir = Path(self.bk_window_dir)

        fn_000 = wc_dir / f'wcmat_000_{tracer_str}_{self.region}_{z_str}_HF_finebin.txt'
        fn_202 = wc_dir / f'wcmat_202_{tracer_str}_{self.region}_{z_str}_HF_finebin.txt'

        for fn in (fn_000, fn_202):
            if not fn.exists():
                raise FileNotFoundError(
                    f'Bispectrum window matrix file not found:\n  {fn}\n'
                    'Update FitConfig.bk_window_dir or the file naming '
                    'conventions in FitConfig.load_bk_window_matrices().'
                )
        wmat_000 = np.loadtxt(fn_000)
        wmat_202 = np.loadtxt(fn_202)
        return wmat_000, wmat_202


# ===========================================================================
# Cosmology
# ===========================================================================

def setup_cosmo() -> Cosmoprimo:
    """Create and configure the Cosmoprimo cosmology calculator.

    The sampled parameters are ``omega_cdm``, ``omega_b``, ``logA``, and
    ``h``; ``n_s`` and ``tau_reio`` are fixed.

    Returns
    -------
    Cosmoprimo
        Configured cosmology calculator.
    """
    cosmo = Cosmoprimo(engine='class')
    cosmo.init.params['H0']      = dict(derived=True)
    cosmo.init.params['Omega_m'] = dict(derived=True)
    cosmo.init.params['sigma8_m'] = dict(derived=True)

    cosmo.params['n_s'].update(fixed=True)
    cosmo.params['tau_reio'].update(fixed=True)
    cosmo.params['omega_b'].update(
        fixed=False,
        prior={'dist': 'norm', 'loc': 0.02237, 'scale': 0.00037},
    )
    cosmo.params['h'].update(
        fixed=False,
        prior={'dist': 'uniform', 'limits': [0.5, 0.9]},
    )
    cosmo.params['omega_cdm'].update(
        fixed=False,
        prior={'dist': 'uniform', 'limits': [0.05, 0.2]},
    )
    cosmo.params['logA'].update(
        fixed=False,
        prior={'dist': 'uniform', 'limits': [2.0, 4.0]},
    )
    return cosmo


# ===========================================================================
# Nuisance parameter priors
# ===========================================================================

def make_params(
    prior_basis: str,
    width_EFT: float,
    width_SN0: float,
    width_SN2: float,
    pt_model: str = 'EFT',
    b3_coev: bool = False,
    sigma8_fid: float = None,
    n_bar: float = None,
) -> dict:
    """Build nuisance parameter priors for the joint PS+BK fit.

    This extends the PS-only priors by including bispectrum-specific nuisance
    parameters (c1, c2, Pshot, Bshot) for both the physical and standard bases.

    Parameters
    ----------
    prior_basis : str
        ``'physical_prior_doc'`` / ``'physical'`` or ``'standard'``.
    width_EFT : float
        Gaussian prior width for alpha0/2/4 EFT counterterms.
    width_SN0 : float
        Gaussian prior width for shot-noise monopole sn0.
    width_SN2 : float
        Gaussian prior width for shot-noise quadrupole sn2.
    pt_model : str, optional
        PT model name.  Default ``'EFT'``.
    b3_coev : bool, optional
        Enforce co-evolution prior on b3.  Default ``False``.
    sigma8_fid : float, optional
        Fiducial sigma_8(z).  Required for physical basis.
    n_bar : float, optional
        Mean galaxy number density [h^3/Mpc^3].  When provided, the Pshot
        and Bshot prior widths are scaled by 1/n_bar.  If ``None`` and
        ``prior_basis == 'standard'``, a default of 0.0002118763 is used.

    Returns
    -------
    dict
        Mapping of parameter name to prior/value specification dict.
    """
    params = {}

    if prior_basis in ('physical', 'physical_prior_doc'):
        if sigma8_fid is None:
            raise ValueError('sigma8_fid is required for the physical prior basis.')

        # Galaxy bias
        params['b1p'] = {'prior': {'dist': 'uniform', 'limits': [0.1, 4]}}
        params['b2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        params['bsp'] = {
            'prior': {'dist': 'norm', 'loc': -2 / 7 * sigma8_fid**2, 'scale': 5}
        }
        if b3_coev:
            params['b3p'] = {'derived': '32/315*({b1p}-1)'}
        else:
            params['b3p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}

        # PS EFT counterterms
        params['alpha0p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha4p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['sn0p']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN0}}
        params['sn2p']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN2}}

        if pt_model == 'EFT':
            params['X_FoG_pp'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
        else:
            params['X_FoG_p']  = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}

        # BS nuisance parameters (physical basis uses normalised versions)
        params['c1p']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        params['c2p']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        params['Pshotp'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
        params['Bshotp'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}

    else:  # standard basis
        # Galaxy bias
        params['b1'] = {'prior': {'dist': 'uniform', 'limits': [1e-5, 10]}}
        params['b2'] = {'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        params['bs'] = {'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        if b3_coev:
            params['b3'] = {'derived': '32/315*({b1}-1)'}
        else:
            params['b3'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}

        # PS EFT counterterms
        params['alpha0'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha2'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha4'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['sn0']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN0}}
        params['sn2']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN2}}

        if pt_model == 'EFT':
            params['X_FoG_p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
        else:
            params['X_FoG_p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}

        # BS nuisance parameters  (standard basis, physical units)
        # The Poisson shot noise sets a natural scale for Pshot and Bshot.
        _n_bar_default = 0.0002118763  # default n_bar [h^3/Mpc^3] for LRG2
        _n_bar_eff = n_bar if n_bar is not None else _n_bar_default
        P_poisson  = 1.0 / _n_bar_eff

        params['c1']    = {'prior': {'dist': 'norm', 'loc': 66.6,  'scale': 66.6 * 4}}
        params['c2']    = {'prior': {'dist': 'norm', 'loc': 0,     'scale': 1 * 4}}
        params['Pshot'] = {'prior': {'dist': 'norm', 'loc': 0,     'scale': P_poisson * 4}}
        params['Bshot'] = {'prior': {'dist': 'norm', 'loc': 0,     'scale': P_poisson * 4}}

    return params


# ===========================================================================
# Theory setup
# ===========================================================================

def setup_theories(
    cfg: FitConfig,
    cosmo: Cosmoprimo,
    fiducial,
) -> dict:
    """Build PS and BS theory objects for the configured tracer.

    Parameters
    ----------
    cfg : FitConfig
        Configuration object.
    cosmo : Cosmoprimo
        Configured cosmology calculator.
    fiducial : cosmoprimo Cosmology
        Fiducial DESI cosmology.

    Returns
    -------
    dict
        ``{'ps': ps_theory, 'bs': bs_theory}``
    """
    tracer     = cfg.tracer
    z          = TRACER_REDSHIFTS[tracer]
    sigma8_fid = fiducial.sigma8_z(z)

    template = DirectPowerSpectrumTemplate(fiducial=fiducial, cosmo=cosmo, z=z)

    # Load bispectrum window matrices required by FOLPSv2 bispec calculator
    wmat_000, wmat_202 = cfg.load_bk_window_matrices()

    # Power spectrum theory
    ps_theory = FOLPSv2TracerPowerSpectrumMultipoles(
        template=template,
        prior_basis=cfg.prior_basis,
        A_full=cfg.A_full,
        b3_coev=cfg.b3_coev,
        damping=cfg.damping,
        sigma8_fid=sigma8_fid,
        h_fid=fiducial.h,
    )

    # Bispectrum theory
    bs_theory = FOLPSv2TracerBispectrumMultipoles(
        template=template,
        prior_basis=cfg.prior_basis,
        A_full=cfg.A_full,
        damping=cfg.damping,
        sigma8_fid=sigma8_fid,
        h_fid=fiducial.h,
        k_window=cfg.k_window_b,
        wmat_000=wmat_000,
        wmat_202=wmat_202,
    )

    # Apply nuisance parameter priors to both theories
    params = make_params(
        prior_basis=cfg.prior_basis,
        width_EFT=cfg.width_EFT,
        width_SN0=cfg.width_SN0,
        width_SN2=cfg.width_SN2,
        pt_model=cfg.pt_model,
        b3_coev=cfg.b3_coev,
        sigma8_fid=sigma8_fid,
    )
    for name, param_spec in params.items():
        for theory in (ps_theory, bs_theory):
            if name in theory.init.params:
                theory.params[name].update(**param_spec)

    return {'ps': ps_theory, 'bs': bs_theory}


# ===========================================================================
# Observable setup
# ===========================================================================

def setup_observables(cfg: FitConfig, theories: dict) -> dict:
    """Build PS and BS observables and the associated joint covariance.

    Parameters
    ----------
    cfg : FitConfig
        Configuration object.
    theories : dict
        ``{'ps': ps_theory, 'bs': bs_theory}`` from :func:`setup_theories`.

    Returns
    -------
    dict with keys
        ps_obs : TracerPowerSpectrumMultipolesObservable
            Power spectrum observable.
        bs_obs : TracerPowerSpectrumMultipolesObservable
            Bispectrum observable.
        cov_joint : ndarray
            Full joint covariance matrix for [P0, P2, B000, B202].
    """
    dataset = build_pk_bk_data_cutsky(
        tracer=cfg.tracer,
        region=cfg.region,
        k_max_p=cfg.k_max_p,
        k_max_b0=cfg.k_max_b0,
        k_max_b2=cfg.k_max_b2,
    )

    # ---- Power spectrum observable ----
    data_ps = np.concatenate([dataset['p0'], dataset['p2']])

    ps_obs = TracerPowerSpectrumMultipolesObservable(
        data=data_ps,
        covariance=dataset['cov_pk'],
        theory=theories['ps'],
        kin=dataset['k_window'],
        ellsin=[0, 2, 4],
        ells=(0, 2),
        k=dataset['k_data'],
        wmatrix=dataset['window_matrix'],
    )

    # ---- Bispectrum observable ----
    # Triangle k values: pick the first k-leg (k1) for each configuration.
    kr_b0 = dataset['kr_b0'][:, 0]   # shape (N_b0,)
    kr_b2 = dataset['kr_b2'][:, 0]   # shape (N_b2,)
    data_bs = np.concatenate([dataset['b000'], dataset['b202']])

    # Sub-block of the joint covariance for the bispectrum data alone.
    n_ps    = len(data_ps)
    n_b0    = len(dataset['b000'])
    n_b2    = len(dataset['b202'])
    cov_joint = dataset['covariance']
    cov_bs    = cov_joint[n_ps:, n_ps:]   # bottom-right block: (N_b0+N_b2)^2

    # NOTE: The bispectrum window convolution is handled *inside*
    # FOLPSv2TracerBispectrumMultipoles (which receives wmat_000/wmat_202 at
    # construction time; see setup_theories).  The observable therefore does
    # not need a separate wmatrix argument — the theory outputs the
    # window-convolved multipoles directly at the observed k points.
    # If your version of desilike requires an explicit wmatrix here, add one
    # by building a block-diagonal matrix from wmat_000 / wmat_202 (remember
    # to apply the same k_max row mask used when selecting the data triangles).
    bs_obs = TracerPowerSpectrumMultipolesObservable(
        data=data_bs,
        covariance=cov_bs,
        theory=theories['bs'],
        ells=[0, 2],
        k=[kr_b0, kr_b2],
    )

    return dict(ps_obs=ps_obs, bs_obs=bs_obs, cov_joint=cov_joint)


# ===========================================================================
# Emulator
# ===========================================================================

def emulate_ps_theory(theories: dict) -> dict:
    """Replace the PS theory with a 4th-order Taylor emulator.

    The BS theory is not emulated because its evaluation is already fast
    (it shares intermediate results with the PS theory) and emulating it
    would require a separate sample grid.

    Parameters
    ----------
    theories : dict
        ``{'ps': ps_theory, 'bs': bs_theory}``.

    Returns
    -------
    dict
        Same structure with the PS entry replaced by the emulated calculator.
    """
    logger.info('Building Taylor emulator for PS theory …')
    emulator = Emulator(theories['ps'], engine=TaylorEmulatorEngine(order=4))
    emulator.set_samples()
    emulator.fit()
    theories = dict(theories)   # shallow copy
    theories['ps'] = emulator.to_calculator()
    logger.info('PS emulator done.')
    return theories


# ===========================================================================
# Parameter namespacing & analytic marginalisation
# ===========================================================================

def namespace_and_marginalise(cfg: FitConfig, theories: dict):
    """Assign a tracer namespace and set up analytic marginalisation.

    EFT counterterms and shot-noise parameters of the PS theory are
    analytically marginalised (``derived='.marg'``).  Both PS and BS
    theories receive the same namespace (the tracer label).

    Parameters
    ----------
    cfg : FitConfig
        Configuration object.
    theories : dict
        ``{'ps': ps_theory, 'bs': bs_theory}`` (modified in-place).
    """
    if cfg.prior_basis in ('physical', 'physical_prior_doc'):
        marg_params = ['alpha0p', 'alpha2p', 'alpha4p', 'sn0p', 'sn2p']
    else:
        marg_params = ['alpha0', 'alpha2', 'alpha4', 'sn0', 'sn2']

    # Analytic marginalisation (PS theory only – BS nuisance params are sampled)
    for param_name in marg_params:
        if param_name in theories['ps'].params:
            theories['ps'].params[param_name].update(derived='.marg')

    # Assign tracer-specific namespace to both PS and BS theories
    for comp in ('ps', 'bs'):
        for param in theories[comp].init.params:
            param.update(namespace=cfg.tracer)

    if logger.isEnabledFor(logging.DEBUG):
        for comp in ('ps', 'bs'):
            for param in theories[comp].all_params:
                logger.debug(
                    '%s.%s [%s] prior=%s',
                    cfg.tracer, param, comp,
                    theories[comp].all_params[param].prior,
                )


# ===========================================================================
# Likelihood
# ===========================================================================

def build_likelihood(
    cfg: FitConfig,
    ps_obs,
    bs_obs,
    cov_joint: np.ndarray,
) -> SumLikelihood:
    """Build the joint PS+BK Gaussian likelihood.

    Both observables share the full joint covariance matrix so that
    cross-correlations between the power spectrum and bispectrum are
    accounted for.

    Parameters
    ----------
    cfg : FitConfig
        Configuration object (unused currently, kept for API consistency).
    ps_obs : TracerPowerSpectrumMultipolesObservable
        Power spectrum observable.
    bs_obs : TracerPowerSpectrumMultipolesObservable
        Bispectrum observable.
    cov_joint : ndarray
        Full joint covariance matrix.

    Returns
    -------
    SumLikelihood
        Likelihood wrapping the joint PS+BK Gaussian.
    """
    joint_likelihood = ObservablesGaussianLikelihood(
        observables=[ps_obs, bs_obs],
        covariance=cov_joint,
    )
    return SumLikelihood([joint_likelihood])


# ===========================================================================
# Chain I/O helpers
# ===========================================================================

def load_chain(path, burnin: float = 0.3):
    """Load and concatenate a desilike chain, discarding the burn-in.

    Parameters
    ----------
    path : str or Path
        Path to the chain .npy file.
    burnin : float, optional
        Fraction of steps to discard as burn-in.  Default 0.3.

    Returns
    -------
    desilike.samples.Chain
    """
    from desilike.samples import Chain
    chain = Chain.load(path).remove_burnin(burnin)
    chains = chain.concatenate([chain])
    logger.info('Loaded chain: %s', chains)
    return chains


# ===========================================================================
# Sampling
# ===========================================================================

def run_cobaya_sampler(likelihood: SumLikelihood, cfg: FitConfig):
    """Run the Cobaya MCMC sampler.

    Parameters
    ----------
    likelihood : SumLikelihood
        The joint PS+BK likelihood.
    cfg : FitConfig
        Configuration object.
    """
    from desilike.samplers.cobaya import CobayaSampler

    chain_path = Path(cfg.chain_name())
    chain_path.parent.mkdir(parents=True, exist_ok=True)

    sampler = CobayaSampler(
        likelihood,
        save_fn=str(chain_path) + '.npy',
        seed=42,
    )
    sampler.run(
        check_every=200,
        check_info={'max_diag_cl': cfg.GR_criteria},
        resume=cfg.restart_chain,
    )


def run_emcee_sampler(likelihood: SumLikelihood, cfg: FitConfig):
    """Run the emcee ensemble sampler.

    Parameters
    ----------
    likelihood : SumLikelihood
        The joint PS+BK likelihood.
    cfg : FitConfig
        Configuration object.
    """
    from desilike.samplers import EmceeSampler

    chain_path = Path(cfg.chain_name())
    chain_path.parent.mkdir(parents=True, exist_ok=True)

    sampler = EmceeSampler(
        likelihood,
        save_fn=str(chain_path) + '.npy',
        nwalkers=None,
        seed=42,
    )
    sampler.run(check_every=100)


# ===========================================================================
# Plotting helpers
# ===========================================================================

def plot_bestfit(cfg: FitConfig, likelihood: SumLikelihood, ps_obs, bs_obs):
    """Evaluate the likelihood at the chain maximum and plot the best-fit model.

    Parameters
    ----------
    cfg : FitConfig
        Configuration object.
    likelihood : SumLikelihood
        Assembled likelihood.
    ps_obs, bs_obs : observables
        PS and BS observables (used for plotting).
    """
    chain = load_chain(cfg.chain_name() + '.npy')
    best  = chain.choice(index='argmax', input=True)
    logger.info('Best-fit parameters: %s', best)
    likelihood(**best)
    ps_obs.plot(fn=f'bestfit_ps_{cfg.tracer}_{cfg.region}.png', kw_save={'dpi': 250})
    bs_obs.plot(fn=f'bestfit_bs_{cfg.tracer}_{cfg.region}.png', kw_save={'dpi': 250})


def plot_chains(cfg: FitConfig):
    """Plot posterior triangle from the chain using GetDist.

    Parameters
    ----------
    cfg : FitConfig
        Configuration object.
    """
    import matplotlib as mpl
    from getdist import MCSamples, plots

    chain   = load_chain(cfg.chain_name() + '.npy')
    samples = chain.to_getdist()

    mpl.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})

    planck_truths = {
        'h':         0.6736,
        'omega_cdm': 0.12,
        'omega_b':   0.02237,
        'logA':      np.log(10**10 * 2.0830e-9),
    }

    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = 14
    g.settings.lab_fontsize  = 14
    g.triangle_plot(
        [samples],
        params=['h', 'omega_cdm', 'omega_b', 'logA'],
        filled=True,
        markers=planck_truths,
    )
    fn = f'chains_{cfg.tracer}_{cfg.region}_pk_bk.png'
    g.export(fn)
    logger.info('Chain plot saved to %s', fn)


# ===========================================================================
# Main
# ===========================================================================

def main():
    """Entry point: parse CLI arguments, build the pipeline, and dispatch."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description='Joint EFT PS+BK fit for a DESI Y1 cutsky tracer.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--run_chains',   action='store_true',
                        help='Run the MCMC sampler.')
    parser.add_argument('--test',         action='store_true',
                        help='Evaluate the likelihood once.')
    parser.add_argument('--plot_bestfit', action='store_true',
                        help='Plot the best-fit model from an existing chain.')
    parser.add_argument('--plot_chains',  action='store_true',
                        help='Plot posterior contours from an existing chain.')
    # Key settings overridable from the CLI
    parser.add_argument('--tracer',  default='LRG2',
                        choices=['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO'],
                        help='Tracer to fit.')
    parser.add_argument('--region', default='SGC',
                        choices=['SGC', 'NGC', 'GCcomb'],
                        help='Sky region.')
    parser.add_argument('--k_max_p',  type=float, default=0.201,
                        help='Maximum k [h/Mpc] for the power spectrum.')
    parser.add_argument('--k_max_b0', type=float, default=0.20,
                        help='Maximum k [h/Mpc] for the bispectrum monopole.')
    parser.add_argument('--k_max_b2', type=float, default=0.03,
                        help='Maximum k [h/Mpc] for the bispectrum quadrupole.')
    parser.add_argument('--no_emulator', action='store_true',
                        help='Disable the Taylor emulator.')
    parser.add_argument('--sampler', default='cobaya',
                        choices=['cobaya', 'emcee'],
                        help='Sampler backend.')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build configuration
    # ------------------------------------------------------------------
    cfg = FitConfig(
        tracer=args.tracer,
        region=args.region,
        k_max_p=args.k_max_p,
        k_max_b0=args.k_max_b0,
        k_max_b2=args.k_max_b2,
        use_emulator=not args.no_emulator,
        sampler=args.sampler,
    )
    logger.info('Chain output: %s', cfg.chain_name())

    # ------------------------------------------------------------------
    # Build the inference pipeline
    # Emulate PS theory first so the observable holds the fast calculator.
    # ------------------------------------------------------------------
    fiducial = DESI()
    cosmo    = setup_cosmo()

    theories = setup_theories(cfg, cosmo, fiducial)

    if cfg.use_emulator:
        theories = emulate_ps_theory(theories)

    obs_dict  = setup_observables(cfg, theories)
    ps_obs    = obs_dict['ps_obs']
    bs_obs    = obs_dict['bs_obs']
    cov_joint = obs_dict['cov_joint']

    namespace_and_marginalise(cfg, theories)

    likelihood = build_likelihood(cfg, ps_obs, bs_obs, cov_joint)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    if args.test:
        loglkl = likelihood()
        logger.info('log-likelihood = %.4f', loglkl)
        logger.info('Varied parameters: %s', likelihood.varied_params)
        logger.info('Test successful.')

    if args.plot_bestfit:
        plot_bestfit(cfg, likelihood, ps_obs, bs_obs)

    if args.plot_chains:
        plot_chains(cfg)

    if args.run_chains:
        if cfg.sampler == 'cobaya':
            run_cobaya_sampler(likelihood, cfg)
        else:
            run_emcee_sampler(likelihood, cfg)


if __name__ == '__main__':
    main()
