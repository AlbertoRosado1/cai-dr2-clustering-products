"""
fit_pk_eft.py
=============
EFT power spectrum fitting for multiple DESI Y1 cutsky tracers.

This script sets up and runs an EFT (Effective Field Theory) likelihood fit
to the anisotropic galaxy power spectrum P(k, mu) multipoles measured from
DESI Y1 cutsky data.  It supports simultaneous fitting of multiple tracers
with shared cosmological parameters and per-tracer nuisance parameters,
optional Taylor emulation of the theory, and analytic marginalisation over
EFT counterterms and shot-noise parameters.

Supported tracers  : BGS, LRG1, LRG2, LRG3, ELG, QSO
Supported PT models: FOLPSv2 (EFT)
Supported samplers : Cobaya (default), emcee

Command-line usage
------------------
Run a chain::

    python fit_pk_eft.py --run_chains

Test the likelihood once (fast sanity check)::

    python fit_pk_eft.py --test

Plot best-fit theory after sampling::

    python fit_pk_eft.py --plot_bestfit

Plot posterior chains::

    python fit_pk_eft.py --plot_chains
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Optional path configuration – point to local desilike / FOLPS builds if
# the system-installed versions should be overridden.
# ---------------------------------------------------------------------------
# sys.path.insert(0, '/path/to/local/desilike')
# sys.path.insert(0, '/path/to/local/FOLPS_JAX/folps')

os.environ.setdefault('FOLPS_BACKEND', 'jax')

from desilike import setup_logging
from desilike.theories import Cosmoprimo
from desilike.theories.galaxy_clustering import (
    DirectPowerSpectrumTemplate,
)
from desilike.theories.galaxy_clustering.full_shape import (
    FOLPSv2TracerPowerSpectrumMultipoles,
    REPTVelocileptorsTracerPowerSpectrumMultipoles,
)
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.emulators import Emulator, TaylorEmulatorEngine
from desilike.likelihoods import ObservablesGaussianLikelihood, SumLikelihood
from cosmoprimo.fiducial import DESI

from cutsky_data_tools import (
    TRACER_REDSHIFTS,
    build_pk_data_cutsky,
)

logger = logging.getLogger('fit_pk_eft')


# ===========================================================================
# Configuration
# ===========================================================================

class FitConfig:
    """Central configuration for the EFT power spectrum fitting.

    Edit the attributes below to customise the run.  All flags that affect
    output filenames are propagated into :func:`make_chain_name` automatically.

    Attributes
    ----------
    tracers : list of str
        Tracer labels to include in the joint fit.
    region : str
        Sky region: ``'SGC'``, ``'NGC'``, or ``'GCcomb'``.
    prior_basis : str
        Nuisance parameter parametrisation.  ``'physical_prior_doc'`` uses
        the *physical* bias basis (b1p, b2p, bsp, …) with Gaussian priors
        centred at perturbation-theory motivated values.  ``'standard'`` uses
        the *standard* bias basis (b1, b2, bs) with broad uniform priors.
    pt_model : str
        Perturbation theory model.  ``'EFT'`` uses FOLPSv2; set to
        ``'rept_velocileptors'`` for REPTVelocileptors.
    damping : str
        FoG damping profile: ``'lor'`` (Lorentzian), ``'exp'`` (Gaussian),
        or ``'vdg'`` (van den Bosch).
    k_max_p : float
        Maximum wavenumber [h/Mpc] included in the power spectrum fit.
    use_emulator : bool
        When ``True``, replace exact theory evaluations with a low-latency
        4th-order Taylor emulator.  Strongly recommended for MCMC sampling.
    A_full : bool
        Whether to use the full A-matrix in FOLPS (vs. the approximate form).
    b3_coev : bool
        If ``True``, enforce co-evolution relation b3 = 32/315*(b1-1).
        Otherwise b3 is a free (sampled) parameter.
    width_EFT : float
        Gaussian prior width for EFT counterterm parameters alpha0/2/4.
    width_SN0 : float
        Gaussian prior width for shot-noise monopole parameter sn0.
    width_SN2 : float
        Gaussian prior width for shot-noise quadrupole parameter sn2.
    GR_criteria : float
        Gelman–Rubin convergence criterion R-1 < GR_criteria for Cobaya.
    sampler : str
        Sampler backend: ``'cobaya'`` or ``'emcee'``.
    base_dir : str
        Root output directory for chains.
    """

    # ---- Tracer selection ----
    tracers: list = None   # e.g. ['LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO']
    region: str = 'GCcomb'

    # ---- Model choices ----
    prior_basis: str = 'physical_prior_doc'
    pt_model: str = 'EFT'
    damping: str = 'lor'
    A_full: bool = False
    b3_coev: bool = False

    # ---- k-range ----
    k_max_p: float = 0.201

    # ---- Nuisance prior widths ----
    width_EFT: float = 12.5
    width_SN0: float = 2.0
    width_SN2: float = 5.0

    # ---- Emulator ----
    use_emulator: bool = True

    # ---- Sampler ----
    sampler: str = 'cobaya'
    GR_criteria: float = 0.03
    restart_chain: bool = False

    # ---- Output ----
    base_dir: str = (
        '/global/cfs/cdirs/desicollab/users/prakharb/mock_challenge/'
        'cutsky_mocks/base'
    )

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f'Unknown configuration key: {key!r}')
            setattr(self, key, val)
        if self.tracers is None:
            self.tracers = ['LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO']

    def chain_name(self) -> str:
        """Return the output chain filename (without extension) derived from settings."""
        tracers_str = (
            'all' if set(self.tracers) == {'BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO'}
            else '_'.join(self.tracers)
        )
        name = (
            f'{self.base_dir}/{self.region}/{tracers_str}'
            f"_{'std' if self.prior_basis == 'standard' else 'phys'}"
            f'_kr{self.k_max_p:.3f}'
            f'_{self.pt_model}'
            f"_{'Afull' if self.A_full else 'Ano'}"
            f"{f'_damping_{self.damping}' if self.damping != 'lor' else ''}"
        )
        return name


# ===========================================================================
# Cosmology
# ===========================================================================

def setup_cosmo() -> Cosmoprimo:
    """Create and configure the Cosmoprimo cosmology calculator.

    The cosmological parameters varied in the fit are ``omega_cdm``, ``omega_b``,
    ``logA``, and ``h``.  ``n_s`` and ``tau_reio`` are held fixed.
    ``H0``, ``Omega_m``, and ``sigma8_m`` are derived.

    Returns
    -------
    Cosmoprimo
        Configured cosmology calculator.
    """
    cosmo = Cosmoprimo(engine='class')

    # Derived parameters to track
    cosmo.init.params['H0'] = dict(derived=True)
    cosmo.init.params['Omega_m'] = dict(derived=True)
    cosmo.init.params['sigma8_m'] = dict(derived=True)

    # Set priors
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
) -> dict:
    """Build per-tracer nuisance parameter priors.

    Parameters
    ----------
    prior_basis : str
        ``'physical_prior_doc'`` / ``'physical'`` for the physical bias
        basis; ``'standard'`` for the standard basis.
    width_EFT : float
        Gaussian prior width (sigma) for alpha0/2/4 counterterms.
    width_SN0 : float
        Gaussian prior width for the shot-noise monopole sn0.
    width_SN2 : float
        Gaussian prior width for the shot-noise quadrupole sn2.
    pt_model : str, optional
        Perturbation theory model (distinguishes EFT from folpsD parameter
        names).  Default ``'EFT'``.
    b3_coev : bool, optional
        Enforce the co-evolution prior on b3.  Default ``False``.
    sigma8_fid : float, optional
        Fiducial sigma_8(z) used to set the Gaussian prior centre for bsp.
        Required when ``prior_basis`` is physical.

    Returns
    -------
    dict
        Mapping of parameter name to prior/value specification dict.
    """
    params = {}

    if prior_basis in ('physical', 'physical_prior_doc'):
        # Shared galaxy bias parameters
        params['b1p'] = {'prior': {'dist': 'uniform', 'limits': [0.1, 4]}}
        params['b2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        if sigma8_fid is None:
            raise ValueError(
                'sigma8_fid is required for physical prior basis to set the '
                'bsp prior centre.'
            )
        params['bsp'] = {
            'prior': {'dist': 'norm', 'loc': -2 / 7 * sigma8_fid**2, 'scale': 5}
        }
        if b3_coev:
            params['b3p'] = {'derived': '32/315*({b1p}-1)'}
        else:
            params['b3p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}

        # EFT counterterms (power-spectrum only)
        params['alpha0p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha4p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['sn0p']   = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN0}}
        params['sn2p']   = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN2}}

        if pt_model == 'EFT':
            params['X_FoG_pp'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
        else:
            params['X_FoG_p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}

    else:  # standard basis
        params['b1'] = {'prior': {'dist': 'uniform', 'limits': [1e-5, 10]}}
        params['b2'] = {'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        params['bs'] = {'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        if b3_coev:
            params['b3'] = {'derived': '32/315*({b1}-1)'}
        else:
            params['b3'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}

        params['alpha0'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha2'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha4'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['sn0']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN0}}
        params['sn2']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN2}}

        if pt_model == 'EFT':
            params['X_FoG_p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
        else:
            params['X_FoG_p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}

    return params


# ===========================================================================
# Theory setup
# ===========================================================================

def setup_theories(cfg: FitConfig, cosmo: Cosmoprimo, fiducial) -> dict:
    """Build a power spectrum theory object for each tracer.

    Parameters
    ----------
    cfg : FitConfig
        Configuration object.
    cosmo : Cosmoprimo
        Configured cosmology calculator (shared across all tracers).
    fiducial : cosmoprimo Cosmology
        Fiducial DESI cosmology used for the power spectrum template.

    Returns
    -------
    dict
        ``{tracer: theory_object}`` mapping.
    """
    theories = {}

    for tracer in cfg.tracers:
        z = TRACER_REDSHIFTS[tracer]
        sigma8_fid = fiducial.sigma8_z(z)

        template = DirectPowerSpectrumTemplate(fiducial=fiducial, cosmo=cosmo, z=z)

        if cfg.pt_model == 'rept_velocileptors':
            theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(
                template=template,
                prior_basis=cfg.prior_basis,
            )
        else:  # default: FOLPSv2 EFT
            theory = FOLPSv2TracerPowerSpectrumMultipoles(
                template=template,
                prior_basis=cfg.prior_basis,
                A_full=cfg.A_full,
                b3_coev=cfg.b3_coev,
                damping=cfg.damping,
                sigma8_fid=sigma8_fid,
                h_fid=fiducial.h,
            )

        # Apply per-tracer parameter priors
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
            if name in theory.init.params:
                theory.params[name].update(**param_spec)

        theories[tracer] = theory

    return theories


# ===========================================================================
# Observable setup
# ===========================================================================

def setup_observables(cfg: FitConfig, theories: dict) -> dict:
    """Build a ``TracerPowerSpectrumMultipolesObservable`` for each tracer.

    Parameters
    ----------
    cfg : FitConfig
        Configuration object.
    theories : dict
        ``{tracer: theory_object}`` mapping from :func:`setup_theories`.

    Returns
    -------
    dict
        ``{tracer: observable}`` mapping.
    """
    observables = {}

    for tracer in cfg.tracers:
        dataset = build_pk_data_cutsky(
            tracer=tracer,
            region=cfg.region,
            k_max_p=cfg.k_max_p,
        )

        data = np.concatenate([dataset['p0'], dataset['p2']])

        observable = TracerPowerSpectrumMultipolesObservable(
            data=data,
            covariance=dataset['cov_pk'],
            theory=theories[tracer],
            kin=dataset['k_window'],
            ellsin=[0, 2, 4],
            ells=(0, 2),
            k=dataset['k_data'],
            wmatrix=dataset['window_matrix'],
        )
        observables[tracer] = observable

    return observables


# ===========================================================================
# Emulator
# ===========================================================================

def emulate_theories(theories: dict) -> dict:
    """Replace exact theory calculators with Taylor emulators.

    Each theory is approximated by a 4th-order Taylor expansion about the
    fiducial cosmology.  This reduces the cost of a single likelihood
    evaluation from O(seconds) to O(milliseconds), which is critical for
    MCMC efficiency.

    Parameters
    ----------
    theories : dict
        ``{tracer: theory_object}`` mapping.

    Returns
    -------
    dict
        ``{tracer: emulated_theory}`` mapping.
    """
    emulated = {}
    for tracer, theory in theories.items():
        logger.info('Building Taylor emulator for %s …', tracer)
        emulator = Emulator(theory, engine=TaylorEmulatorEngine(order=4))
        emulator.set_samples()
        emulator.fit()
        emulated[tracer] = emulator.to_calculator()
        logger.info('Emulator for %s done.', tracer)
    return emulated


# ===========================================================================
# Parameter namespacing & analytic marginalisation
# ===========================================================================

def namespace_and_marginalise(cfg: FitConfig, theories: dict):
    """Assign tracer-specific namespaces and configure analytic marginalisation.

    Each nuisance parameter is prefixed with the tracer label so that, for
    example, the LRG2 EFT counterterm is called ``LRG2.alpha0p`` rather than
    just ``alpha0p``.  Cosmological parameters are shared and therefore not
    namespaced.

    The EFT counterterms and shot-noise parameters are analytically
    marginalised using desilike's ``'.marg'`` mechanism, which eliminates
    them from the MCMC chain and speeds up convergence.

    Parameters
    ----------
    cfg : FitConfig
        Configuration object.
    theories : dict
        ``{tracer: theory_object}`` mapping (modified in-place).
    """
    if cfg.prior_basis in ('physical', 'physical_prior_doc'):
        marg_params = ['alpha0p', 'alpha2p', 'alpha4p', 'sn0p', 'sn2p']
    else:
        marg_params = ['alpha0', 'alpha2', 'alpha4', 'sn0', 'sn2']

    for tracer, theory in theories.items():
        # Analytic marginalisation over EFT / shot-noise nuisance params
        for param_name in marg_params:
            if param_name in theory.params:
                theory.params[param_name].update(derived='.marg')

        # Assign tracer-specific namespace
        for param in theory.init.params:
            param.update(namespace=tracer)

        if logger.isEnabledFor(logging.DEBUG):
            for param in theory.all_params:
                logger.debug('%s.%s : prior=%s', tracer, param, theory.all_params[param].prior)


# ===========================================================================
# Likelihood
# ===========================================================================

def build_likelihood(cfg: FitConfig, observables: dict) -> SumLikelihood:
    """Build the joint likelihood as a sum over per-tracer Gaussian likelihoods.

    Parameters
    ----------
    cfg : FitConfig
        Configuration object.
    observables : dict
        ``{tracer: observable}`` mapping.

    Returns
    -------
    SumLikelihood
        Joint likelihood object.
    """
    likelihoods = [
        ObservablesGaussianLikelihood(observables[tracer])
        for tracer in cfg.tracers
    ]
    return SumLikelihood(likelihoods)


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
        Flattened chain with burn-in removed.
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
        The joint likelihood to sample.
    cfg : FitConfig
        Configuration object (used for chain name, GR criterion, restart).
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
        The joint likelihood to sample.
    cfg : FitConfig
        Configuration object.
    """
    from desilike.samplers import EmceeSampler

    chain_path = Path(cfg.chain_name())
    chain_path.parent.mkdir(parents=True, exist_ok=True)

    sampler = EmceeSampler(
        likelihood,
        save_fn=str(chain_path) + '.npy',
        nwalkers=None,   # auto-set to 2 * n_params
        seed=42,
    )
    sampler.run(check_every=100)


# ===========================================================================
# Plotting helpers
# ===========================================================================

def plot_bestfit(cfg: FitConfig, likelihood: SumLikelihood, observables: dict):
    """Evaluate the likelihood at the chain maximum and plot the best-fit model.

    Parameters
    ----------
    cfg : FitConfig
        Configuration object.
    likelihood : SumLikelihood
        Assembled likelihood.
    observables : dict
        ``{tracer: observable}`` mapping.
    """
    chain = load_chain(cfg.chain_name() + '.npy')
    best = chain.choice(index='argmax', input=True)
    logger.info('Best-fit parameters: %s', best)
    likelihood(**best)
    # Save plot for each tracer
    for tracer in cfg.tracers:
        observables[tracer].plot(
            fn=f'bestfit_{tracer}_{cfg.region}.png',
            kw_save={'dpi': 250},
        )


def plot_chains(cfg: FitConfig):
    """Plot the posterior triangle using GetDist.

    Parameters
    ----------
    cfg : FitConfig
        Configuration object.
    """
    import matplotlib as mpl
    from getdist import MCSamples, plots

    chain = load_chain(cfg.chain_name() + '.npy')
    samples = chain.to_getdist()

    mpl.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})

    planck_truths = {
        'h':        0.6736,
        'omega_cdm': 0.12,
        'omega_b':  0.02237,
        'logA':     np.log(10**10 * 2.0830e-9),
    }

    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = 14
    g.settings.lab_fontsize = 14
    g.triangle_plot(
        [samples],
        params=['h', 'omega_cdm', 'omega_b', 'logA'],
        filled=True,
        markers=planck_truths,
    )
    fn = f'chains_{cfg.region}_pk.png'
    g.export(fn)
    logger.info('Chain plot saved to %s', fn)


# ===========================================================================
# Main
# ===========================================================================

def main():
    """Entry point: parse CLI arguments, build the pipeline, and dispatch."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description='EFT power spectrum fit for DESI Y1 cutsky tracers.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--run_chains', action='store_true',
                        help='Run the MCMC sampler and save the chain.')
    parser.add_argument('--test',       action='store_true',
                        help='Evaluate the likelihood once and print value.')
    parser.add_argument('--plot_bestfit', action='store_true',
                        help='Plot the best-fit model from an existing chain.')
    parser.add_argument('--plot_chains',  action='store_true',
                        help='Plot posterior contours from an existing chain.')
    # Allow overriding key settings from the command line
    parser.add_argument('--tracers', nargs='+',
                        default=['LRG1', 'LRG2', 'LRG3', 'ELG', 'QSO'],
                        help='Tracer labels to include in the joint fit.')
    parser.add_argument('--region', default='GCcomb',
                        choices=['SGC', 'NGC', 'GCcomb'],
                        help='Sky region.')
    parser.add_argument('--k_max_p', type=float, default=0.201,
                        help='Maximum k [h/Mpc] for the power spectrum.')
    parser.add_argument('--no_emulator', action='store_true',
                        help='Disable the Taylor emulator (slower but exact).')
    parser.add_argument('--sampler', default='cobaya',
                        choices=['cobaya', 'emcee'],
                        help='Sampler backend.')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build configuration
    # ------------------------------------------------------------------
    cfg = FitConfig(
        tracers=args.tracers,
        region=args.region,
        k_max_p=args.k_max_p,
        use_emulator=not args.no_emulator,
        sampler=args.sampler,
    )
    logger.info('Chain output: %s', cfg.chain_name())

    # ------------------------------------------------------------------
    # Build the inference pipeline
    # Ordering matters: emulate theories *before* building observables so
    # that the observable holds a reference to the fast emulated calculator.
    # ------------------------------------------------------------------
    fiducial = DESI()
    cosmo    = setup_cosmo()

    theories = setup_theories(cfg, cosmo, fiducial)

    if cfg.use_emulator:
        theories = emulate_theories(theories)

    observables = setup_observables(cfg, theories)

    namespace_and_marginalise(cfg, theories)

    likelihood = build_likelihood(cfg, observables)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    if args.test:
        loglkl = likelihood()
        logger.info('log-likelihood = %.4f', loglkl)
        logger.info('Varied parameters: %s', likelihood.varied_params)
        logger.info('Test successful.')

    if args.plot_bestfit:
        plot_bestfit(cfg, likelihood, observables)

    if args.plot_chains:
        plot_chains(cfg)

    if args.run_chains:
        if cfg.sampler == 'cobaya':
            run_cobaya_sampler(likelihood, cfg)
        else:
            run_emcee_sampler(likelihood, cfg)


if __name__ == '__main__':
    main()
