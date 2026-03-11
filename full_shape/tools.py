import os
import logging
from pathlib import Path

import numpy as np
from numpy import linalg
import lsstypes as types

from clustering_statistics.tools import (float2str, get_full_tracer, get_simple_tracer, _make_tuple,
                                         get_simple_stats, _unzip_catalog_options, setup_logging)
from clustering_statistics import tools as clustering_tools


logger = logging.getLogger('tools')


def get_cosmology():
    """Return a desilike Cosmoprimo calculator and the DESI fiducial cosmology.

    The calculator has free parameters h, omega_cdm, omega_b, logA with
    uniform priors; n_s and tau_reio are fixed.  A Gaussian prior is placed
    on omega_b from simulations.

    Returns
    -------
    cosmo : Cosmoprimo
    fiducial : cosmoprimo.Cosmology
    """
    from desilike.theories import Cosmoprimo
    from cosmoprimo.fiducial import DESI

    fiducial = DESI()
    cosmo = Cosmoprimo(engine='class', fiducial=fiducial)

    params = {
        'H0':       {'derived': True},
        'Omega_m':  {'derived': True},
        'sigma8_m': {'derived': True},
        'tau_reio': {'fixed': True},
        'n_s':      {'fixed': True},
        'omega_b':  {'fixed': False, 'prior': {'dist': 'norm',    'loc': 0.02237,  'scale': 0.00037}},
        'h':        {'fixed': False, 'prior': {'dist': 'uniform', 'limits': [0.5,  0.9]}},
        'omega_cdm':{'fixed': False, 'prior': {'dist': 'uniform', 'limits': [0.05, 0.2]}},
        'logA':     {'fixed': False, 'prior': {'dist': 'uniform', 'limits': [2.0,  4.0]}},
    }
    for name, config in params.items():
        cosmo.init.params[name].update(**config)
    return cosmo, fiducial



def _get_default_theory_nuisance_priors(model, stat, prior_basis, b3_coev=True, sigma8_fid=1.):
    """Build a dict of nuisance parameter prior configurations.

    Parameters
    ----------
    model : str
        PT model tag. When 'EFT', FoG parameters are fixed.
    stat : str
        Observable, ['mesh2_spectrum', 'mesh2_spectrum'].
    prior_basis : str
        'physical' or 'physical_aap' uses physical bias parameters
        (b1p, b2p,...).  Any other value uses the standard Eulerian basis
        (b1, b2, ...).
    b3_coev : bool
        Fix b3 to its co-evolution value.
    sigma8_fid : float or None
        Fiducial sigma_8(z_eff), used as prior centre in the physical basis.

    Returns
    -------
    params : dict[str, dict]
        Maps parameter name to a dict of keyword arguments accepted by
        ``Parameter.update()`` (e.g. ``{'fixed': True}`` or
        ``{'prior': {...}}``).
    """
    params = {}
    scale_eft = 12.5
    scale_sn0 = 2.0
    scale_sn2 = 5.0

    if prior_basis in ['physical', 'physical_aap', 'tcm_chudaykin_aap']:
        # ── Bias parameters ───────────────────────────────────────────────
        params['b1p'] = {'prior': {'dist': 'uniform', 'limits': [0.1, 4]}}
        params['b2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        params['bsp'] = {'prior': {'dist': 'norm', 'loc': -2. / 7. * sigma8_fid**2, 'scale': 5}}
        if b3_coev:
            params['b3p'] = {'fixed': True}
        else:
            params['b3p'] = {'prior': {'dist': 'norm', 'loc': 23. / 42. * sigma8_fid**4, 'scale': sigma8_fid**4},
                             'fixed': False}
        if 'mesh2_spectrum' in stat:
            # ── PS counter-terms and shot noise ───────────────────────────────
            for n in [0, 2, 4]:
                params[f'alpha{n:d}p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_eft}}
            params['sn0p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn0}}
            params['sn2p']  = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn2}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_pp'] = {'fixed': True}
            else:
                params['X_FoG_pp'] = {'prior': {'dist': 'uniform', 'limits': [0, 10]}}
        elif 'mesh3_spectrum' in stat:
            # ── BS stochastic parameters (only for bs / joint) ────────────────
            params['c1p']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
            params['c2p']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
            params['Pshotp'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
            params['Bshotp'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_bp'] = {'fixed': True}
            else:
                params['X_FoG_bp'] = {'prior': {'dist': 'uniform', 'limits': [0, 15]}}

    else:
        # ── Bias parameters (standard Eulerian basis) ─────────────────────
        params['b1'] = {'prior': {'dist': 'uniform', 'limits': [1e-5, 10]}}
        params['b2'] = {'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        params['bs'] = {'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        if b3_coev:
            params['b3'] = {'fixed': True}
        else:
            params['b3'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}, 'fixed': False}
        if 'mesh2_spectrum' in stat:
            # ── PS counter-terms and shot noise ───────────────────────────────
            for n in [0, 2, 4]:
                params[f'alpha{n:d}'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_eft}}
            params['sn0'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn0}}
            params['sn2']  = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn2}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_p'] = {'fixed': True}
            else:
                params['X_FoG_p'] = {'prior': {'dist': 'uniform', 'limits': [0, 10]}}
        elif 'mesh3_spectrum' in stat:
            # ── BS stochastic parameters (only for bs / joint) ────────────────
            shotnoise = 1 / 0.0002118763
            params['c1']    = {'prior': {'dist': 'norm', 'loc': 66.6, 'scale': 66.6 * 4}}
            params['c2']    = {'prior': {'dist': 'norm', 'loc': 0,    'scale': 4}}
            params['Pshot'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': shotnoise * 4}}
            params['Bshot'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': shotnoise * 4}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_bp'] = {'fixed': True}
            else:
                params['X_FoG_bp'] = {'prior': {'dist': 'uniform', 'limits': [0, 15]}}
    return params


def get_theory(stat: str, theory: dict, z: float, cosmo, fiducial):
    from desilike.theories.galaxy_clustering import (DirectPowerSpectrumTemplate, REPTVelocileptorsTracerPowerSpectrumMultipoles,
    FOLPSv2TracerPowerSpectrumMultipoles, FOLPSv2TracerBispectrumMultipoles)
    template = DirectPowerSpectrumTemplate(fiducial=fiducial, cosmo=cosmo, z=z)
    theory = None
    if 'mesh2_spectrum' in stat:
        if theory['model'] == 'reptvelocileptors':
            theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(template=template, **theory.get('options', {}))
        elif theory['model'] in ['folpsD', 'folpsEFT']:
            kw = {name: theory[name] for name in ['damping', 'prior_basis', 'b3_coev']}
            theory = FOLPSv2TracerPowerSpectrumMultipoles(template=template, **kw, **theory.get('options', {}))
            sigma8_fid = fiducial.get_fourier().sigma8(of='cb', z=z)
            params = _get_default_theory_nuisance_priors(theory['model'], stat, prior_basis=kw['prior_basis'], b3_coev=kw['b3_coev'], sigma8_fid=sigma8_fid)
            for name, config in params.items():
                theory.init.params[name].update(**config)
            if theory['marg']:
                for param in theory.init.params.select(basename=['alpha*', 'sn*']):
                    param.update(derived='.auto')
    elif 'mesh3_spectrum' in stat:
        if theory['model'] in ['folpsD', 'folpsEFT']:
            kw = {name: theory[name] for name in ['damping', 'prior_basis', 'b3_coev']}
            theory = FOLPSv2TracerBispectrumMultipoles(template=template, **kw, **theory.get('options', {}))
            sigma8_fid = fiducial.get_fourier().sigma8(of='cb', z=z)
            params = _get_default_theory_nuisance_priors(theory['model'], stat, prior_basis=kw['prior_basis'], b3_coev=kw['b3_coev'], sigma8_fid=sigma8_fid)
            for name, config in params.items():
                theory.init.params[name].update(**config)
    if theory is None:
        raise ValueError(f'theory not found for {stat} and {repr(theory)}')
    return theory


def pack_stats(stats, **labels):
    if isinstance(stats[0], types.ObservableLike):
        return types.ObservableTree(stats, **labels)
    elif isinstance(stats[0], types.WindowMatrix):
        windows = stats
        values = [window.value() for window in windows]
        observables = [window.observable for window in windows]
        theories = [window.theory for window in windows]
        return types.WindowMatrix(
            value=linalg.block_diag(*values),
            observable=pack_stats(observables, **labels),
            theory=pack_stats(theories, **labels),
        )
    else:
        raise ValueError(f'unrecognized type {stats[0]}')


def unpack_stats(stats):
    if isinstance(stats, types.ObservableLike):
        return stats.flatten(level=1)  # iter over labels
    elif isinstance(stats, types.WindowMatrix):
        window = stats
        windows = []
        for label in window.observable.labels(level=1):
            windows.append(window.at.observable.get(**label).at.theory.get(**label))
        return windows
    elif isinstance(stats, types.GaussianLikelihood):
        likelihood = stats
        return (unpack_stats(likelihood.observable), unpack_stats(likelihood.window), likelihood.covariance)


def get_stats(observables: list, covariance: dict=None, unpack: bool=False, get_stats_fn=clustering_tools.get_stats_fn, cache_dir: str | Path=None):

    observables_options = observables
    cache_fn = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        _str_from_options = str_from_likelihood_options({'observables': observables_options, 'covariance': covariance}, level={'catalog': 100, 'select': 100})
        cache_fn = cache_dir / 'prepared_stats' / f'{_str_from_options}.h5'
        if cache_fn.exists():
            likelihood = types.read(cache_fn)
            if unpack:
                return unpack_stats(likelihood)
            return likelihood

    # Helper: iterate over (stat, tracer) combinations
    def iter_stat_tracer_combinations(observables_options, **kwargs):
        """
        Yield (key, labels, file_kwargs) for every (stat, tracer) combination.

        key        : (stat, tracer) — identifies the combination.
        labels     : {'observables': ..., 'tracers': ...} for ObservableTree.
        file_kwargs: keyword arguments to pass to get_stats_fn.
        """
        for observable_options in observables_options:
            stat = observable_options['stat']
            tracers = _make_tuple(observable_options['catalog']['tracer'])
            version = observable_options['catalog'].get('version', None)
            for tracer in tracers:
                full_tracer = get_full_tracer(tracer, version=version)
                n_fields = 3 if 'mesh3' in stat else 2
                simple_tracers = get_simple_tracer(tracer)
                simple_tracers += (simple_tracers[-1],) * (n_fields - len(simple_tracers))
                labels = {
                    'observables': get_simple_stats(stat),
                    'tracers': simple_tracers,
                }
                file_kw = observable_options['catalog'] | {'tracer': full_tracer} | kwargs
                yield stat, labels, file_kw, dict(observable_options)

    def _apply_select(observable: types.ObservableTree, select: dict=None):
        if select is None:
            return observable
        for ell, limit in select.items():
            pole = observable.get(ells=ell)
            coord_name = list(pole.coords())[0]
            if len(limit) == 3:
                step = limit[2]
                edge = pole.edges(coord_name)[0]
                rebin = int(np.rint(np.mean(step / (edge[..., 1] - edge[..., 0]))) + 0.5)
                observable = observable.at(ells=ell).select(**{coord_name: slice(0, None, rebin)})
            observable = observable.at(ells=ell).select(**{coord_name: limit[:2]})
        return observable

    # Loading data, window
    loaded_data, loaded_window, joint_labels = [], [], {'observables': [], 'tracers': []}
    for stat, labels, file_kw, kw in iter_stat_tracer_combinations(observables_options):
        fn = get_stats_fn(kind=stat, **file_kw)
        if isinstance(fn, list):
            fns = fn
            mocks = [types.read(fn) for fn in fns if fn.exists()]
            data = types.mean(mocks)
        else:
            data = types.read(fn)
        data = _apply_select(data, select=kw.get('select', None))
        for field, value in labels.items():
            joint_labels[field].append(value)
        file_kw = dict(file_kw)
        imock = file_kw.get('imock', None)
        if imock is not None:
            file_kw['imock'] = 0
        fn = get_stats_fn(kind=f'window_{stat}', **file_kw)
        window = types.read(fn).at.observable.match(data)
        loaded_data.append(data)
        loaded_window.append(window)

    data = pack_stats(loaded_data, **joint_labels)
    window = pack_stats(loaded_window, joint_labels)

    # Mock-based covariance
    all_fns = []
    for stat, labels, file_kw, kw in iter_stat_tracer_combinations(observables_options):
        file_kw = file_kw | covariance
        file_kw['tracer'] = get_full_tracer(file_kw['tracer'], version=file_kw['version'])
        file_kw.setdefault('imock', '*')
        all_fns.append(get_stats_fn(kind=stat, **file_kw))
    all_fns = list(zip(*all_fns, strict=True))  # get a list of list of file names
    mocks = []
    for fns in all_fns:
        if all(fn.exists() for fn in fns):
            mock = types.ObservableTree([types.read(fn) for fn in fns], **joint_labels)
            mocks.append(mock)
    covariance = types.cov(mocks)
    covariance = covariance.at.observable.match(data)
    likelihood = types.GaussianLikelihood(
        observable=data,
        window=window,
        covariance=covariance,
    )
    if cache_fn is not None:
        likelihood.write(cache_fn)
    if unpack:
        return unpack_stats(likelihood)
    return likelihood


def get_single_likelihood(likelihood_options, stats: types.GaussianLikelihood=None,
                          cosmo=None, fiducial=None, get_stats_fn=clustering_tools.get_stats_fn, cache_dir:str | Path=None):
    from desilike.observables.galaxy_clustering import TracerSpectrum2PolesObservable, TracerSpectrum3PolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    # likelihood_options: {'observables': [observable_options], 'covariance': {}}
    observables_options = likelihood_options['observables']
    covariance = likelihood_options.get('covariance', {})
    if cosmo is None:
        cosmo, fiducial = get_cosmology()
    if stats is None:
        stats = get_stats(observables_options, covariance=covariance, unpack=False, get_stats_fn=get_stats_fn, cache_dir=cache_dir)
    data, windows, covariance = unpack_stats(stats)
    labels = covariance.observable.labels(level=1)
    observables = []
    for observable_options, data, window, label in zip(observables_options, data, windows, labels, strict=True):
        stat = observable_options['stat']
        if 'mesh2_spectrum' in stat:
            cls = TracerSpectrum2PolesObservable
        elif 'mesh3_spectrum' in stat:
            cls = TracerSpectrum3PolesObservable
        else:
            raise NotImplementedError(stat)
        for label, pole in window.observable.items(level=None):
            z = pole.attrs['zeff']
        theory = get_theory(stat, theory=observable_options['theory'], z=z, cosmo=cosmo, fiducial=fiducial)
        theory_params = theory.init.params
        observable = cls(data=data, window=window, theory=theory)
        observable()
        if observable_options['emulator'] is not None:
            assert cache_dir is not None, 'cache_dir must be provided for emulator'
            filename = cache_dir / _str_from_observable_options(observable_options, level={'theory': 100, 'catalog': 2}) / 'emulator.npy'
            filename.parent.mkdir(parents=True, exist_ok=True)
            from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine
            if filename.exists():
                logger.info(f'Loading emulator {filename}')
                emulated_pt = EmulatedCalculator.load(filename)
            else:
                logger.info(f'Fitting emulator {filename}')
                emulator = Emulator(
                    theory.pt,
                    engine=TaylorEmulatorEngine(method='finite', order=observable_options['emulate'].get('order', 3)),
                )
                emulator.set_samples()
                emulator.fit()
                emulated_pt = emulator.to_calculator()
                emulated_pt.save(filename)
            theory.init.update(pt=emulated_pt)
            theory.init.params.update(theory_params)
        observables.append(observable)
    return ObservablesGaussianLikelihood(observables, covariance=covariance.value())


def get_likelihood(likelihoods_options, cosmo=None, fiducial=None, get_stats_fn=clustering_tools.get_stats_fn, cache_dir:str | Path=None):
    """
    Build a desilike :class:`SumLikelihood, summed over all tracers.

    Parameters
    ----------
    likelihoods_options : list
        List of options {'observables': [observable_options, ...], 'covariance': {}}

    Returns
    -------
    SumLikelihood
    """
    from desilike.likelihoods import SumLikelihood
    if cosmo is None:
        cosmo, fiducial = get_cosmology()
    if isinstance(likelihoods_options, dict):
        likelihoods_options = [likelihoods_options]
    likelihoods = [get_single_likelihood(likelihood_options,
                  cosmo=cosmo, fiducial=fiducial, get_stats_fn=get_stats_fn, cache_dir=cache_dir) for likelihood_options in likelihoods_options]
    return SumLikelihood(likelihoods)


def propose_fiducial_observable_options(stat, tracer=None, zrange=None):
    """
    Propose fiducial fitting options for given tracer and input kind.
    """
    propose_select = {'mesh2_spectrum': {0: [0.02, 0.2, 0.005], 2: [0.02, 0.2, 0.005]},
                      'mesh3_spectrum': {(0, 0, 0): [0.02, 0.12, 0.005],
                                        (2, 0, 2): [0.02, 0.08, 0.005]},
                                        'basis': 'sugiyama-diagonal'}
    select = None
    for _stat in propose_select:
        if _stat in stat:
            select = propose_select[_stat]
            break
    propose_fiducial = {'stat': stat,
                        'select': select,
                        'catalog': {'weight': 'default-FKP'},
                        'theory': {'model': 'folpsD', 'prior_basis': 'physical_aap', 'damping': 'lor', 'marg': True},
                        'emulator': {},
                        'window': {}}
    return propose_fiducial


def propose_fiducial_covariance_options():
    return {'version': 'holi-v1-altmtl'}


def propose_fiducial_sampler_options(sampler=None):
    if sampler is None:
        sampler = 'emcee'
    fiducial_options = {'sampler': sampler, 'init': {},' run': {}, 'nchains': 4}
    return fiducial_options


def propose_fiducial_profiler_options(profiler=None):
    if profiler is None:
        profiler = 'minuit'
    fiducial_options = {'profiler': profiler, 'init': {},' run': {}}
    return fiducial_options


def fill_fiducial_observable_options(options):
    """Fill missing options with fiducial values."""
    options = dict(options)
    stat = options['stat']
    tracer, zrange = (options['catalog'][name] for name in ['tracer', 'zrange'])
    fiducial_options = propose_fiducial_observable_options(stat, tracer, zrange)
    options = fiducial_options | options
    for key, value in options.items():
        if key != 'stat':
            options[key] = fiducial_options[key] | value
    return options


def fill_fiducial_likelihood_options(options):
    if isinstance(options, dict):
        options = dict(options)
        options['observables'] = [fill_fiducial_observable_options(options) for options in options['observables']]
        options['covariance'] = propose_fiducial_covariance_options() | (options.get('covariance', {}) or {})
        return options
    return type(options)(fill_fiducial_likelihood_options(opts) for opts in options)


def fill_fiducial_options(options):
    options = dict(options)
    likelihoods = options.get('likelihoods', None)
    if likelihoods is not None:
        options['likelihoods'] = fill_fiducial_likelihood_options(options['likelihoods'])
    name = 'sampler'
    options[name] = propose_fiducial_sampler_options(options[name].get('sampler')) | options[name]
    name = 'profiler'
    options[name] = propose_fiducial_profiler_options(options[name].get('profiler')) | options[name]
    return options


def generate_likelihood_options_helper(stats=('mesh2_spectrum', 'mesh3_spectrum'),
                                       tracer='LRG', zrange=(0.4, 0.6), region='GCcomb',
                                       version='abacus-2ndgen-complete', covariance='holi-v1-altmtl'):
    observables = []
    tracer, zrange = get_full_tracer_zrange(tracer)
    for stat in stats:
        observables.append({'stat': stat,
                            'catalog': {'version': version, 'tracer': tracer, 'zrange': zrange, 'region': region}})
    covariance = {'version': covariance}
    return fill_fiducial_likelihood_options({'observables': observables, 'covariance': covariance})


def get_full_tracer_zrange(tracerz=None, zrange=None):
    translate_zrange = {'BGS1': (0.1, 0.4),
                        'LRG1': (0.4, 0.6), 'LRG2': (0.6, 0.8), 'LRG3': (0.8, 1.1),
                        'ELG1': (0.8, 1.1), 'ELG2': (1.1, 1.6),
                        'QSO1': (0.8, 2.1)}
    if tracerz is None:
        return translate_zrange

    def _get_full_tracer_zrange(tracerz, zrange=zrange):
        if 'x' in tracerz:
            return list(zip(*[_get_full_tracer_zrange(t, zrange=zrange) for t in tracerz.split('x')]))
        if tracerz in translate_zrange:
            # Return tracer and z-range from translate_zrange
            tracer = tracerz[:-1]
            zrange = translate_zrange[tracerz]
        else:
            # Not in translate_zrange
            tracer = tracerz
        return tracer, zrange

    if isinstance(tracerz, str):
        return _get_full_tracer_zrange(tracerz)
    else:  # tuple/list of tracers
        return type(tracerz)(zip(*map(_get_full_tracer_zrange, tracerz)))


def _get_level(level: int | dict=None):
    _default_level = {'stat': 1, 'select': 0, 'catalog': 1, 'theory': 0, 'covariance': 0}
    if level is None: level = {}
    if not isinstance(level, dict):
        level = {name: level for name in _default_level}
    level = _default_level | level
    return level


def _str_from_observable_options(options: dict, level: int=None) -> str:
    """
    Return string given input observable options, with ``level`` of details.
    """
    level = _get_level(level)
    out_str = []

    # First, catalog
    catalog = _unzip_catalog_options(options['catalog'])

    def _str_zrange(zrange):
        return f'z{float2str(zrange[0], prec_min=1, prec_max=5)}-{float2str(zrange[1], prec_min=1, prec_max=5)}'

    if level['catalog'] >= 1:
        translate_tracerz = get_full_tracer_zrange(tracerz=None)
        catalog_str = []
        for tracer in catalog:
            stracer = get_simple_tracer(tracer)
            catalog_options = catalog[tracer]
            found = False
            for tracerz, zrange in translate_tracerz.items():
                if tracerz.startswith(stracer) and np.allclose(catalog_options['zrange'], zrange):
                    stracer = tracerz  # e.g. LRG1
                    found = True
                    break
            tracer_catalog_str = [stracer]
            if not found or level['catalog'] >= 2:
                tracer_catalog_str.append(_str_zrange(catalog_options['zrange']))
            if level['catalog'] >= 3:
                tracer_catalog_str.append(catalog_options['region'])
            if level['catalog'] >= 4:
                tracer_catalog_str.append('weight-' + catalog_options['weight'])
            catalog_str.append('-'.join(tracer_catalog_str))
        out_str.append('x'.join(catalog_str))

    # Then, stat and select, e.g. S2-ell0-0.02-0.2-ell2-0.02-0.2
    translate_stat_name = {'S2': ['mesh2_spectrum'],
                      'S3': ['mesh3_spectrum'],
                      'BAOR': ['bao', 'recon'],
                      'C2R': ['particle2_correlation', 'recon']}
    stat = options['stat']
    # Then, stat and select, e.g. S2-ell0-0.02-0.2-ell2-0.02-0.2
    if level['stat'] >= 1:
        found = None
        for name in translate_stat_name:
            if all(t in stat for t in translate_stat_name[name]):
                found = name
                break
        if found is None:
            raise ValueError(f'could not find shot naame for {stat}')
        out_str.append(found)
    if level['select'] >= 1:
        select_str = []
        for ell, limits in options.get('select', {}).items():
            if isinstance(limits, (list, tuple)):
                if isinstance(ell, (list, tuple)):
                    ell = ''.join([str(ell) for ell in ell])
                else:
                    ell = str(ell)
                prec = dict(prec_min=2, prec_max=3) if name.startswith('S') else dict(prec_min=0, prec_max=0)
                select_str.append(f'ell{ell}-' + '-'.join(float2str(lim, **prec) for lim in limits))
        select_str = '-'.join(select_str)
        out_str.append(select_str)

    if level['theory'] > 0:
        out_str.append('th')
        out_str.append(options['theory']['model'])

    return '-'.join(out_str)


def str_from_likelihood_options(likelihood_options, level: int=None):
    level = _get_level(level)
    out_str = []
    for options in likelihood_options['observables']:
        out_str.append(_str_from_observable_options(options, level=level))
    if level['covariance'] > 0:
        out_str.append('cov-' + likelihood_options['covariance']['version'])
    return '+'.join(out_str)


def get_fits_fn(fits_dir=Path(os.getenv('SCRATCH', '.')) / 'fits', kind='chain', likelihoods: list=None,
                sampler: dict=None, profiler: dict=None, ichain: int=None,
                level=None, extra='', ext='npy'):
    """
    Return measurement filename for given parameters.

    Parameters
    ----------
    fits_dir : str, Path
        Directory containing the measurements.
    kind : str
        Fitting product. Options are 'chain', 'profiles', etc.
    extra : str, optional
        Extra string to append to file name.
    ext : str
        File extension. Default is 'npy'.

    Returns
    -------
    fn : Path
        Fit file name.
    """
    fits_dir = Path(fits_dir)
    _str_from_options = [str_from_likelihood_options(likelihood_options, level=level) for likelihood_options in likelihoods]
    _str_from_options = '_'.join(_str_from_options)
    extra = f'_{extra}' if extra else ''
    ichain = '_{ichain:d}' if ichain is not None else ''
    return fits_dir / f'{_str_from_options}{extra}' / f'{kind}{ichain}.{ext}'