import os
from pathlib import Path
import functools
import argparse

from desilike.base import compile, Prior, Posterior, params as get_params
from desilike.distributed import get_mpicomm

from full_shape.tools import get_likelihood, fill_fiducial_options, generate_likelihood_options_helper, setup_logging
from full_shape import tools
from clustering_statistics import tools as clustering_tools


def print_priors(calculator, varied=True, **kwargs):
    print(f"{'param':20} {'prior':50} {'reference':50} derived")
    for p in get_params(calculator).select(varied=varied, **kwargs):
        print(f"{p.name:20} {str(p.prior):50} {str(p.ref):50} {p.derived}")


def run_fit_from_options(actions,
                         get_stats_fn=clustering_tools.get_stats_fn,
                         get_fits_fn=tools.get_fits_fn,
                         cache_dir:str | Path=None, cache_mode: str='rw', **kwargs):
    """
    Build a likelihood from options and run fitting actions (profile / sample).

    This helper constructs desi-like likelihood(s) from provided options,
    instantiates the corresponding desilike :class:`ObservablesGaussianLikelihood`
    and then runs fitting.

    Parameters
    ----------
    actions : str or sequence[str]
        One or more actions to run. Supported values: 'profile' (maximize using a
        profiler) and 'sample' (run MCMC sampler).
    get_stats_fn : callable, optional
        Function used to locate/read measurement files (passed to likelihood builder).
    get_fits_fn : callable, optional
        Function that constructs file paths for fit outputs (used to name saved chains/profiles).
    cache_dir : str or pathlib.Path, optional
        Directory used for caching emulators and precomputed products.
    cache_mode : str, optional
        'rw' for read/write; 'r' for read-only.
    **kwargs :
        Top-level options dictionary consumed by fill_fiducial_options. Must include
        a 'likelihoods' entry; may include sampler/profiler configuration and init/run kwargs.

    """
    if isinstance(actions, str):
        actions = [actions]
    options = fill_fiducial_options(kwargs)
    likelihood = get_likelihood(likelihoods_options=options['likelihoods'],
                                cosmology_options=options['cosmology'],
                                get_stats_fn=get_stats_fn, cache_dir=cache_dir, cache_mode=cache_mode)
    mpicomm = get_mpicomm()
    if mpicomm.rank == 0:
        print('priors:')
        print_priors(likelihood)
    fn = get_fits_fn(kind='config', **options, ext='yaml')
    tools.write_options(fn, options)
    for action in actions:
        if action == 'build':
            pass  # likelihood already built and cached above
        elif action == 'sample':
            sampler_options = dict(options['sampler'])
            cls = tools.get_sampler_cls(sampler_options.pop('sampler', 'emcee'))
            nchains = sampler_options.pop('nchains', 1)
            directory = get_fits_fn(kind='samples', **options).parent
            posterior = compile(Posterior(likelihood))
            sampler = cls(posterior, nchains=nchains, directory=directory,
                          **sampler_options.get('init', {}))
            sampler.run(**sampler_options.get('run', {}))
        elif action == 'profile':
            profiler_options = dict(options['profiler'])
            cls = tools.get_profiler_cls(profiler_options.pop('profiler', 'minuit'))
            save_fn = get_fits_fn(kind='profiles', **options)
            from desilike.base import copy
            likelihood_profiler = copy(likelihood)
            for param in get_params(likelihood_profiler).select(solved=True):
                param.update(derived='best')
            posterior_profiler = compile(Posterior(likelihood_profiler))
            profiler = cls(posterior_profiler, save_fn=save_fn, **profiler_options.get('init', {}))
            profiler.maximize(**profiler_options.get('maximize', {}))
            if mpicomm.rank == 0:
                print(profiler.profiles.to_stats(tablefmt='pretty'))
        else:
            raise NotImplementedError(f'{action} not implemented')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Fit DESI cutsky clustering statistics.',
    )
    parser.add_argument(
        '--actions', type=str, default='profile',
        choices=['build', 'profile', 'sample'], nargs='*',
        help='Run best fit (maximize) and / or sample. Use "build" to pre-warm caches without running inference.',
    )
    parser.add_argument(
        '--stats', type=str, nargs='*', default=['mesh2_spectrum'],
        choices=['mesh2_spectrum', 'mesh3_spectrum'],
        help='Statistics to fit.',
    )
    parser.add_argument(
        '--tracers', nargs='*', type=str, default=['LRG2'],
        help='Tracer labels (default: LRG2).',
    )
    parser.add_argument(
        '--region', type=str, default='GCcomb',
        help='Sky region (default: GCcomb).',
    )
    parser.add_argument(
        '--data', type=str, default='abacus-2ndgen-complete',
        help='Data product identifier (default: abacus-2ndgen-complete).',
    )
    parser.add_argument(
        '--covariance', type=str, default='holi-v1-altmtl',
        help='Covariance mock set (default: holi-v1-altmtl).',
    )
    parser.add_argument(
        '--project', type=str, default='',
        help='Optional measurement project subdirectory under stats_dir.',
    )
    fits_dir = Path(os.getenv('SCRATCH', '.')) / 'fits'
    parser.add_argument(
        '--fits_dir', type=str, default=fits_dir,
        help=f'base directory for fits, default is {fits_dir}'
    )
    cache_dir = Path('.') / '_cache'
    parser.add_argument(
        '--cache_dir', type=str, default=cache_dir,
        help=f'cache directory for emulators and pre-computed covariance, default is {cache_dir}'
    )
    args = parser.parse_args()

    setup_logging()
    options = {'likelihoods': []}
    for tracer in args.tracers:
        likelihood_options = generate_likelihood_options_helper(stats=args.stats, version=args.data, tracer=tracer, region=args.region,
                                                                covariance=args.covariance, project=args.project)
        options['likelihoods'].append(likelihood_options)
    run_fit_from_options(args.actions,
                         get_fits_fn=functools.partial(tools.get_fits_fn, fits_dir=args.fits_dir),
                         cache_dir=args.cache_dir, **options)
