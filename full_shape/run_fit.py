import os
from pathlib import Path
import functools
import argparse

from . import tools
from .tools import get_likelihood, fill_fiducial_options, generate_likelihood_options_helper


def run_fit_from_options(actions,
                         get_stats_fn=tools.get_stats_fn,
                         get_fits_fn=tools.get_fits_fn,
                         cache_dir=str | Path=None, **kwargs):
    if isinstance(actions, str):
        actions = [actions]
    options = fill_fiducial_options(kwargs)
    likelihoods_options = options['likelihoods']
    likelihood = get_likelihood(likelihoods_options, get_stats_fn=get_stats_fn, cache_dir=cache_dir)
    likelihood()
    for action in actions:
        if action == 'sample':
            from desilike.samplers import EmceeSampler
            samplers = {'emcee': EmceeSampler}
            sampler_options = dict(options['sample'])
            cls = sampler_options.pop('sampler', 'emcee')
            cls = samplers[cls]
            save_fn = [get_fits_fn(kind='chain', likelihoods=likelihoods_options, ichain=ichain)\
                       for ichain in range(sampler_options['nchains'])]
            sampler = cls(**options['init'], save_fn=save_fn)
            sampler.run(**options['run'])
        elif action == 'profile':
            from desilike.profilers import MinuitProfiler
            profilers = {'minuit': MinuitProfiler}
            options = dict(options['profile'])
            cls = options.pop('profiler', 'minuit')
            cls = profilers[cls]
            save_fn = get_fits_fn(kind='profiles', likelihoods=likelihoods_options)
            profiler = cls(**options['init'], save_fn=save_fn)
            profiler.maximize(**options['maximize'])
        else:
            raise NotImplementedError(f'{action} not implemented')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Fit DESI cutsky clustering statistics.',
    )
    parser.add_argument(
        '--action', type=str, default='sample',
        choices=['profile', 'sample'], nargs='*',
        help='Run best fit (maximize) and / or sample.',
    )
    parser.add_argument(
        '--stats', type=str, nargs='*', default='mesh2_spectrum',
        choices=['mesh2_spectrum', 'mesh3_spectrum'],
        help='Statistics to fit.',
    )
    parser.add_argument(
        '--tracers', nargs='+', type=str, default=['LRG2'],
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
    fits_dir = Path(os.getenv('SCRATCH')) / 'fits'
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
    options = {'likelihoods': []}
    for tracer in args.tracers:
        likelihood_options = generate_likelihood_options_helper(stat=args.stats, version=args.data, tracer=args.tracer, region=args.region,
                                                                covariance=args.covariance)
        options['likelihoods'].append(likelihood_options)
    run_fit_from_options(args.actions, get_fits_fn=functools.partial(tools.get_fits_fn, fits_dir=args.fits_dir), **options)