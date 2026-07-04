"""
High-level orchestration for cutsky clustering measurements.

This module provides the main CLI entry point (`clustering-stats`) and the
pipeline driver used for DESI lightcone clustering statistics.

Main functions
--------------
* `compute_stats_from_options`, which takes as input a list of summary statistics to compute and a dictionary of options,
and orchestrates the workflow:
- fill fiducial defaults
- read clustering catalogs and randoms
- optionally run reconstruction
- dispatch to statistic-specific backends, such as `compute_mesh2_spectrum` for power spectrum measurement or `compute_particle2_correlation` for correlation function measurement.
* `postprocess_stats_from_options`, which can be used to run postprocessing steps,
such as combining measurements from different regions or computing rotation matrices for the power spectrum.
"""

import os
import logging
import functools
import copy
import warnings
from pathlib import Path
import itertools

import numpy as np
import jax
import jax.experimental.multihost_utils
import lsstypes as types

from . import tools
from .tools import fill_fiducial_options, _merge_options, Catalog, _compute_binned_weight, setup_logging
from . import catalog_blinding

from .correlation2_tools import compute_particle2_angular_upweights, compute_particle2_correlation, compute_particle2_correlation_close_pair_correction, compute_covariance_particle2_correlation
from .spectrum2_tools import (compute_mesh2_spectrum, compute_mesh2_spectrum_close_pair_correction, compute_window_mesh2_spectrum, compute_covariance_mesh2_spectrum, run_preliminary_fit_mesh2_spectrum, compute_rotation_mesh2_spectrum, compute_window_mesh2_spectrum_fm)

from .correlation3_tools import compute_particle3_angular_upweights, compute_particle3_correlation, compute_particle3_correlation_close_pair_correction
from .spectrum3_tools import (compute_mesh3_spectrum, compute_window_mesh3_spectrum, compute_mesh3_spectrum_close_pair_correction,
                              compute_covariance_mesh3_spectrum, run_preliminary_fit_mesh3_spectrum)

from .recon_tools import compute_reconstruction
from .systematic_templates import include_systematic_templates


logger = logging.getLogger('summary-statistics')


def _expand_cut_auw_options(stat, options):
    # Helper to generate separate option dictionaries for raw, theta-cut, and angular upweight variants
    # For spectrum measurements, create variants with different options
    if 'spectrum' in stat or 'correlation' in stat:
        keys = ['cut', 'auw']
        kw = dict(options)
        for key in keys: kw.pop(key, None)
        args = {'raw': kw}
        # Generate options for each variant (cut or auw)
        for key in keys:
            kw = dict(options)
            if not kw.get(key, False):
                continue
            else:
                # Keep only the current variant, remove others
                for name in keys:
                    if name != key: kw.pop(name, None)  # keep only if spectrum is with cut (resp. auw)
                args[key] = kw
    else:
        # For non-spectrum stats, use single dictionary
        args = {'stat': options}
    return args


def _make_list_zrange(zranges):
    # Convert zrange to list of tuples
    if np.ndim(zranges[0]) == 0:
        zranges = [zranges]
    return list(zranges)


def compute_stats_from_options(stats, analysis='full_shape', cache=None,
                               get_stats_fn=tools.get_stats_fn,
                               get_catalog_fn=None,
                               read_catalog=tools.read_catalog,
                               prepare_catalog=tools.prepare_catalog,
                               mask_catalog=tools.mask_catalog,
                               **kwargs):
    """
    Compute summary statistics based on the provided options.

    Parameters
    ----------
    stats : str or list of str
        Summary statistics to compute.
        Choices: ['mesh2_spectrum', 'mesh3_spectrum', 'particle2_correlation', 'recon_particle2_correlation', 'particle3_correlation', 'recon_particle3_correlation', 'close_pair_correction', 'window_mesh2_spectrum', 'window_mesh3_spectrum'].
        If 'close_pair_correction', add angular upweight or theta-cut correction to pre-computed standard 'mesh2_spectrum', 'mesh3_spectrum', 'particle2_correlation', 'particle3_correlation'.
    analysis : str, optional
        Type of analysis, 'full_shape' or 'png_local', to set fiducial options.
    cache : dict, optional
        Cache to store intermediate results (binning class and parent/reference random catalog).
        See :func:`spectrum2_tools.compute_mesh2_spectrum`, :func:`spectrum3_tools.compute_mesh3_spectrum`,
        and func:`tools.read_catalog` for details.
    get_stats_fn : callable, optional
        Function to get the filename for storing the measurement.
    get_catalog_fn : callable, optional
        Function to get the filename for reading the catalog.
        If provided, it is given to ``read_catalog``.
    read_catalog : callable, optional
        Function to read the catalog.
    prepare_catalog : callable, optional
        Function to prepare the clustering ('data', 'randoms') or 'full' ('fibered_data', 'parent_data', 'fibered_randoms', 'parent_randoms') catalogs.
    **kwargs : dict
        Options for catalog, reconstruction, and summary statistics.
    """
    # Ensure stats is a list (handle both string and list inputs)
    if isinstance(stats, str):
        stats = [stats]

    cache = cache or {}
    # Fill in fiducial defaults for all options
    options = fill_fiducial_options(kwargs, analysis=analysis)
    catalog_options = options['catalog']
    # tracers is a list of tracer1, tracer2, ... for cross-correlations
    tracers = list(catalog_options.keys())

    # Create redshift range lists for each tracer (support multiple z-bins)
    zranges = {tracer: _make_list_zrange(catalog_options[tracer].pop('zrange')) for tracer in tracers}
    region = {tracer: catalog_options[tracer].get('region') for tracer in tracers}
    catalog_blinding_options = {tracer: catalog_options[tracer].pop('blinding', None) for tracer in tracers}
    with_catalog_blinding = any(catalog_blinding.get_blinding_modes(catalog_blinding_options[tracer]) for tracer in tracers)
    catalog_blinding_params = {
        tracer: catalog_blinding.get_blinding_parameters(catalog_blinding_options[tracer], tracer=tracer)
        for tracer in tracers if catalog_blinding_options[tracer]
    }
    output_catalog_options = copy.deepcopy(catalog_options)
    for tracer, params in catalog_blinding_params.items():
        output_catalog_options[tracer]['version'] = catalog_blinding.output_version(output_catalog_options[tracer].get('version', None), params)
    catalog_blinding_attrs = catalog_blinding.blinding_attrs(catalog_blinding_params)

    def write_stats(filename, stats):
        if catalog_blinding_attrs:
            getattr(stats, 'attrs', {}).update(catalog_blinding_attrs)
        tools.write_stats(filename, stats)

    # Wrap catalog readers with catalog filename lookup function
    if get_catalog_fn is not None:
        read_catalog = functools.partial(read_catalog, get_catalog_fn=get_catalog_fn)
        prepare_catalog = functools.partial(prepare_catalog, mask_catalog=mask_catalog)

    # Check if any statistic requires reconstruction
    with_recon = any('recon' in stat and 'covariance' not in stat for stat in stats)
    with_catalogs = True

    # Initialize catalogs and randoms dictionaries
    data, randoms, raw_randoms, raw_full_data = {}, {}, {}, {}
    with_stats_blinding = False
    if with_catalogs:
        # Load data and random catalogs for each tracer
        for tracer in tracers:
            _catalog_options = dict(catalog_options[tracer])
            _catalog_options['region'] = 'ALL'
            # Expand redshift range to cover all requested z-bins

            # Add bitwise weight information (PIP, completeness) if needed
            binned_weight = {}
            if any(name in _catalog_options.get('weight', '') for name in ['bitwise', 'compntile']):
                # sets NTILE-MISSING-POWER (missing_power) and per-tile completeness (completeness)
                raw_full_data[tracer] = read_catalog(kind='full_data', **_catalog_options)
                raw_full_data[tracer] = catalog_blinding.apply_bao_blinding_to_catalogs(raw_full_data[tracer], catalog_blinding_params.get(tracer, None))
                binned_weight.update(raw_full_data[tracer].attrs)
            _catalog_options['binned_weight'] = binned_weight

            # Add reconstruction options if needed
            if with_recon:
                recon_options = options['recon'][tracer]
                # pop as we don't need it anymore
                _catalog_options |= {key: recon_options.pop(key) for key in list(recon_options) if key in ['nran']}

            # Check if analysis requires blinding (e.g., protected samples)
            if not with_catalog_blinding:
                with_stats_blinding |= tools.check_if_stats_requires_blinding(analysis=analysis, **_catalog_options)
            # Prepare incomplete catalog handling if completeness weights provided
            if isinstance(_catalog_options.get('complete', None), dict):
                _catalog_options.setdefault('reshuffle', {})  # to pass on complete data

            # Read data and random catalogs
            data[tracer] = prepare_catalog(read_catalog(kind='data', **_catalog_options, concatenate=True), kind='data', **(_catalog_options | dict(keep_columns=True)))
            binned_weight.update(data[tracer].attrs)  # update with any additional info from prepared data catalog
            #_catalog_options.pop('complete', None)
            #_catalog_options.pop('reshuffle', None)
            raw_randoms[tracer] = read_catalog(kind='randoms', **_catalog_options, concatenate=False)
            raw_randoms[tracer] = catalog_blinding.apply_bao_blinding_to_catalogs(raw_randoms[tracer], catalog_blinding_params.get(tracer, None))
            randoms[tracer] = prepare_catalog(raw_randoms[tracer], kind='randoms', **_catalog_options)
            if tools.check_if_requires_renormalization(**_catalog_options):
                for random in randoms[tracer]:
                    tools.renormalize_randoms_over_data(random, data[tracer], tracer=tracer)
            catalog_options[tracer]['binned_weight'] = binned_weight  # store binned weight info in catalog options for later use in stats computation
            output_catalog_options[tracer]['binned_weight'] = binned_weight

    # Warn user if blinding will be applied
    if with_stats_blinding:
        warnings.warn('Output clustering statistics will be blinded on-the-fly.\nIf you do not want blinding, pass "protected" in the "analysis" argument.')

    # Initialize reconstruction attributes storage
    stat_recon_attrs = {}
    if with_recon:
        # data_rec, randoms_rec = {}, {}
        stat_recon_attrs = {'recon_mode': [], 'recon_smoothing_radius': []}
        for tracer in tracers:
            recon_options = dict(options['recon'][tracer])
            recon_options.pop('zrange', None)  # not a kwarg of compute_reconstruction
            # Store reconstruction mode and radius for each tracer
            for name in stat_recon_attrs:
                stat_recon_attrs[name].append(recon_options[name[len('recon_'):]])
            # Run reconstruction to get shifted positions
            data[tracer]['POSITION_REC'], randoms_rec_positions = compute_reconstruction(lambda: {'data': data[tracer], 'randoms': Catalog.concatenate(randoms[tracer])}, **recon_options)

            # Assign reconstructed positions to random catalogs
            start = 0
            for random in randoms[tracer]:
                size = len(random['POSITION'])
                random['POSITION_REC'] = randoms_rec_positions[start:start + size]
                start += size
            # Keep only the requested number of random files (for reduced memory footprint)
            randoms[tracer] = randoms[tracer][:catalog_options[tracer]['nran']]  # keep only relevant random files

    # Loop over all requested redshift bins
    for zvals in zip(*(zranges[tracer] for tracer in tracers)):
        zrange = dict(zip(tracers, zvals))
        zdata, zrandoms = {}, {}
        if with_catalogs:
            # Slice catalogs to current redshift bin
            for tracer in tracers:
                zdata[tracer] = mask_catalog(data[tracer], 'data', region=region[tracer], zrange=zrange[tracer])
                zrandoms[tracer] = [mask_catalog(random, 'randoms', region=region[tracer], zrange=zrange[tracer]) for random in randoms[tracer]]
        fn_catalog_options = {tracer: output_catalog_options[tracer] | dict(zrange=zrange[tracer]) for tracer in tracers}

        # Compute angular upweights for fiber collision corrections if requested
        auw_options = {}
        funcs = {2: compute_particle2_angular_upweights, 3: compute_particle3_angular_upweights}
        for npt, func in funcs.items():
            # Extract selection weights if provided (e.g., NX**(-1. / 3.) weighting)
            # FIXME: how to generalize to any stat (correlation) or OQE weights?
            spectrum_options = options[f'mesh{npt:d}_spectrum']
            selection_weights = spectrum_options.get('selection_weights', None)
            _cache_auw = {}

            def get_data(tracer):
                # Load full parent catalogs (before any selection) for AUW computation
                _catalog_options = dict(fn_catalog_options[tracer])
                _zdata = zdata[tracer]
                if selection_weights:
                    _zdata = selection_weights[tracer](_zdata)

                _catalog_options['binned_weight']['weight_ntile'] = {column: _compute_binned_weight(_zdata[column], _zdata['INDWEIGHT'] / _zdata['WEIGHT_COMP'], mpicomm=_zdata.mpicomm) for column in ['NTILE']}
                del _zdata
                toret = {}
                for kind in ['fibered_data', 'parent_data'] + (['fibered_randoms', 'parent_randoms'] if npt > 2 else []):
                    if kind not in _cache_auw:
                        if tracer not in raw_full_data:
                            raw_full_data[tracer] = read_catalog(kind='full_data', **_catalog_options, concatenate=True)
                            #_catalog_options['binned_weight'].update(raw_full_data[tracer].attrs)  # update binned weight info for AUW computation
                        if 'randoms' in kind:
                            _cache_auw[kind] = prepare_catalog(raw_randoms[tracer], kind=kind, **_catalog_options)
                            if tools.check_if_requires_renormalization(**_catalog_options):
                                for random in _cache_auw[kind]:
                                    tools.renormalize_randoms_over_data(random, _cache_auw[kind.replace('randoms', 'data')], tracer=tracer)
                        else:
                            _cache_auw[kind] = prepare_catalog(raw_full_data[tracer], kind=kind, **_catalog_options)

                    toret[kind] = _cache_auw[kind]
                return toret

            stats_npt = [stat for stat in stats if any(name in stat for name in [f'mesh{npt:d}', f'particle{npt:d}']) and options[stat].get('auw', False)]
            if any(stats_npt):
                # Compute angular upweights from fibered vs parent catalogs
                fn = get_stats_fn(kind=f'particle{npt:d}_angular_upweights', catalog=fn_catalog_options)
                if False: #fn.exists():
                    auw = types.read(fn)
                else:
                    auw = func(*[functools.partial(get_data, tracer) for tracer in tracers])
                    # Write computed angular upweights to disk
                    write_stats(fn, auw)
                # Update all statistics options with computed angular upweights
                for stat in stats_npt:
                    auw_options[stat] = auw  # update with angular upweight
            del _cache_auw

        def get_catalog_recon(catalog):
            # Replace positions with reconstructed positions
            return catalog.clone(POSITION=catalog['POSITION_REC'])

        # Summary statistics computation loop
        for recon in ['', 'recon_']:
            funcs = {f'{recon}particle2_correlation': (compute_particle2_correlation, compute_particle2_correlation_close_pair_correction), f'{recon}particle3_correlation': (compute_particle3_correlation, compute_particle3_correlation_close_pair_correction)}
            for stat, func in funcs.items():
                if stat in stats:
                    correlation_options = dict(options[stat]) | dict(auw=auw_options.get(stat, None))
                    # Extract selection weights if provided (e.g., NX**(-1. / 3.) weighting)
                    selection_weights = correlation_options.pop('selection_weights', None)
                    # Optional per-tracer number of random files for this statistic (subset of the loaded randoms),
                    # e.g. recon_particle2_correlation uses fewer randoms than the catalog/recon nran.
                    correlation_nran = correlation_options.pop('nran', None)
                    if correlation_nran is not None and jax.process_index() == 0:
                        logger.info(f'{stat}: using {correlation_nran} random file(s) per tracer '
                                    f'(out of {{{", ".join(f"{t!r}: {len(zrandoms[t])}" for t in tracers)}}} loaded).')

                    def get_data(tracer):
                        # Prepare data for spectrum measurement
                        # Optionally restrict to a reduced number of random files for this statistic
                        _zrandoms = zrandoms[tracer]
                        if correlation_nran is not None:
                            nran = correlation_nran[tracer] if isinstance(correlation_nran, dict) else correlation_nran
                            _zrandoms = _zrandoms[:nran]
                        if recon:
                            # Use reconstructed positions, with same shifts applied to randoms
                            toret = {'data': get_catalog_recon(zdata[tracer]), 'randoms': _zrandoms,
                                    'shifted': [get_catalog_recon(zrandom) for zrandom in _zrandoms]}
                        else:
                            # Default: use original positions
                            toret = {'data': zdata[tracer], 'randoms': _zrandoms}
                        # Apply selection weights if provided (for bispectrum, NX**(-1. / 3.) weighting, etc.)
                        if selection_weights:
                            toret = {name: selection_weights[tracer](catalog) for name, catalog in toret.items()}
                        return toret

                    # Compute 2 or 3-point correlation function
                    if 'close_pair_correction' in stats:
                        # Add close pair correction (angular upweighting or theta-cut)
                        _correlation_options = correlation_options | dict(auw=False, cut=False)
                        fn = get_stats_fn(kind=stat, catalog=fn_catalog_options, **_correlation_options)
                        correlation = types.read(fn)
                        correlation = func[1](*[functools.partial(get_data, tracer) for tracer in tracers], correlation=correlation, **correlation_options)

                        # Write all correlation variants to disk
                        for key, kw in _expand_cut_auw_options(stat, correlation_options).items():
                            fn = get_stats_fn(kind=stat, catalog=fn_catalog_options, **kw)
                            if key != 'raw':
                                write_stats(fn, correlation[key])
                    else:
                        # Base calculation
                        correlation = func[0](*[functools.partial(get_data, tracer) for tracer in tracers], **correlation_options)

                        # Write all spectrum variants to disk
                        for key, kw in _expand_cut_auw_options(stat, correlation_options).items():
                            if key not in correlation: continue
                            fn = get_stats_fn(kind=stat, catalog=fn_catalog_options, **kw)
                            # Apply blinding if requested
                            if with_stats_blinding:
                                correlation[key] = tools.apply_blinding(correlation[key], tracers, zrange=sum(zrange.values(), start=tuple()))
                            # Store reconstruction metadata
                            if recon:
                                correlation[key].attrs.update(stat_recon_attrs)
                            tools.write_stats(fn, correlation[key])

            # Map of spectrum statistics to computation functions
            funcs = {f'{recon}mesh2_spectrum': (compute_mesh2_spectrum, compute_mesh2_spectrum_close_pair_correction), f'{recon}mesh3_spectrum': (compute_mesh3_spectrum, compute_mesh3_spectrum_close_pair_correction)}

            for stat, func in funcs.items():
                if stat in stats:
                    spectrum_options = dict(options[stat]) | dict(auw=auw_options.get(stat, None))
                    # Extract selection weights if provided (e.g., NX**(-1. / 3.) weighting)
                    selection_weights = spectrum_options.pop('selection_weights', None)

                    def get_data(tracer):
                        # Prepare data for spectrum measurement
                        # Concatenate all random catalogs into single object
                        czrandoms = Catalog.concatenate(zrandoms[tracer])
                        if recon:
                            # Use reconstructed positions, with same shifts applied to randoms
                            toret = {'data': get_catalog_recon(zdata[tracer]), 'randoms': czrandoms,
                                     'shifted': get_catalog_recon(czrandoms)}
                        else:
                            # Default: use original positions
                            toret = {'data': zdata[tracer], 'randoms': czrandoms}
                        # Apply selection weights if provided (for bispectrum, NX**(-1. / 3.) weighting, etc.)
                        if selection_weights:
                            toret = {name: selection_weights[tracer](catalog) for name, catalog in toret.items()}
                        return toret

                    if 'close_pair_correction' in stats:
                        # Add close pair correction (angular upweighting or theta-cut)
                        _spectrum_options = spectrum_options | dict(auw=False, cut=False)
                        fn = get_stats_fn(kind=stat, catalog=fn_catalog_options, **_spectrum_options)
                        spectrum = types.read(fn)
                        spectrum = func[1](*[functools.partial(get_data, tracer) for tracer in tracers], spectrum=spectrum, **spectrum_options)

                        # Write all spectrum variants to disk
                        for key, kw in _expand_cut_auw_options(stat, spectrum_options).items():
                            fn = get_stats_fn(kind=stat, catalog=fn_catalog_options, **kw)
                            if key != 'raw':
                                write_stats(fn, spectrum[key])
                    else:
                        # Compute power spectrum or bispectrum
                        spectrum = func[0](*[functools.partial(get_data, tracer) for tracer in tracers], cache=cache, **spectrum_options)
                        # Ensure spectrum is a dictionary (may contain raw, cut, auw variants)
                        if not isinstance(spectrum, dict): spectrum = {'raw': spectrum}

                        # Write all spectrum variants to disk
                        for key, kw in _expand_cut_auw_options(stat, spectrum_options).items():
                            fn = get_stats_fn(kind=stat, catalog=fn_catalog_options, **kw)
                            # Apply blinding if requested
                            if with_stats_blinding:
                                spectrum[key] = tools.apply_blinding(spectrum[key], tracers, zrange=sum(zrange.values(), start=tuple()))
                            # Store reconstruction metadata
                            if recon:
                                spectrum[key].attrs.update(stat_recon_attrs)
                            tools.write_stats(fn, spectrum[key])

        # Synchronize across all processes before proceeding to windows
        jax.experimental.multihost_utils.sync_global_devices('spectrum')  # wait for the writer

        # Window matrix
        funcs = {"window_mesh2_spectrum": compute_window_mesh2_spectrum,
                 "window_mesh3_spectrum": compute_window_mesh3_spectrum}

        for stat, func in funcs.items():
            if stat in stats:
                window_options = dict(options[stat])
                # Extract selection weights if provided
                selection_weights = window_options.pop('selection_weights', None)

                def get_data(tracer):
                    # Prepare randoms for window function computation
                    czrandoms = Catalog.concatenate(zrandoms[tracer])
                    toret = {'data': zdata[tracer], 'randoms': czrandoms}
                    # Apply selection weights if provided
                    if selection_weights:
                        toret = {name: selection_weights[tracer](catalog) for name, catalog in toret.items()}
                    return toret

                # Load measured spectrum (or compute if not provided)
                spectrum_fn = window_options.pop('spectrum', None)
                fn_window_options = window_options | dict(auw=False)
                if spectrum_fn is None:
                    # Auto-detect spectrum filename from options
                    spectrum_stat = stat.replace("window_", "")
                    fn_window_options = options[spectrum_stat] | fn_window_options
                    spectrum_fn = get_stats_fn(kind=spectrum_stat, catalog=fn_catalog_options, **(options[spectrum_stat] | dict(auw=False, cut=False)))
                spectrum = types.read(spectrum_fn)

                def get_extra(ibatch):
                    # Generate batch identifier string for window correlation functions
                    if ibatch is None:
                        return ''
                    ibatch, nbatch = ibatch
                    return f'batch-{ibatch:d}-{nbatch:d}'

                # Check if computing window in batches (for memory efficiency)
                ibatch = window_options.get('ibatch', None)
                extra = get_extra(ibatch)

                # Load previously computed batch windows if continuing
                batches = window_options.get('computed_batches', [])
                if batches:
                    if not isinstance(batches, (tuple, list)):
                        batches = [(ibatch, batches) for ibatch in np.arange(batches)]
                    # Load window multipole batches computed in previous runs
                    method = window_options.get('method', 'smooth_mesh')
                    npt = {'window_mesh2_spectrum': 2, 'window_mesh3_spectrum': 3}[stat]
                    fns = [get_stats_fn(kind=f'window_{method}{npt:d}_correlation_raw', catalog=fn_catalog_options, **(fn_window_options | dict(battrs={'s': None, 'pole': None}, extra=get_extra(ibatch)))) for ibatch in batches]
                    window_options['computed_batches'] = [types.read(fn) for fn in fns]
                # Remove basis from options (will be extracted from spectrum)
                window_options.pop('basis', None)
                # Compute window function
                window = func(*[functools.partial(get_data, tracer) for tracer in tracers], spectrum=spectrum, **window_options)

                # Write window matrix to disk
                for key, kw in _expand_cut_auw_options(stat, fn_window_options).items():
                    if key in window:
                        # Extract basis from spectrum if available
                        basis = getattr(next(iter(window[key].observable)), 'basis', None)
                        if basis is not None: kw['basis'] = basis
                        # Also save under "geometry", to assemble with forward-modeled window "window_mesh2_spectrum_fm"
                        for suffix in ['', '_geometry']:  # FIXME the suffix won't be caught by list_stats
                            fn = get_stats_fn(kind=stat + suffix, catalog=fn_catalog_options, **kw)
                            write_stats(fn, window[key])

                # Write raw correlation functions (intermediate products) to disk
                for key in window:
                    if 'correlation' in key:  # window functions
                        fn = get_stats_fn(kind=key, catalog=fn_catalog_options, **(fn_window_options | dict(battrs={'s': None, 'pole': None}, cut=False, extra=extra)))
                        write_stats(fn, window[key])
        # Synchronize before window forward model computation
        jax.experimental.multihost_utils.sync_global_devices('window')  # wait for the writer

        # Window matrix using forward model (for RIC and AMR effects)
        funcs = {"window_mesh2_spectrum_fm": compute_window_mesh2_spectrum_fm}
        for stat, func in funcs.items():
            if stat in stats:
                # len(tracers) == 1 if autocorr, else 2
                window_options = dict(options[stat])
                selection_weights = window_options.pop("selection_weights", None)

                def get_data(tracer):
                    # Prepare randoms for forward model window computation
                    toret = {"data": data[tracer], "randoms": Catalog.concatenate(randoms[tracer])}
                    if selection_weights:
                        toret = {name: selection_weights[tracer](catalog) for name, catalog in toret.items()}
                    return toret

                # Get fiducial theory for computing forward model derivatives
                theory_stat = stat.replace("window_", "theory_").replace("_fm", "")
                theory_fn = window_options.pop("theory", None)

                if theory_fn is None:
                    # Auto-compute fiducial theory from spectrum and window
                    products_fn = {spectrum_region: {} for spectrum_region in window_options["spectrum_regions_zranges"]}
                    # Collect power spectrum and window, for each region if relevant
                    for _region, _zrange in window_options["spectrum_regions_zranges"]:
                        for name in ["spectrum", "window"]:
                            kind_stat = (
                                stat.replace("window_", "").replace("_fm", "") if name == "spectrum" else stat.replace("window_", f"{name}_").replace("_fm", "")
                            )
                            fn = window_options.pop(name, None)
                            if fn is None:
                                # Auto-detect measurement filename
                                kw = options[kind_stat] | {"auw": False, "cut": False}
                                fn = get_stats_fn(
                                    kind=kind_stat,
                                    catalog={tracer: fn_catalog_options[tracer] for tracer in tracers},
                                    **kw | {"region": _region, "zrange": _zrange},
                                )
                            products_fn[(_region, _zrange)][name] = fn

                    # Load spectra and windows from disk
                    spectra = [types.read(products_fn[(_region, _zrange)]["spectrum"]) for _region, _zrange in window_options["spectrum_regions_zranges"]]
                    windows = [types.read(products_fn[(_region, _zrange)]["window"]) for _region, _zrange in window_options["spectrum_regions_zranges"]]
                    # Combine measurements from multiple regions and fit for theory
                    theory = types.sum(
                        [
                            run_preliminary_fit_mesh2_spectrum(data=spectrum, window=window, theory="kaiser")
                            for spectrum, window in zip(spectra, windows, strict=True)
                        ]
                    )
                    theory_fn = get_stats_fn(
                        kind=theory_stat,
                        catalog={tracer: fn_catalog_options[tracer] for tracer in tracers},
                    )
                    write_stats(theory_fn, theory)

                # Synchronize before reading theory
                jax.experimental.multihost_utils.sync_global_devices("theory")  # such that theory ready for window
                theory = types.read(theory_fn)

                theory_rebin = window_options.pop('theory_rebin', None)
                if theory_rebin is not None:
                    # Rebin theory to speed up window function computation
                    theory = theory.select(k=slice(0, None, theory_rebin))

                # Load example of output measurements. If spectra_fn provided, use it; otherwise use spectra loaded for the preliminary fit in the theory block above
                spectra_fn = window_options.pop("spectra", None)
                fn_window_options = window_options | {"auw": False, "cut": False}
                if spectra_fn is None:
                    spectra_fn = []
                    spectrum_stat = stat.replace("window_", "").replace("_fm", "")
                    for _region, _zrange in window_options["spectrum_regions_zranges"]:
                        fn_window_options = options[spectrum_stat] | fn_window_options
                        spectra_fn.append(
                            get_stats_fn(
                                kind=spectrum_stat, catalog=fn_catalog_options, **(options[spectrum_stat] | {"auw": False, "cut": False} | {"region": _region, "zrange": _zrange})
                            )
                        )
                spectra = [types.read(spectrum_fn) for spectrum_fn in spectra_fn]

                # Now compute window function using forward model with derivatives
                window = func(*[functools.partial(get_data, tracer) for tracer in tracers], spectra=spectra, theory=theory, **window_options)
                # This is a dict of dict of lists of windows : {modeled_effect: {spectrum_region: [window, ...], ...}, ...}
                for effect in window:  # geo, RIC or RIC+AMR
                    for _region, _zrange in window_options["spectrum_regions_zranges"]:  # window[effect]:  # eg NGC, SGC and a zrange
                        for i, seed in enumerate(window_options['seeds']):
                            # FIXME this overrides the extra option pre-defined in get_stats_fn through e.g. functools.partial. Not sure this is an actual issue.
                            if window_options['ellsout'] is None:
                                extra = f"{effect}_seed={seed}"
                            else:
                                listell = "".join(map(str, window_options['ellsout']))
                                extra = f'{effect}_{listell}_seed={seed}'

                            options = fn_window_options | {"extra": extra, "region": _region, "zrange": _zrange}
                            write_stats(get_stats_fn(kind=stat, catalog=fn_catalog_options, **options), window[effect][(_region, _zrange)][i])

                # synchronize here to avoid postprocess trying to load windows that haven't been written yet
                jax.experimental.multihost_utils.sync_global_devices("window_fm_IO")  # wait for the writer

        # Covariance matrix computation
        for recon in ['', 'recon_']:
            funcs = {f'covariance_{recon}mesh2_spectrum': compute_covariance_mesh2_spectrum, f'covariance_{recon}particle2_correlation': compute_covariance_particle2_correlation}
            for stat, func in funcs.items():
                if stat in stats:
                    covariance_options = dict(options[stat])
                    theory_stat = stat.replace('covariance_', 'theory_')
                    theory_fn = covariance_options.pop('theory', None)

                    def get_data(tracer):
                        # Prepare catalogs for covariance computation
                        czrandoms = Catalog.concatenate(zrandoms[tracer])
                        return {'data': zdata[tracer], 'randoms': czrandoms}

                    def _check_fn(fn, tracers, name=''):
                        # Convert single filename to tracer pair dictionary
                        if len(tracers) == 1:
                            fn = {(tracer, tracer): fn for tracer in tracers}
                        else:
                            raise ValueError(f'provide a dictionary of (tracer1, tracer2): {name} for tracer1, tracer2 in {tracers}')
                        return fn

                    def _read_tracer(fns, tracers2):
                        # Read file for tracer pair (handle ordering)
                        if tracers2 not in fns: tracers2 = tracers2[::-1]
                        return types.read(fns[tracers2])

                    if theory_fn is None:
                        # Auto-compute fiducial theory from spectrum and window
                        products_fn = {}
                        # Collect power spectrum and window
                        for name in ['spectrum', 'window']:
                            kind_stat = {'spectrum': f'{recon}mesh2_spectrum', 'window': 'window_mesh2_spectrum'}[name]
                            fn = covariance_options.pop(name, None)
                            if fn is None:
                                # Auto-detect measurement files for each tracer pair
                                kw = options[kind_stat] | dict(auw=False, cut=False)
                                fn = {(tracer, tracer): get_stats_fn(kind=kind_stat, catalog=fn_catalog_options[tracer], **kw) for tracer in tracers}
                                # Add cross-correlation file if multiple tracers
                                if len(tracers) > 1:
                                    fn[tuple(tracers)] = get_stats_fn(kind=kind_stat, catalog=fn_catalog_options, **kw)
                            elif not isinstance(fn, dict):
                                _check_fn(fn, tracers, name=name)
                            products_fn[name] = fn

                        # Compute theory for each tracer pair
                        theory_fn = {}
                        for tracers2 in itertools.combinations_with_replacement(tracers, r=2):
                            spectrum = _read_tracer(products_fn['spectrum'], tracers2)
                            window = _read_tracer(products_fn['window'], tracers2)
                            # Fit theory to measurement (preliminary fit for covariance)
                            theory = run_preliminary_fit_mesh2_spectrum(data=spectrum, window=window, theory='recon' if recon else 'rept')
                            theory_fn[tracers2] = get_stats_fn(kind=theory_stat, catalog=(fn_catalog_options[tracers2[0]] if tracers2[1] == tracers2[0] else {tracer: fn_catalog_options[tracer] for tracer in tracers2}))
                            # Write theory to disk
                            tools.write_stats(theory_fn[tracers2], theory)
                    else:
                        _check_fn(theory_fn, tracers, name='theory')

                    # Synchronize before reading theory
                    jax.experimental.multihost_utils.sync_global_devices('theory')  # such that theory ready for window

                    # Load theory for all tracer pairs
                    fields = {tracer: tools.get_simple_tracer(tracer) for tracer in tracers}
                    theory = {tuple(fields[tracer] for tracer in tracers2): _read_tracer(theory_fn, tracers2) for tracers2 in itertools.product(tracers, repeat=2)}
                    theory = types.ObservableTree(list(theory.values()), fields=list(theory.keys()))

                    if 'particle2' in stat:
                        RR_fn = covariance_options.pop('RR', None)
                        kind_stat = stat.replace('covariance_', '')
                        if RR_fn is None:
                            kw = dict(auw=False, cut=False) | options[kind_stat]
                            RR_fn = {(tracer, tracer): get_stats_fn(kind=kind_stat, catalog=fn_catalog_options[tracer], **kw) for tracer in tracers}
                            # Add cross-correlation file if multiple tracers
                            if len(tracers) > 1:
                                RR_fn[tuple(tracers)] = get_stats_fn(kind=kind_stat, catalog=fn_catalog_options, **kw)
                        elif not isinstance(RR_fn, dict):
                            _check_fn(RR_fn, tracers, name=name)
                        # Load RR for all tracer pairs
                        RR = {tuple(fields[tracer] for tracer in tracers2): _read_tracer(RR_fn, tracers2) for tracers2 in itertools.product(tracers, repeat=2)}
                        RR = {fields: RR[fields].get('RR') if 'count_names' in RR[fields].labels(return_type='keys') else RR[fields] for fields in RR}
                        covariance_options['RR'] = types.ObservableTree(list(RR.values()), fields=list(RR.keys()))

                    # Compute covariance matrix
                    covariance = func(*[functools.partial(get_data, tracer) for tracer in tracers], theory=theory, fields=list(fields.values()), **covariance_options)

                    def add_label(covariance):
                        # Add observables label, and fields => tracers
                        simple_stat = tools.get_simple_stats(stat.replace('covariance_', ''))
                        # Create observable tree with proper labels
                        observable = types.ObservableTree(list(covariance.observable), observables=[simple_stat] * len(fields), tracers=covariance.observable.fields)
                        return covariance.clone(observable=observable)

                    # Write covariance matrix to disk
                    for key, kw in _expand_cut_auw_options(stat, covariance_options).items():
                        fn = get_stats_fn(kind=stat, catalog=fn_catalog_options, **kw)
                        if key in covariance:
                            tools.write_stats(fn, add_label(covariance[key]))

                    # Write intermediate correlation functions to disk
                    for key in covariance:
                        if 'correlation' in key:  # window functions
                            fn = get_stats_fn(kind=key, catalog=fn_catalog_options, **(covariance_options | dict(auw=False, cut=False)))
                            tools.write_stats(fn, covariance[key])

        # Joint 2-point + 3-point covariance. Unlike covariance_mesh2_spectrum, this only
        # supports a single tracer: compute_covariance_mesh3_spectrum does not implement
        # multi-tracer cross-covariance (it always labels both P and B with the same field).
        stat = 'covariance_mesh3_spectrum'
        if stat in stats:
            assert len(tracers) == 1, f'{stat} only supports a single tracer, got {tracers}'
            tracer = tracers[0]
            simple_tracer = tools.get_simple_tracer(tracer)
            covariance_options = dict(options[stat])
            # theory is a python callable (P/B/T kernels at best-fit bias), not an lsstypes
            # object: unlike covariance_mesh2_spectrum's theory, it is not written to disk.
            theory = covariance_options.pop('theory', None)
            shotnoise = covariance_options.pop('shotnoise', None)
            # Optional override of the box volume assumed in the preliminary bias fit
            # (defaults to the measurement's embedding box, which overestimates the survey
            # volume); see run_preliminary_fit_mesh3_spectrum.
            prelim_mattrs = covariance_options.pop('prelim_mattrs', None)

            def get_data(tracer):
                czrandoms = Catalog.concatenate(zrandoms[tracer])
                return {'data': zdata[tracer], 'randoms': czrandoms}

            # Auto-detect the raw (no cut/auw) measured P(k), B(k1, k2): they set the covariance
            # binning and, if theory is not provided, are fit to get the bias parameters.
            spectrum2_fn = covariance_options.pop('spectrum2', None)
            if spectrum2_fn is None:
                kw = options['mesh2_spectrum'] | dict(auw=False, cut=False)
                spectrum2_fn = get_stats_fn(kind='mesh2_spectrum', catalog=fn_catalog_options[tracer], **kw)
            spectrum3_fn = covariance_options.pop('spectrum3', None)
            if spectrum3_fn is None:
                kw = options['mesh3_spectrum'] | dict(auw=False, cut=False)
                spectrum3_fn = get_stats_fn(kind='mesh3_spectrum', catalog=fn_catalog_options[tracer], **kw)
            spectrum2, spectrum3 = types.read(spectrum2_fn), types.read(spectrum3_fn)

            if shotnoise is None:
                # (1 + alpha) / nbar, read off the FKP monopole shot noise level
                shotnoise = float(np.mean(spectrum2.get(spectrum2.ells[0]).values('shotnoise')))

            if theory is None:
                # Effective redshift: measured spectra carry no 'zeff' (only window matrices
                # do), so read it off the already-computed window instead.
                window_fn = get_stats_fn(kind='window_mesh2_spectrum', catalog=fn_catalog_options[tracer],
                                         **(options['window_mesh2_spectrum'] | dict(auw=False)))
                window = types.read(window_fn)
                z = window.observable.get(ells=0).attrs['zeff']
                if prelim_mattrs is not None:
                    from jaxpower import MeshAttrs
                    prelim_mattrs = MeshAttrs(**prelim_mattrs)
                # Fit bias parameters on the joint (P, B) data vector
                theory = run_preliminary_fit_mesh3_spectrum(spectrum2, spectrum3, mattrs=prelim_mattrs, z=z, shotnoise=shotnoise)

            results = compute_covariance_mesh3_spectrum(functools.partial(get_data, tracer), spectrum2=spectrum2, spectrum3=spectrum3,
                                                        theory=theory, shotnoise=shotnoise, fields=[simple_tracer], **covariance_options)

            def add_label(covariance):
                # Label the two stacked (P, B) blocks with their observable kind and tracer(s)
                observable = types.ObservableTree(list(covariance.observable), observables=['mesh2_spectrum', 'mesh3_spectrum'],
                                                  tracers=covariance.observable.fields)
                return covariance.clone(observable=observable)

            # Write covariance matrix to disk
            for key, kw in _expand_cut_auw_options(stat, covariance_options).items():
                fn = get_stats_fn(kind=stat, catalog=fn_catalog_options, **kw)
                if key in results:
                    tools.write_stats(fn, add_label(results[key]))

            # Write intermediate covariance-window correlation functions to disk
            for key in results:
                if 'correlation' in key:
                    fn = get_stats_fn(kind=key, catalog=fn_catalog_options, **(covariance_options | dict(auw=False, cut=False)))
                    tools.write_stats(fn, results[key])


def list_stats(stats, get_stats_fn=tools.get_stats_fn, **kwargs):
    """
    List measurements produced by :func:`compute_stats_from_options`.

    Parameters
    ----------
    stats : str or list of str
        Summary statistics to list.
    get_stats_fn : callable, optional
        Function to get the filename for storing the measurement.
    **kwargs : dict
        Options for catalog and summary statistics. For example:
            catalog = dict(version='holi-v1-altmtl', tracer='LRG', zrange=[(0.4, 0.6), (0.8, 1.1)], imock=451)
            mesh2_spectrum = dict(cut=True, auw=True, ells=(0, 2, 4), mattrs=dict(boxsize=7000., cellsize=8.))  # all arguments for compute_mesh2_spectrum
            mesh3_spectrum = dict(basis='sugiyama-diagonal', ells=[(0, 0, 0)], mattrs=dict(boxsize=7000., cellsize=10.))  # all arguments for compute_mesh3_spectrum
    """
    # Ensure stats is a list
    if isinstance(stats, str):
        stats = [stats]

    # Fill fiducial defaults
    extra = kwargs.pop('extra', None)
    kwargs = fill_fiducial_options(kwargs)
    catalog_options = kwargs['catalog']
    tracers = list(catalog_options.keys())
    catalog_blinding_options = {tracer: catalog_options[tracer].pop('blinding', None) for tracer in tracers}
    for tracer, blinding_options in catalog_blinding_options.items():
        if blinding_options:
            catalog_options[tracer]['version'] = catalog_blinding.output_version_from_options(catalog_options[tracer].get('version', None), blinding_options)
    # Build list of redshift ranges for each tracer
    zranges = {tracer: _make_list_zrange(catalog_options[tracer]['zrange']) for tracer in tracers}

    toret = {stat: [] for stat in stats}
    # Iterate over all combinations of redshift bins and statistics
    for zvals in zip(*(zranges[tracer] for tracer in tracers)):
        zrange = dict(zip(tracers, zvals))
        _catalog_options = {tracer: catalog_options[tracer] | dict(zrange=zrange[tracer]) for tracer in tracers}
        for stat in stats:
            # Generate option combinations (raw, cut, auw)
            for kw in _expand_cut_auw_options(stat, kwargs[stat]).values():
                kw = dict(catalog=_catalog_options, **kw)
                fn = get_stats_fn(kind=stat, **(kw | {"extra": extra}))
                toret[stat].append((fn, kw))
    return toret


def postprocess_stats_from_options(postprocess, analysis='full_shape', get_stats_fn=tools.get_stats_fn, **kwargs):
    """
    Postprocess summary statistics based on the provided options.

    Parameters
    ----------
    postprocess : str or list of str
        Postprocessing.
        Choices: ['combine_regions', 'combine_window_mesh2_spectrum', 'rotation_mesh2_spectrum', 'systematic_templates']
    analysis : str, optional
        Type of analysis, 'full_shape' or 'png_local', to set fiducial options.
    get_stats_fn : callable, optional
        Function to get the filename for storing the measurement.
    **kwargs : dict
        Options for summary statistics, and choices in ``postprocess``.
    """
    # Ensure postprocess is a list
    if isinstance(postprocess, str):
        postprocess = [postprocess]

    imocks = kwargs.pop('imocks', None)
    extra = kwargs.pop('extra', None)

    # Fill fiducial defaults
    options = fill_fiducial_options(kwargs, analysis=analysis)
    catalog_options = options['catalog']
    tracers = list(catalog_options.keys())
    catalog_blinding_options = {tracer: catalog_options[tracer].pop('blinding', None) for tracer in tracers}
    for tracer, blinding_options in catalog_blinding_options.items():
        if blinding_options:
            catalog_options[tracer]['version'] = catalog_blinding.output_version_from_options(catalog_options[tracer].get('version', None), blinding_options)
    # Set default region to combined
    for tracer in tracers:
        catalog_options[tracer].setdefault('region', 'GCcomb')  # default, for rotation, rotate
    # Build redshift range lists
    zranges = {tracer: _make_list_zrange(catalog_options[tracer]['zrange']) for tracer in tracers}
    # Default imock if not specified
    if imocks is None:
        imocks = [catalog_options[tracers[0]].get('imock', None)]

    def _iter_on_mocks(options, imocks=imocks):
        # Helper to iterate over multiple mock realizations
        _options = copy.deepcopy(options)
        for imock in imocks:
            for tracer in _options['catalog']:
                _options['catalog'][tracer]['imock'] = imock
            yield _options

    # Loop over redshift bins
    for zvals in zip(*(zranges[tracer] for tracer in tracers)):
        zrange = dict(zip(tracers, zvals))
        fn_catalog_options = {tracer: catalog_options[tracer] | dict(zrange=zrange[tracer]) for tracer in tracers}

        if 'combine_regions' in postprocess:
            # Combine measurements from different sky regions (NGC, SGC)
            combine_options = dict(options.get('combine_regions', {}))
            regions = combine_options.pop('regions', ['NGC', 'SGC'])
            stats = combine_options.pop('stats', ['mesh2_spectrum', 'mesh3_spectrum'])

            def _combine_stats(stat, region_comb, regions, get_stats_fn=get_stats_fn, **options):
                # Helper to combine statistics from multiple regions
                all_fns = {}
                # List all measurement files for each region
                for region in regions + [region_comb]:
                    kwargs = dict(options)
                    kwargs['catalog'] = {tracer: options['catalog'][tracer] | dict(region=region) for tracer in options['catalog']}
                    all_fns[region] = list_stats(stat, get_stats_fn=get_stats_fn, **kwargs)
                stats = next(iter(all_fns.values())).keys()
                # Combine each statistic variant (auw, cut)
                for stat in stats:
                    for ifn, (fn_comb, _) in enumerate(all_fns[region_comb][stat]):
                        fns = [all_fns[region][stat][ifn][0] for region in regions]  # [1] is kwargs
                        exists = {os.path.exists(fn): fn for fn in fns}
                        if all(exists):
                            # Read and combine measurements from different regions
                            combined = tools.combine_stats([types.read(fn) for fn in fns])
                            tools.write_stats(fn_comb, combined)
                        else:
                            logger.info(f'Skipping {fn_comb} as {[fn for ex, fn in exists.items() if not ex]} do not exist')

            # Get all possible region combinations
            for region_comb, regions in tools.possible_combine_regions(regions).items():
                for stat in stats:
                    if 'window' in stat or 'covariance' in stat:
                        # Window and covariance don't need to loop over mocks
                        _combine_stats(stat, region_comb, regions, get_stats_fn=get_stats_fn, **(options| {"extra": extra}))
                    else:
                        # Measurements need to loop over mocks
                        for _options in _iter_on_mocks(options | dict(catalog=fn_catalog_options), imocks=imocks):
                            _combine_stats(stat, region_comb, regions, get_stats_fn=get_stats_fn, **(_options | {"extra": extra}))

        if 'combine_window_mesh2_spectrum' in postprocess:
            # Combine base window calculation with forward-modeled windows
            stat = 'window_mesh2_spectrum'
            combine_options = dict(options.get('combine_window_mesh2_spectrum', {}))
            effect = combine_options.pop('effect', 'RIC+AMR')
            window_options = options.get(stat, {})
            window_fm_options = options.get(f'{stat}_fm', {})
            window_fm = None

            for window_geometry_fn, kw in list_stats(stat, get_stats_fn=get_stats_fn, catalog=fn_catalog_options, **{stat: window_options, 'mesh2_spectrum': options.get('mesh2_spectrum', {})})[stat]:
                window_geometry = types.read(window_geometry_fn)
                if window_fm is None:
                    window_realizations = []
                    for i, seed in enumerate(window_fm_options['seeds']):
                        diff = []
                        for _effect in ['geometry', effect]:
                            if window_fm_options['ellsout'] is None:
                                extra = f"{_effect}_seed={seed}"
                            else:
                                listell = "".join(map(str, window_fm_options['ellsout']))
                                extra = f'{_effect}_{listell}_seed={seed}'
                            diff.append(types.read(get_stats_fn(kind=f'{stat}_fm', **(kw | {"extra": extra}))))
                        window_realizations.append(diff[0].clone(value=diff[1].value() - diff[0].value()))
                    window_fm = window_realizations[0].clone(value=np.mean([window.value() for window in window_realizations], axis=0))
                    # build downscaling matrix
                    block = types.utils.matrix_spline_interp(xt=window_geometry.theory.get(**window_geometry.theory.labels()[0]).coords(0), xo=window_fm.theory.get(0).coords(0))
                    downscale = np.kron(np.eye(len(window_geometry.theory.ells)), block)
                    # use it to "interpolate" the forward-modeled window to the geometry of the base window
                    window_fm = window_fm.clone(value=window_fm.value().dot(downscale))

                fn = get_stats_fn(kind=stat, **(kw | {"extra": effect}))
                # Adding all effects
                window = window_geometry.clone(value=window_geometry.value() + window_fm.value())
                tools.write_stats(fn, window)

        if 'systematic_templates' in postprocess:
            stat = 'systematic_templates'
            systematic_options = dict(options.get(stat, {}))
            for stat in systematic_options.get('stats', []):
                for window_fn, kw_window in list_stats(f'window_{stat}', get_stats_fn=get_stats_fn, catalog=fn_catalog_options, **{stat: options.get(stat, {}), f'window_{stat}': options.get(f'window_{stat}', {})})[f'window_{stat}']:
                    window = types.read(window_fn)
                    effects = list(systematic_options.get('effects', []))
                    templates = {}
                    for key, fns in systematic_options.get('templates', {}).items():
                        if key == 'auw' and kw_window.get('cut', None):
                            effects = [effect for effect in effects if effect != 'auw']
                            continue
                        if isinstance(fns, dict):
                            imocks = fns.get('imock', imocks)
                            fns = [get_stats_fn(kind=stat, **{**kw_window, 'imock': imock, 'auw': None, 'cut': None, **fns}) for imock in imocks]
                        if not isinstance(fns, (tuple, list)):
                            fns = [fns]
                        templates[key] = types.mean([types.read(fn) for fn in fns])
                    if effects:
                        window = include_systematic_templates(window, templates=templates, effects=effects)
                        fn = get_stats_fn(kind=f'window_{stat}', **kw_window, templates=list(effects))
                        tools.write_stats(fn, window)

        if 'rotation_mesh2_spectrum' in postprocess:
            # Compute rotation matrix for power spectrum (corrections for systematic effects)
            stat = 'rotation_mesh2_spectrum'
            kind_stat = stat.replace('rotation_', '')
            rotation_options = dict(options.get(stat, {}))
            products = {}
            # Read window and covariance (required for rotation computation)
            for name, kind in zip(['window', 'covariance'], [f'window_{kind_stat}', f'covariance_{kind_stat}']):
                fn = rotation_options.pop(name, None)
                if fn is None:
                    # Auto-detect window/covariance filenames
                    # FIXME, in case covariance with theta-cut is available
                    kw = options.get(kind_stat, {}) | dict(auw=False, cut=(name == 'window'))
                    fn = get_stats_fn(kind=kind, catalog=fn_catalog_options, **kw)
                # Read from disk or use provided object
                if isinstance(fn, types.ObservableTree):
                    products[name] = fn
                else:
                    products[name] = types.read(fn)

            # Select auto-covariance for single tracer
            tracers2 = tuple(tracers * (2 // len(tracers)))
            #print(products['covariance'].observable.labels())
            products['covariance'] = products['covariance'].at.observable.get(observables=tools.get_simple_stats(kind_stat), tracers=tuple(tools.get_simple_tracer(tracer) for tracer in tracers2))

            # Read or compute data and theory measurements
            for name in ['data', 'theory']:
                fns = rotation_options.pop(name, {})
                if isinstance(fns, dict):
                    # Auto-detect filenames
                    kw = dict(catalog=fn_catalog_options) | options.get(kind_stat, {}) | dict(auw=False, cut=(name == 'data')) | fns
                    fns = get_stats_fn(kind=kind_stat, **kw)
                # Read from disk or use provided object
                if isinstance(fns, types.ObservableTree):
                    products[name] = fns
                else:
                    # Read and average multiple measurements
                    if isinstance(fns, (str, Path)): fns = [fns]
                    products[name] = types.mean([types.read(fn) for fn in fns])

            # Compute rotation on single process only (not parallelized)
            if jax.process_index() == 0:
                rotation = compute_rotation_mesh2_spectrum(**products, **rotation_options)
                # Save rotation matrix to disk
                kw = options.get(kind_stat, {}) | dict(auw=False, cut=True)
                fn = get_stats_fn(kind=stat, catalog=fn_catalog_options, **kw)
                tools.write_stats(fn, rotation)


def combine_stats_from_options(stats, region_comb, regions, get_stats_fn=tools.get_stats_fn, **kwargs):
    """
    Combine summary statistics from multiple regions based on the provided options.

    Warning
    --------
    Use postprocess_from_options(['combine_regions']) instead.

    Parameters
    ----------
    stats : str or list of str
        Summary statistics to combine.
    region_comb : str
        Combined region name, e.g. 'GCcomb'.
    regions : list of str
        Regions to combine, e.g. ['NGC', 'SGC'].
    get_stats_fn : callable, optional
        Function to get the filename for storing the measurement.
    **kwargs : dict
        Options for catalog and summary statistics. For example:
            catalog = dict(version='holi-v1-altmtl', tracer='LRG', zrange=[(0.4, 0.6), (0.8, 1.1)], imock=451)
            mesh2_spectrum = dict(cut=True, auw=True, ells=(0, 2, 4), mattrs=dict(boxsize=7000., cellsize=8.))  # all arguments for compute_mesh2_spectrum
            mesh3_spectrum = dict(basis='sugiyama-diagonal', ells=[(0, 0, 0)], mattrs=dict(boxsize=7000., cellsize=10.))  # all arguments for compute_mesh3_spectrum
    """
    warnings.warn("deprecated; use postprocess_from_options(['combine_regions']) instead")
    options = fill_fiducial_options(kwargs)
    regions = list(regions)
    all_fns = {}
    # List all measurement files for each region
    for region in regions + [region_comb]:
        kwargs = dict(options)
        kwargs['catalog'] = {tracer: options['catalog'][tracer] | dict(region=region) for tracer in options['catalog']}
        all_fns[region] = list_stats(stats, get_stats_fn=get_stats_fn, **kwargs)

    stats = next(iter(all_fns.values())).keys()
    # Combine each statistic
    for stat in stats:
        for ifn, (fn_comb, _) in enumerate(all_fns[region_comb][stat]):
            fns = [all_fns[region][stat][ifn][0] for region in regions]  # [1] is kwargs
            exists = {os.path.exists(fn): fn for fn in fns}
            if all(exists):
                # Read and combine measurements from different regions
                combined = tools.combine_stats([types.read(fn) for fn in fns])
                tools.write_stats(fn_comb, combined)
            else:
                logger.debug(f'Skipping {fn_comb} as {[fn for ex, fn in exists.items() if not ex]} do not exist')


def main(**kwargs):
    r"""
    This is an example main, which can be run from command line to compute fiducial statistics.
    Let's try to keep it simple; write your own if you need anything fancier.
    Or just use :func:`compute_stats_from_options` directly; see example in :mod:`job_scripts/desipipe_holi_mocks.py`.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats', help='what do you want to compute?', type=str, nargs='*', choices=['mesh2_spectrum', 'recon_mesh2_spectrum', 'mesh3_spectrum', 'particle2_correlation', 'recon_particle2_correlation', 'particle3_correlation', 'recon_particle3_correlation', 'close_pair_correction', 'window_mesh2_spectrum', 'window_mesh3_spectrum'], default=['mesh2_spectrum'])
    parser.add_argument('--version', help='catalog version; e.g. holi-v1-altmtl', type=str, default=None)
    parser.add_argument('--cat_dir', help='where to find catalogs', type=str, default=None)
    parser.add_argument('--tracer', help='tracer(s) to be selected - e.g. LRG ELG for cross-correlation', nargs='*', type=str, default='LRG')
    parser.add_argument('--zrange', help='redshift bins; 0.4 0.6 0.8 1.1 to run (0.4, 0.6), (0.8, 1.1)', nargs='*', type=float, default=None)
    parser.add_argument('--imock', help='mock number', type=int, nargs='*', default=[None])
    parser.add_argument('--region', help='regions', type=str, nargs='*', choices=['N', 'S', 'NGC', 'SGC', 'NGCnoN', 'SGCnoDES', 'DES', 'ACT_DR6', 'PLANCK_PR4', 'GAL040', 'GAL060'], default=['NGC', 'SGC'])
    parser.add_argument('--analysis', help='type of analysis', type=str, choices=['full_shape', 'png_local', 'full_shape_protected'], default='full_shape')
    parser.add_argument('--weight',  help='type of weights to use for tracer; "default" just uses WEIGHT column', type=str, default='default-FKP')
    parser.add_argument('--thetacut',  help='Apply theta-cut', action='store_true', default=None)
    parser.add_argument('--auw',  help='Apply angular upweighting', action='store_true', default=None)
    parser.add_argument('--boxsize',  help='box size', type=float, default=None)
    parser.add_argument('--cellsize', help='cell size', type=float, default=None)
    parser.add_argument('--nran', help='number of random files to combine together (1-18 available)', type=int, default=None)
    parser.add_argument('--make_complete', help='make on-the-fly (completeness-weighted) complete catalogs', action='store_true', default=None)
    parser.add_argument('--expand_randoms', help='expand catalog of randoms; provide version of parent randoms (must be registered in get_catalog_fn)', type=str, default=None)
    meas_dir = Path(os.getenv('SCRATCH')) / 'measurements'
    parser.add_argument('--stats_dir',  help=f'base directory for measurements, default is {meas_dir}', type=str, default=meas_dir)
    parser.add_argument('--stats_extra',  help='extra string to include in measurement filename', type=str, default='')
    parser.add_argument('--combine', help='combine measurements in two regions', type=str, nargs='*', default=None, choices=['mesh2_spectrum', 'recon_mesh2_spectrum', 'mesh3_spectrum', 'particle2_correlation', 'recon_particle2_correlation', 'window_mesh2_spectrum', 'window_mesh3_spectrum'])

    args = parser.parse_args()
    # Set JAX to use 90% of GPU memory (leave 10% for overhead)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    import jax
    # Initialize distributed JAX if computing statistics
    if args.stats:
        jax.distributed.initialize()

    # Set up logging
    setup_logging()

    # Get default redshift ranges for tracer and analysis type
    if args.zrange is None:
        zranges = tools.propose_fiducial('zranges', tracer=tools.join_tracers(args.tracer), analysis=args.analysis)
    else:
        # Parse redshift range from command line (pairs of values)
        assert len(args.zrange) % 2 == 0
        zranges = list(zip(args.zrange[::2], args.zrange[1::2]))

    # Build mesh options (boxsize and cellsize)
    mattrs = {key: value for key, value in dict(boxsize=args.boxsize, cellsize=args.cellsize).items() if value is not None}
    options = {'mattrs': mattrs}
    # Apply theta-cut and angular upweighting options if requested
    for stat in ['mesh2_spectrum', 'particle2_correlation']:
        options.setdefault(stat, {})
        options[stat].update(cut=args.thetacut, auw=args.auw)

    # Set up catalog filename lookup function
    get_catalog_fn = tools.get_catalog_fn
    # Set up statistics filename generation function with custom directory and extra string
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=args.stats_dir, extra=args.stats_extra)
    cache = {}

    def _keep_if_not_none(**kwargs):
        # Helper to filter out None values from kwargs
        return {k: v for k, v in kwargs.items() if v is not None}

    # Build catalog options from command-line arguments
    catalog_options = dict(tracer=args.tracer, zrange=zranges, ext=None)
    catalog_options |= _keep_if_not_none(weight=args.weight, version=args.version, cat_dir=args.cat_dir, nran=args.nran)
    # Merge all options and fill fiducial defaults
    options = _merge_options(fill_fiducial_options(dict(catalog=catalog_options) | options, analysis=args.analysis), kwargs)

    # Compute statistics for each mock realization and sky region
    if args.stats:
        for imock in args.imock:
            for region in args.region:
                _options_imock = dict(options)
                for tracer in _options_imock['catalog']:
                    _options_imock['catalog'][tracer] = _options_imock['catalog'][tracer] | dict(region=region, imock=imock)
                    # Add expanded random catalog if requested (not all columns saved)
                    if args.expand_randoms:
                        _options_imock['catalog'][tracer]['expand'] = {'parent_randoms_fn': get_catalog_fn(kind='parent_randoms', version=args.expand_randoms, tracer=tracer, region=region, nran=max(value['nran'] for value in _options_imock['recon'].values()))}
                    # Enable completeness-weighted catalogs if requested
                    if args.make_complete:
                        _options_imock['catalog'][tracer]['complete'] = {}
                # Compute all requested statistics
                compute_stats_from_options(args.stats, get_catalog_fn=get_catalog_fn, get_stats_fn=get_stats_fn, cache=cache, analysis=args.analysis, **_options_imock)
                # Synchronize all processes before next region
                jax.experimental.multihost_utils.sync_global_devices(region)

    # Postprocess statistics (combine regions, compute rotations)
    if args.combine is not None and jax.process_index() == 0:
        stats = []
        if args.combine: stats = args.combine
        elif args.stats: stats = [stat for stat in args.stats if stat != 'close_pair_correction'] # avoid passing 'close_pair_correction' to `postprocess_stats_from_options`
        else: stats = ['mesh2_spectrum', 'mesh3_spectrum']  # best guess, if not argument was provided
        postprocess_stats_from_options(['combine_regions'], get_stats_fn=get_stats_fn, combine_regions=dict(stats=stats), **options, imocks=args.imock)

    # Shutdown distributed JAX
    if args.stats:
        jax.distributed.shutdown()


if __name__ == '__main__':

    from jax import config
    # Enable 64-bit precision for higher accuracy (slower computation)
    config.update('jax_enable_x64', False)
    main()
