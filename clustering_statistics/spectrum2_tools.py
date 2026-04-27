"""
Fourier-space 2-point clustering measurements.

Main functions
--------------
* `prepare_jaxpower_particles`: Convert catalogs into mesh-ready particle inputs.
* `compute_mesh2_spectrum`: Main `P(k)` measurement backend.
* `compute_window_mesh2_spectrum`: Compute the power spectrum window matrix.
* `compute_window_mesh2_spectrum_fm`: Build forward-model window matrix.
* `compute_covariance_mesh2_spectrum`: Estimate Fourier-space covariance.
* `run_preliminary_fit_mesh2_spectrum`: Run preliminary fits used in covariance matrix.
"""

import logging
from collections.abc import Callable

import numpy as np
import jax
from jax import numpy as jnp
import lsstypes as types

from .tools import default_mpicomm, _format_bitweights, compute_fkp_effective_redshift, combine_stats


logger = logging.getLogger('spectrum2')


@default_mpicomm
def prepare_jaxpower_particles(*get_data_randoms, mattrs=None, add_data=tuple(), add_randoms=tuple(), check=True, **kwargs):
    """
    Prepare :class:`jaxpower.ParticleField` objects from data and randoms catalogs.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data' (optionally 'randoms', 'shifted') catalogs.
        Each catalog must contain 'POSITION' and 'INDWEIGHT', and optionally 'BITWEIGHT' for bitwise weights and 'TARGETID'
        for randoms IDs to allow process-invariant random split in bispectrum normalization.
    mattrs : dict, optional
        Mesh attributes ('boxsize', 'meshsize' or 'cellsize', 'boxcenter') to define the :class:`ParticleField` objects. If ``None``, default attributes are used.
    kwargs : dict, optional
        Additional keyword arguments to pass to :class:`ParticleField`.

    Returns
    -------
    all_particles : list of dictionaries
        List of dictionaries of :class:`ParticleField`  'data' (optionally 'randoms', 'shifted') objects for each input catalog.
    """
    # Import mesh attribute computation and particle field creation from jaxpower
    from jaxpower.mesh import get_mesh_attrs, ParticleField
    # Use MPI backend for distributed particle processing across processes
    backend = 'mpi'
    # Extract MPI communicator from kwargs (added by @default_mpicomm decorator)
    mpicomm = kwargs['mpicomm']

    # Load all catalogs by calling the provided functions
    all_catalogs = [_get_data_randoms() for _get_data_randoms in get_data_randoms]

    # Define the mesh attributes; pass in positions only
    # check=True validates that all positions are within mesh bounds
    mattrs = get_mesh_attrs(*[catalog["POSITION"] for catalogs in all_catalogs for catalog in catalogs.values()], check=check, **(mattrs or {}))
    if jax.process_index() == 0:
        logger.info(f'Using mesh {mattrs}.')

    # Use IDS instead
    def collective_arange(local_size):
        # Compute global array indices across all MPI processes
        # This allows each process to know its global position in the distributed array
        sizes = mpicomm.allgather(local_size)
        return sum(sizes[:mpicomm.rank]) + np.arange(local_size)

    all_particles = []
    # Dictionary mapping 'data' and 'randoms' to their respective extra columns to load
    add = {'data': add_data, 'randoms': add_randoms}
    for catalogs in all_catalogs:
        particles = {}
        for name, catalog in catalogs.items():
            extra = {}
            # Start with individual weights from catalog
            indweights = catalog['INDWEIGHT']
            if name == 'data':
                # Extract and process bitwise weights (fiber weights, completeness, etc.)
                bitweights = None
                if 'BITWEIGHT' in catalog and 'BITWEIGHT' in add[name]:
                    # Parse bitwise weight array into individual weight components
                    bitweights = _format_bitweights(catalog['BITWEIGHT'])
                    from cucount.jax import BitwiseWeight
                    # Compute individual inverse probability weight (IIP) from bitwise components
                    # p_correction_nbits=False: no impact on IIP computation
                    iip = BitwiseWeight(weights=bitweights, p_correction_nbits=False)(bitweights)
                    # Store original bitweights in extra
                    extra['BITWEIGHT'] = [indweights] + bitweights
                    # Multiply individual weights by IIP to correct fiber assignment at large scales
                    indweights = indweights * iip
                # Add any additional columns (e.g., Z, WEIGHT_FKP) to extra dictionary
                for column in add[name]:
                    if column != 'BITWEIGHT': extra[column] = catalog[column]
            elif name == 'randoms':
                # Extract target IDs from random catalog for reproducible random splitting
                if 'TARGETID' in catalog and 'IDS' in add[name]:
                    extra['IDS'] = catalog['TARGETID']
                # Add other requested columns to extra dictionary
                for column in add[name]:
                    if column != 'IDS': extra[column] = catalog[column]
            # Create ParticleField object: positions + weights + mesh attributes
            # exchange=True: distribute particles across MPI processes by spatial location
            # This ensures load balancing across processes
            particle = ParticleField(catalog["POSITION"], indweights, attrs=mattrs, exchange=True, backend=backend, extra=extra, **kwargs)
            particles[name] = particle
        all_particles.append(particles)
    if jax.process_index() == 0:
        logger.info(f'All particles on the device')

    return all_particles


def _get_jaxpower_attrs(*all_particles):
    """Return summary attributes from :class:`jaxpower.ParticleField` objects: total weight and size."""
    # Get mesh attributes from first particle set (same for all)
    mattrs = next(iter(all_particles[0].values())).attrs
    # Creating FKP fields
    attrs = {}
    for particles in all_particles:
        for name in particles:
            if particles[name] is not None:
                # Store total weight sum for each particle type
                if f'wsum_{name}' not in attrs:
                    #attrs[f'size_{name}'] = [[]]  # size is process-dependent
                    attrs[f'wsum_{name}'] = [[]]
                #attrs[f'size_{name}'][0].append(particles[name].size)
                # Sum weights across all processes using MPI
                attrs[f'wsum_{name}'][0].append(particles[name].sum())
    # Extract and preserve mesh geometric information for output
    for name in ['boxsize', 'boxcenter', 'meshsize']:
        attrs[name] = mattrs[name]
    return attrs


def compute_mesh2_spectrum(*get_data_randoms, mattrs=None, cut=None, auw=None,
                           ells=(0, 2, 4), edges=None, los='firstpoint', optimal_weights=None,
                           norm: dict=None, cache=None):
    r"""
    Compute the 2-point spectrum multipoles using mesh-based FKP fields with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data', 'randoms' (optionally 'shifted') catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'. If ``None``, default attributes are used.
    cut : bool, optional
        If True, apply a theta-cut of (0, 0.05) in degrees.
    auw : ObservableTree, optional
        Angular upweights to apply. If ``None``, no angular upweights are applied.
    ells : list of int, optional
        List of multipole moments to compute. Default is (0, 2, 4).
    edges : dict, optional
        Edges for the binning; array or dictionary with keys 'start' (minimum :math:`k`), 'stop' (maximum :math:`k`), 'step' (:math:`\Delta k`).
        If ``None``, default step of :math:`0.001 h/\mathrm{Mpc}` is used.
        See :class:`jaxpower.BinMesh2SpectrumPoles` for details.
    los : {'local', 'firstpoint', 'x', 'y', 'z', array-like}, optional
        Line-of-sight definition. 'local' uses local LOS, 'firstpoint' uses the position of the first point in the pair,
        'x', 'y', 'z' use fixed axes, or provide a 3-vector.
    optimal_weights : callable or None, optional
        Function taking (ell, catalog) as input and returning total weights to apply to data and randoms.
        It can have an optional attribute 'columns' that specifies which additional columns are needed to compute the optimal weights.
        As a default, ``optimal_weights.columns = ['Z']`` to indicate that redshift information is needed.
        A dictionary ``catalog`` of columns is provided, containing 'INDWEIGHT' and the requested columns.
        If ``None``, no optimal weights are applied.
    norm : dict, optional
        Optional arguments for computing normalization.
        Default is ``{'cellsize': 10.}`` (density computed with ``cellsize = 10.``)
    cache : dict, optional
        Cache to store binning class (can be reused if ``meshsize`` and ``boxsize`` are the same).
        If ``None``, a new cache is created.

    Returns
    -------
    spectrum : Mesh2SpectrumPoles or dict of Mesh2SpectrumPoles
        The computed 2-point spectrum multipoles. If `cut` or `auw` are provided, returns a dict with keys 'raw', 'cut', and/or 'auw'.
    """

    # Import FKP field, power spectrum computation, and binning tools from jaxpower
    from jaxpower import (create_sharding_mesh, FKPField, compute_fkp2_normalization, compute_fkp2_shotnoise, BinMesh2SpectrumPoles, compute_mesh2_spectrum,
                          BinParticle2SpectrumPoles, BinParticle2CorrelationPoles, compute_particle2, compute_particle2_shotnoise)

    # Collect column names needed for optimal weight computation
    columns_optimal_weights = []
    if optimal_weights is not None:
        # Get required columns (default: Z for redshift-dependent weights)
        columns_optimal_weights += getattr(optimal_weights, 'columns', ['Z'])   # to compute optimal weights, e.g. for fnl
    mattrs = mattrs or {}
    # Set up distributed mesh computation across JAX devices (multi-GPU/CPU)
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        # Load particles and prepare for FKP field creation
        # add_data=['BITWEIGHT'] for fiber collision corrections
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_data=['BITWEIGHT'] + columns_optimal_weights, add_randoms=columns_optimal_weights)

        # Initialize or retrieve cached binning object from previous runs
        if cache is None: cache = {}
        # Set default k-space binning step (0.001 h/Mpc)
        if edges is None: edges = {'step': 0.001}
        # Set default normalization computation parameters (density from 10 Mpc/h cells)
        if norm is None: norm = {'cellsize': 10.}
        kw_norm = dict(norm)

        def _compute_spectrum_ell(all_particles, ells, fields=None):
            # Compute power spectrum for input given multipoles
            # Gather particle attributes (weights, mesh info) for output metadata
            attrs = _get_jaxpower_attrs(*all_particles)
            # Store line-of-sight direction in attributes
            attrs.update(los=los)
            # Get mesh attributes from first data particle
            mattrs = all_particles[0]['data'].attrs

            # Define the binner for k-space binning
            key = 'bin_mesh2_spectrum_{}'.format('_'.join(map(str, ells)))
            bin = cache.get(key, None)
            # Create new binning if not cached or if mesh parameters changed
            if bin is None or not np.all(bin.mattrs.meshsize == mattrs.meshsize) or not np.allclose(bin.mattrs.boxsize, mattrs.boxsize):
                bin = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)
            # Store binning in cache for future use
            cache.setdefault(key, bin)

            all_fkp = [FKPField(particles['data'], particles['randoms']) for particles in all_particles]
            # Computing normalization: integral of density^2, splitting randoms ('split') to avoid common noise
            norm = compute_fkp2_normalization(*all_fkp, bin=bin, **kw_norm)

            # Computing shot noise from shifted catalogs (reconstruction) or use randoms if no shifted available
            all_fkp = [FKPField(particles['data'], particles['shifted'] if particles.get('shifted', None) is not None else particles['randoms']) for particles in all_particles]
            del all_particles
            # Shot noise computed from (shifted) randoms
            num_shotnoise = compute_fkp2_shotnoise(*all_fkp, bin=bin, fields=fields)

            # Wait for normalization and shot noise to complete on all devices
            jax.block_until_ready((norm, num_shotnoise))
            if jax.process_index() == 0:
                logger.info('Normalization and shotnoise computation finished')

            results = {}
            # First compute the theta-cut (close-pair) contribution for contamination correction
            if cut is not None:
                # Define angular selection: only pairs separated by < 0.05 degrees
                sattrs = {'theta': (0., 0.05)}
                #pbin = BinParticle2SpectrumPoles(mattrs, edges=bin.edges, xavg=bin.xavg, sattrs=sattrs, ells=ells)
                # Use correlation binning for close pairs (finer radial bins for accuracy)
                pbin = BinParticle2CorrelationPoles(mattrs, edges={'step': 0.1}, sattrs=sattrs, ells=ells)
                from jaxpower.particle2 import convert_particles
                # Convert FKP fields to particle pairs for direct pair counting
                all_particles = [convert_particles(fkp.particles) for fkp in all_fkp]
                # Count close pairs directly (no mesh needed, exact calculation)
                close = compute_particle2(*all_particles, bin=pbin, los=los)
                # Attach normalization and shot noise, then convert to power spectrum
                close = close.clone(num_shotnoise=compute_particle2_shotnoise(*all_particles, bin=pbin, fields=fields), norm=norm)
                # Convert correlation poles to power spectrum (multiply by bin centers)
                close = close.to_spectrum(bin.xavg)
                # Store negative contribution (contamination to subtract)
                results['cut'] = -close.value()

            # Then compute the AUW-weighted (angular upweight) pairs and bitwise-weighted pairs
            with_bitweights = 'BITWEIGHT' in all_fkp[0].data.extra
            if auw is not None or with_bitweights:
                from cucount.jax import WeightAttrs
                from jaxpower.particle2 import convert_particles
                # Define angular selection for close pairs (< 0.1 degrees for bitwise weights)
                sattrs = {'theta': (0., 0.1)}
                bitwise = angular = None
                if with_bitweights:
                    # Reconstruct weights for fiber collision corrections
                    all_data = [convert_particles(fkp.data, weights=list(fkp.data.extra['BITWEIGHT']) + [fkp.data.weights], exchange_weights=False) for fkp in all_fkp]
                    # Extract bitwise weight structure (sets nrealizations based on BITWEIGHT size, fine to use the first)
                    bitwise = dict(weights=all_data[0].get('bitwise_weight'))
                    if jax.process_index() == 0:
                        logger.info(f'Applying PIP weights {bitwise}.')
                else:
                    # No bitwise weights, remove individual weights from AUW * individual_weight
                    all_data = [convert_particles(fkp.data, weights=[fkp.data.weights] * 2, exchange_weights=False, index_value=dict(individual_weight=1, negative_weight=1)) for fkp in all_fkp]
                # Apply angular upweights if provided (fiber collision corrections)
                if auw is not None:
                    # Extract angular separation and weight values from pre-computed AUW
                    angular = dict(sep=auw.get('DD').coords('theta'), weight=auw.get('DD').value())
                    if jax.process_index() == 0:
                        logger.info(f'Applying AUW {angular}.')
                # Set up weight attributes for pair counting
                wattrs = WeightAttrs(bitwise=bitwise, angular=angular)
                # Create binning for close-pair data-data counts with weights
                pbin = BinParticle2SpectrumPoles(mattrs, edges=bin.edges, xavg=bin.xavg, sattrs=sattrs, wattrs=wattrs, ells=ells)
                # Count weighted pairs directly
                DD = compute_particle2(*all_data, bin=pbin, los=los)
                # Attach normalization and shot noise
                DD = DD.clone(num_shotnoise=compute_particle2_shotnoise(*all_data, bin=pbin, fields=fields), norm=norm)
                results['auw'] = DD.value()

            # Wait for particle-based calculations to complete
            jax.block_until_ready(results)
            if jax.process_index() == 0:
                logger.info(f'Particle-based calculation finished')

            # Paint particles onto mesh grids for Fourier-space power spectrum computation
            kw = dict(resampler='tsc', interlacing=3, compensate=True)
            # out='real' to save memory (store as real arrays instead of complex)
            all_mesh = [fkp.paint(**kw, out='real') for fkp in all_fkp]
            # Free memory from FKP fields
            del all_fkp

            # JIT the mesh-based spectrum computation; helps with memory footprint
            jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'])
            #jitted_compute_mesh2_spectrum = compute_mesh2_spectrum
            # Compute power spectrum from painted mesh grids via FFT
            spectrum = jitted_compute_mesh2_spectrum(*all_mesh, bin=bin, los=los)
            # Attach normalization and shot noise to spectrum
            spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
            # Propagate particle attributes to each multipole
            spectrum = spectrum.map(lambda pole: pole.clone(attrs=attrs))
            # Also attach attributes at spectrum level
            spectrum = spectrum.clone(attrs=attrs)
            # Wait for spectrum computation to complete on all devices
            jax.block_until_ready(spectrum)
            if jax.process_index() == 0:
                logger.info('Mesh-based computation finished')

            # Add theta-cut and AUW contributions to the base spectrum
            for name, value in results.items():
                # Combine contamination corrections with raw spectrum
                results[name] = spectrum.clone(value=spectrum.value() + value)
            # Store raw spectrum without contamination corrections
            results['raw'] = spectrum

            return results

        # Compute power spectrum either without or with optimal weights
        if optimal_weights is None:
            # Standard case: use FKP weights, compute all ells at once
            results = _compute_spectrum_ell(all_particles, ells=ells)
        else:
            # Optimal weights case: compute ell-by-ell due to dependence of optimal weight on multipole
            # Prepare fields tuple for multi-tracer support (pad to length 2)
            fields = tuple(range(len(all_particles)))
            # Pad to length 2 if we have fewer catalogs (for cross-correlation compatibility)
            fields = fields + (fields[-1],) * (2 - len(fields))
            # Pad particle list similarly for processing
            all_particles = tuple(all_particles) + (all_particles[-1],) * (2 - len(all_particles))
            results = {}
            # Loop over each multipole moment
            for ell in ells:
                if jax.process_index() == 0:
                    logger.info(f'Applying optimal weights for ell = {ell:d}')

                def _get_optimal_weights(all_particles):
                    # Generator that yields particles with optimal weights applied for this ell
                    # all_particles is [data1, data2] or [randoms1, randoms2] or [shifted1, shifted2]
                    if all_particles[0] is None:  # shifted is None, yield None
                        while True:
                            yield tuple(None for particles in all_particles)
                    # Get optimal weights from the weight function for all particles
                    for all_weights in optimal_weights(ell, [{'INDWEIGHT': particles.weights} | {column: particles.extra[column] for column in columns_optimal_weights} for particles in all_particles]):
                        # Yield particles with weights replaced by optimal weights
                        yield tuple(particles.clone(weights=weights) for particles, weights in zip(all_particles, all_weights))

                result_ell = {}
                # Get names of particle types (data, randoms, shifted)
                names = list(all_particles[0].keys())
                # Loop over all weight combinations for this ell
                for _all_particles in zip(*[_get_optimal_weights([particles[name] for particles in all_particles]) for name in names]):
                    # _all_particles is a list [(data1, data2), (randoms1, randoms2), [(shifted1, shifted2)]] of tuples of ParticleField with optimal weights applied
                    # Reorder to group by catalog index rather than particle type
                    _all_particles = list(zip(*_all_particles))
                    # _all_particles is now a list of tuples [(data1, randoms1, shifted1), (data2, randoms2, shifted2)] with optimal weights applied
                    # Convert tuples back to dictionaries for _compute_spectrum_ell
                    _all_particles = [dict(zip(names, _particles)) for _particles in _all_particles]
                    # _all_particles is now a list of dictionaries [{'data': data1, 'randoms': randoms1, 'shifted': shifted1}, {'data': data2, 'randoms': randoms2, 'shifted': shifted2}] with optimal weights applied
                    # Compute spectrum for this ell and weight combination
                    _result = _compute_spectrum_ell(_all_particles, ells=[ell], fields=fields)
                    # Collect results for all variants (raw, cut, auw)
                    for key in _result:  # raw, cut, auw
                        result_ell.setdefault(key, [])
                        result_ell[key].append(_result[key])
                # Average over weight combinations (if multiple tracers: sum over 1<->2 cross-weights)
                for key, value in result_ell.items():
                    results.setdefault(key, [])
                    # Combine_stats sums over different weights
                    results[key].append(combine_stats(value))  # sum 1<->2
            # Combine all ells into single observable tree structure
            for key in results:
                # types.join concatenates along multipole axis
                results[key] = types.join(results[key])  # join multipoles

    # Return single result or dictionary of variants
    if len(results) == 1:
        return next(iter(results.values()))
    return results


def compute_window_mesh2_spectrum(*get_data_randoms, spectrum: types.Mesh2SpectrumPoles, optimal_weights: Callable=None, cut
: bool=None, zeff: dict=None, method: str='smooth'):
    r"""
    Compute the 2-point spectrum window with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'randoms' catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    spectrum : Mesh2SpectrumPoles
        Measured 2-point spectrum multipoles.
    optimal_weights : callable or None, optional
        Function taking (ell, catalog) as input and returning total weights to apply to data and randoms.
        It can have an optional attribute 'columns' that specifies which additional columns are needed to compute the optimal weights.
        As a default, ``optimal_weights.columns = ['Z']`` to indicate that redshift information is needed.
        A dictionary ``catalog`` of columns is provided, containing 'INDWEIGHT' and the requested columns.
        If ``None``, no optimal weights are applied.
    zeff : dict, optional
        Optional arguments for computing effective redshift.
        Default is ``{'cellsize': 10.}`` (density computed with ``cellsize = 10.``)

    Returns
    -------
    window : WindowMatrix or dict of WindowMatrix
        The computed 2-point spectrum window. If `auw` is provided, returns a dict with keys 'raw' and 'auw'.
    """
    # FIXME: data is not used, could be dropped, add auw
    # Import window and correlation computation tools from jaxpower
    from jaxpower import (create_sharding_mesh, BinMesh2SpectrumPoles, BinMesh2CorrelationPoles, compute_mesh2_correlation, BinParticle2CorrelationPoles, compute_particle2, compute_particle2_shotnoise,
                           compute_smooth2_spectrum_window, get_smooth2_window_bin_attrs, interpolate_window_function, split_particles, compute_mesh2_spectrum_window)

    # Extract multipole moments from input spectrum
    ells = spectrum.ells
    # Extract mesh parameters from spectrum attributes
    mattrs = {name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    # Extract line-of-sight direction
    los = spectrum.attrs['los']
    # Theory multipoles for window computation (fixed basis for window calculation)
    ellsin = [0, 2, 4]
    # Mesh painting parameters: TSC kernel with 3-fold interlacing for aliasing correction
    kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
    # Set default effective redshift computation parameters
    if zeff is None: zeff = {'cellsize': 10.}
    kw_zeff = dict(zeff)

    # Collect column names needed for optimal weight computation
    columns_optimal_weights = []
    if optimal_weights is not None:
        # Get required columns (default: Z for redshift-dependent weights)
        columns_optimal_weights += getattr(optimal_weights, 'columns', ['Z'])   # to compute optimal weights, e.g. for fnl
    mattrs = mattrs or {}
    # Set up distributed computation mesh
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        # Load random catalogs and prepare particles with IDS for reproducible splitting
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'] + columns_optimal_weights)
        # Extract only randoms (data not used in window computation)
        all_randoms = [particles['randoms'] for particles in all_particles]
        del all_particles

        # Determine k-space binning edges from input spectrum
        stop, step = -np.inf, np.inf
        for pole in spectrum:
            # Get k-space edges from each pole
            edges = pole.edges('k')
            # Find maximum k and minimum step across all poles
            stop = max(edges.max(), stop)
            step = min(np.nanmin(np.diff(edges, axis=-1)), step)
        # Create finer k-binning for theory
        edgesin = np.arange(0., 1.2 * stop, step)
        # Convert to column-stacked format [k_min, k_max] for each bin
        edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])

        def _compute_window_ell(all_randoms, ells, isum=0, fields=None):
            # Compute window function for specified multipole moments
            all_randoms = list(all_randoms)
            # Use IDS (TARGETID) from random catalog for process-invariant random splitting
            seed = [(42, randoms.extra["IDS"]) for randoms in all_randoms]  # for process invariance
            mattrs = all_randoms[0].attrs
            # Get spectrum pole at first ell to extract edges
            pole = spectrum.get(ells[0])
            # Create binning object for window function
            bin = BinMesh2SpectrumPoles(mattrs, edges=pole.edges('k'), ells=ells)
            # Get normalization from input power spectrum
            norm = jnp.concatenate([spectrum.get(ell).values('norm') for ell in ells], axis=0)

            results = {}
            if method == 'smooth':
                correlations = []
                # Get window basis attributes (which multipoles to compute in correlation space)
                kw_window = get_smooth2_window_bin_attrs(ells, ellsin)
                # JIT-compile correlation computation for memory efficiency
                # donate_argnums=[0] allows JAX to reuse memory of first argument
                jitted_compute_mesh2_correlation = jax.jit(compute_mesh2_correlation, static_argnames=['los'], donate_argnums=[0])
                # Window computed in configuration space, summing Bessel functions over the Fourier-space mesh
                # Use logarithmic s-grid for robust interpolation
                coords = jnp.logspace(-3, 5, 4 * 1024)
                list_edges = []
                # Loop over scale factors for multigrid window computation (coarse then fine)
                for scale in [1, 4]:
                    # Create coarser mesh (larger boxsize) for computational efficiency at coarse scales
                    mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize)
                    if jax.process_index() == 0:
                        logger.info(f'Processing scale x{scale:.0f}, using {mattrs2}')
                    all_mesh = []
                    # Paint random catalogs on mesh
                    for iran, randoms in enumerate(split_particles(all_randoms + [None] * (2 - len(all_randoms)), seed=seed, fields=fields)):
                        # Redistribute particles to mesh and exchange across MPI processes
                        randoms = randoms.clone(attrs=mattrs2).exchange(backend='mpi')
                        # Compute weight normalization (data/random density ratio from input spectrum)
                        alpha = pole.attrs['wsum_data'][isum][min(iran, len(all_randoms) - 1)] / randoms.weights.sum()
                        # Paint random particles with proper normalization onto mesh
                        all_mesh.append(alpha * randoms.paint(**kw_paint, out='real'))
                    # Define radial binning for correlation space window
                    # distmax: use 1/4 of boxsize minimum dimension for correlation range
                    distmax, cellsize = mattrs2.boxsize.min() / 4., mattrs2.cellsize.min()
                    # Create radial bins from 0 to distmax
                    edges = np.arange(0., distmax + cellsize, cellsize)
                    list_edges.append(edges)
                    # Create binning for correlation function (configuration space)
                    sbin = BinMesh2CorrelationPoles(mattrs2, edges=edges, **kw_window, basis='bessel')
                    # Compute correlation function via FFT (inverse Fourier transform of painted mesh)
                    correlation = jitted_compute_mesh2_correlation(all_mesh, bin=sbin, los=los).clone(norm=[np.mean(norm)] * len(sbin.ells))
                    # Free mesh memory
                    del all_mesh
                    #if jax.process_index() == 0: correlation.write(f'_tests/window_correlation2_{scale:.0f}.h5')
                    # Interpolate correlation to fine logarithmic k-grid for FFTLog integration
                    correlation = interpolate_window_function(correlation, coords=coords, order=3)
                    correlations.append(correlation)
                # Create transition masks between coarse and fine scale grids
                # Masks ensure each point is covered by exactly one scale
                masks = [coords < edges[-3] for edges in list_edges[:-1]]
                # Last mask covers remainder
                masks.append((coords < np.inf))
                # Convert masks to exclusive regions (each point weighted by only one scale)
                weights = []
                for mask in masks:
                    if len(weights):
                        # Exclude already-weighted regions from previous scales
                        weights.append(mask & (~weights[-1]))
                    else:
                        weights.append(mask)
                # Regularize weights to avoid division by zero
                weights = [np.maximum(mask, 1e-6) for mask in weights]
                # Combine correlations from different scales using smooth weights
                results['window_mesh2_correlation_raw'] = correlation = correlations[0].sum(correlations, weights=weights)
                # Convert correlation to power spectrum window via FFTLog
                window = compute_smooth2_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=('fftlog',))
                # Normalize window by average normalization (in case norm is k-dependent)
                # Division corrects for any k-dependent normalization variation
                window = window.clone(value=window.value() / (norm[..., None] / np.mean(norm)))
            
            elif method == 'exact':
                all_mesh = []
                if jax.process_index() == 0:
                    logger.info(f'Using {mattrs}')
                # Paint random catalogs on mesh
                for iran, randoms in enumerate(split_particles(all_randoms + [None] * (2 - len(all_randoms)), seed=seed, fields=fields)):
                    # Redistribute particles to mesh and exchange across MPI processes
                    randoms = randoms.clone(attrs=mattrs).exchange(backend='mpi')
                    # Compute weight normalization (data/random density ratio from input spectrum)
                    alpha = pole.attrs['wsum_data'][isum][min(iran, len(all_randoms) - 1)] / randoms.weights.sum()
                    # Paint random particles with proper normalization onto mesh
                    all_mesh.append(alpha * randoms.paint(**kw_paint, out='real'))
                # (ellins, 'local') to include first order wide-angle correction:
                window = compute_mesh2_spectrum_window(*all_mesh, edgesin=edgesin, ellsin=(ellsin, 'local'), los=los, bin=bin, pbar=False, flags=('infinite',), norm=1.)
                # FIXME
                window = window.clone(value=window.value().real / norm[..., None])

            # Update window with spectrum normalization from input power spectrum
            observable = window.observable.map(lambda pole, label: pole.clone(norm=spectrum.get(**label).values('norm'), attrs=pole.attrs), input_label=True)
            results['raw'] = window.clone(observable=observable)

            if cut:
                # Compute theta-cut contribution to window
                sattrs = {'theta': (0., 0.05)}
                kw_window = get_smooth2_window_bin_attrs(ells, ellsin)
                #pbin = BinParticle2SpectrumPoles(mattrs, edges=bin.edges, xavg=bin.xavg, sattrs=sattrs, **kw_window)
                # Use correlation binning for close pairs (finer bins than spectrum)
                pbin = BinParticle2CorrelationPoles(mattrs, edges={'step': 0.1}, sattrs=sattrs, **kw_window)
                from jaxpower.particle2 import convert_particles
                all_particles = []
                # Convert randoms to particles and apply weight normalization
                for iran, randoms in enumerate(all_randoms):
                    alpha = pole.attrs['wsum_data'][isum][iran] / randoms.weights.sum()
                    all_particles.append(convert_particles(randoms.clone(weights=alpha * randoms.weights)))
                # Count close pairs directly
                correlation = compute_particle2(*all_particles, bin=pbin, los=los)
                # Attach normalization and shot noise
                correlation = correlation.clone(num_shotnoise=compute_particle2_shotnoise(*all_particles, bin=pbin, fields=fields), norm=[np.mean(norm)] * len(pbin.ells))
                pole = next(iter(correlation))
                # Interpolate close pair correlation to logarithmic grid
                coords = jnp.logspace(-3, 5, 4 * 1024)
                correlation = interpolate_window_function(correlation, coords=coords, order=3)
                results['window_mesh2_correlation_cut'] = correlation
                ## Combine raw and close-pair correlations (remove contributions)
                #correlation = correlation.clone(value=results['window_mesh2_correlation_raw'].value() - correlation.value())
                # Convert combined correlation to power spectrum window
                window = compute_smooth2_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=('fftlog',))
                results['cut'] = window.clone(observable=results['raw'].observable, value=results['raw'].value() - window.value() / (norm[..., None] / np.mean(norm)))
            # Convert correlation results to observable tree structure (for consistency with other outputs)
            for key, result in results.items():
                if 'correlation' in key:
                    # Wrap correlation in ObservableTree with appropriate ell structure
                    results[key] = types.ObservableTree([result], oells=[ells[0] if len(ells) == 1 else tuple(ells)])
            return results

        # Compute window either without or with optimal weights
        if optimal_weights is None:
            # Standard case: use FKP weights, compute all ells at once
            # Compute effective redshift for window computation
            fields = None
            seed = [(42, randoms.extra["IDS"]) for randoms in all_randoms]
            # Compute zeff and normalization fraction (for reporting)
            zeff, norm_zeff = compute_fkp_effective_redshift(*all_randoms, order=2, split=seed, fields=fields, return_fraction=True, **kw_zeff)
            results = _compute_window_ell(all_randoms, ells=ells, fields=fields)
            # Attach zeff to window attributes
            for key in results:
                if 'correlation' not in key:
                    observable = results[key].observable
                    # Add effective redshift to each pole's attributes
                    observable = observable.map(lambda pole: pole.clone(attrs=pole.attrs | dict(zeff=zeff / norm_zeff, norm_zeff=norm_zeff)))
                    results[key] = results[key].clone(observable=observable)
        else:
            # Optimal weights case: compute ell-by-ell due to weight dependence on multipole
            results = {}
            # Prepare fields tuple for multi-tracer support (pad to length 2)
            fields = tuple(range(len(all_randoms)))
            fields = fields + (fields[-1],) * (2 - len(fields))
            # Pad random particle list similarly
            all_randoms = tuple(all_randoms) + (all_randoms[-1],) * (2 - len(all_randoms))
            # Loop over multipoles
            for ell in ells:
                if jax.process_index() == 0:
                    logger.info(f'Applying optimal weights for ell = {ell:d}')

                def _get_optimal_weights(all_particles):
                    # Generator for optimal weights applied to particles
                    # all_particles is [data1, data2] or [randoms1, randoms2] or [shifted1, shifted2]
                    if all_particles[0] is None:  # shifted is None, yield None
                        while True:
                            yield tuple(None for particles in all_particles)
                    def clone(particles, weights):
                        # Clone particle with new weights
                        toret = particles.clone(weights=weights)
                        return toret

                    # Get optimal weights for this ell from the weight function
                    for all_weights in optimal_weights(ell, [{"INDWEIGHT": particles.weights} | {column: particles.extra[column] for column in columns_optimal_weights} for particles in all_particles]):
                        yield tuple(clone(particles, weights=weights) for particles, weights in zip(all_particles, all_weights))

                result_ell = {}
                # Loop over weight combinations for this ell
                for isum, _all_randoms in enumerate(_get_optimal_weights(all_randoms)):
                    # Loop over weight combinations for the same multipole
                    fields = None
                    seed = [(42, randoms.extra["IDS"]) for randoms in _all_randoms]
                    # Compute zeff for this weight combination
                    zeff, norm_zeff = compute_fkp_effective_redshift(*_all_randoms, order=2, split=seed, fields=fields, return_fraction=True, **kw_zeff)
                    _result = _compute_window_ell(_all_randoms, ells=[ell], isum=isum, fields=fields)
                    # Attach zeff to output for each weight combination
                    for key in _result:  # raw, cut, auw
                        if 'correlation' not in key:
                            observable = _result[key].observable
                            observable = observable.map(lambda pole: pole.clone(attrs=pole.attrs | dict(zeff=zeff / norm_zeff, norm_zeff=norm_zeff)))
                            _result[key] = _result[key].clone(observable=observable)
                        result_ell.setdefault(key, [])
                        result_ell[key].append(_result[key])
                # Average over weight combinations
                for key, windows in result_ell.items():
                    results.setdefault(key, [])
                    # windows can be WindowMatrix and ObservableTree (window correlation)
                    window = combine_stats(windows)  # sum 1<->2
                    # Used power spectrum norm is for the sum of the two;
                    # just sum the two components
                    window = window.clone(value=sum(window.value() for window in windows))
                    results[key].append(window)
            # Combine results into final observable structure
            for key in results:
                if 'correlation' in key:
                    # Join correlations along multipole axis
                    results[key] = types.join(results[key])
                else:
                    # Join windows along multipole axis
                    observables = [window.observable for window in results[key]]
                    observable = types.join(observables)
                    value = np.concatenate([window.value() for window in results[key]], axis=0)
                    results[key] = results[key][0].clone(value=value, observable=observable)  # join multipoles

    return results


def compute_window_mesh2_spectrum_fm(
    *get_data_randoms: Callable,
    spectra: list[types.Mesh2SpectrumPoles],
    theory: types.Mesh2SpectrumPoles,
    optimal_weights: Callable | None,
    data_to_randoms_ratio: float,
    catalog_split_seed: int,
    geo: bool,
    ric: bool,
    ric_nbins: int | tuple[int, int],
    ric_regions: list[str] | tuple[list[str], list[str]],
    amr: bool,  # is optional
    ellsout: list[int] | None,
    regression_maps: list[str] | tuple[list[str], list[str]] | None,
    templates_paths_kwargs: dict | tuple[dict, dict] | None,
    amr_regions_zranges: list[tuple[str, tuple[float, float]]] | tuple[list[tuple[str, tuple[float, float]]], list[tuple[str, tuple[float, float]]]] | None,
    spectrum_regions_zranges: list[str] | None,
    total_region_zrange: tuple[str, tuple[float, float]],
    unitary_amplitude: bool = True,
    n_realizations: int,
    seeds: list[int] | None,
    batch_size: int = 4,
) -> dict[str, dict[str, list[types.WindowMatrix]]]:
    """
    Compute the 2-point spectrum window with :mod:`desiwinds`.

    Parameters
    ----------
    *get_data_randoms : Callable
        Functions that return tuples of (data, randoms) catalogs.
    spectra: list[types.Mesh2SpectrumPoles]
        Measured 2-point spectrum multipoles. Only used for their attributes (binning, norm, wsum...), not their values.
    theory: lsstypes.Mesh2SpectrumPoles
        Input theory power spectrum, used as a fiducial for the derivative. Attributes (e.g. ells) used for mock survey generation; value used for the derivative.
    optimal_weights : Callable or None
        Function taking (ell, catalog) as input and returning total weights to apply to data and randoms.
        It can have an optional attribute 'columns' that specifies which additional columns are needed to compute the optimal weights.
        As a default, ``optimal_weights.columns = ['Z']`` to indicate that redshift information is needed.
        A dictionary ``catalog`` of columns is provided, containing 'INDWEIGHT' and the requested columns.
        If ``None``, no optimal weights are applied.
    data_to_randoms_ratio : float
        Population ratio between "data" and "randoms" to pick in the input randoms catalogs. Must be between 0 and 1.
    catalog_split_seed : int
        Random seed to use for the random split between "data" and "randoms" in the input randoms catalogs.
    geo : bool
        Whether to return the sampled window for the geometry. If False, not computed.
    ric : bool
        Wether to return the sampled window for the geometry + RIC (+/- AMR if amr=True). If False, not computed.
    ric_nbins : int | tuple[int, int]
        Number of radial bins to use for the RIC. Can provide as tuple for cross-spectra.
    ric_regions : list[str] | tuple[list[str], list[str]]
        Regions to use for the RIC, e.g. ``["N", "S"]`` or ``["N", "SnoDES", "DES]``. Can provide as tuple for cross-spectra.
    amr : bool
        Whether to apply the angular mode removal (AMR), i.e. to forward model the power loss due to linear angular systematics weights.
    ellsout : list[int] | None
        For which ells the window is computed. Default None and use ellsout extracted from corresponding power spectra. Useful to split the computation.
    regression_maps : list[str] | tuple[list[str], list[str]] | None
        Names of the systematics templates to use for the AMR. Can be set to ``None`` if ``amr=False``. Can provide as tuple for cross-spectra.
    templates_paths_kwargs : dict | tuple[dict, dict] | None
        Keyword arguments to pass to the function loading the templates maps, e.g. paths to the templates files, EBV map, nside, etc. Not needed if ``amr=False``. Must at least contain the keys ``templates_path_N`` and ``templates_path_S`` with the paths to the templates files for the Northern and Southern regions, respectively. Can (must) provide as tuple for cross-spectra.
    amr_regions_zranges : list[tuple[str, tuple[float, float]]] | tuple[list[tuple[str, tuple[float, float]]], list[tuple[str, tuple[float, float]]]] | None
        Regions where to apply the regressions for the AMR, and corresponding redshift ranges. Can be set to ``None`` if ``amr=False``. Can provide as tuple for cross-spectra.
    spectrum_regions_zranges : list[tuple[str, tuple[float, float]]] | None
        Regions for which to compute the window and power spectrum, along with their corresponding redshift ranges. If ``None``, the whole catalog is used as one region. Typically ``[("NGC", (zmin, zmax)), ("SGC", (zmin, zmax))]``.
    total_region_zrange : tuple[str, tuple[float, float]]
        Total region and redshift range to use for the forward model (but not necessarily the spectra). Should at least encompass all the spectrum regions. Should generally be ("ALL", (zmin, zmax)) with the full redshift range of the tracer, with some exceptions for cross-correlations.
    n_realizations : int
        Number of realizations to compute.
    seeds : list[int] | None
        Seeds to use for each realization. If ``None``, defaults to ``2 * i_realization + 3``.
    unitary_amplitude : bool, optional
        Whether to use unitary amplitude for the mock survey mesh generation, by default True.
    batch_size : int, optional
        Number of window computations to run in parallel, by default 4. Depends on the available memory, number of randoms catalogs, size of the mesh... Lower if needed.

    Returns
    -------
    dict[str, dict[str, list[lsstypes.WindowMatrix]]]
        Dictionary, per effect included (geometry, RIC, RIC+AMR) and per region, of lists of window matrices (one per realization).

    Notes
    -----
    * Particles loaded by ``get_data_randoms`` but not present in any of ``spectrum_regions_zranges`` are used for observational effects (RIC, AMR, data-to-randoms ratio renormalization...) but not taken into account in power spectrum computations. In general, ``get_data_randoms`` should load the full footprint and range of redshifts available, *including outside the overlap for cross-correlations*.
    * Power spectrum regions/redshift ranges should not overlap. Overlapping regions require separate calls to this function.
    """
    assert len(seeds) == n_realizations if seeds is not None else True, "If seeds are provided, their number must match n_realizations."
    # Notes to self:
    # * RIC not optional
    # * n_randoms is effectively set by the length of get_data_randoms
    import mpytools as mpy
    from desiwinds.forward import mock_survey_catalog, prepare_AMR, prepare_RIC
    from desiwinds.window import get_window_spikes
    from jaxpower import BinMesh2SpectrumPoles, FKPField, create_sharding_mesh, ParticleField

    from .tools import add_photometric_template_values, select_region

    def _add_photometric_template_values(catalogs: dict[str, mpy.Catalog], regression_maps, templates_paths_kwargs):
        return {name: add_photometric_template_values(catalogs[name], regression_maps, **templates_paths_kwargs) for name in catalogs}

    def _select_region_zrange(catalogs: dict[str, mpy.Catalog], spectrum_region_zrange: tuple[str, tuple[float, float]]) -> dict[str, mpy.Catalog]:
        spectrum_region, zrange = spectrum_region_zrange
        # Mimic read_clustering_catalog: <= on lower, < on upper
        return {name: cat[select_region(ra=cat["RA"], dec=cat["DEC"], region=spectrum_region) & (cat["Z"] >= zrange[0]) & (cat["Z"] < zrange[1])] for name, cat in catalogs.items()}

    def _select_region_zrange_complement(catalogs: dict[str, mpy.Catalog]) -> dict[str, mpy.Catalog]:
        """Select objects outside the spectrum regions and redshift ranges, but inside the total region and redshift range."""
        total_region, total_zrange = total_region_zrange
        return {
            name: cat[
                jnp.invert(
                    jnp.any(
                        jnp.stack(
                            [
                                (select_region(ra=cat["RA"], dec=cat["DEC"], region=spectrum_region) & (cat["Z"] >= zrange[0]) & (cat["Z"] < zrange[1]))
                                for (spectrum_region, zrange) in spectrum_regions_zranges
                            ],
                            axis=0,
                        ),
                        axis=0,
                    )
                )
                & (select_region(ra=cat["RA"], dec=cat["DEC"], region=total_region) & (cat["Z"] >= total_zrange[0]) & (cat["Z"] < total_zrange[1]))
            ]
            for name, cat in catalogs.items()
        }

    def _split_data_randoms(catalogs: dict[str, mpy.Catalog]) -> dict[str, mpy.Catalog]:
        """Split the randoms into "data" and "randoms" based on the provided ratio. Overwrite original "data"."""
        data_size = int(data_to_randoms_ratio * catalogs["randoms"].size)  # MPI local
        randoms_size = catalogs["randoms"].size - data_size
        rng = mpy.random.MPIRandomState(seed=catalog_split_seed, size=catalogs["randoms"].size)  # Use local sizes
        mask_is_data = rng.uniform() < (data_size / (data_size + randoms_size))
        data = catalogs["randoms"][mask_is_data]
        randoms = catalogs["randoms"][~mask_is_data]
        return {"data": data, "randoms": randoms}

    def _safe_divide(a, b):
        return jnp.where(b != 0, a / b, 0.0)

    get_data_randoms = list(get_data_randoms)  # for mutability

    spectrum_regions_zranges = spectrum_regions_zranges or []
    columns_optimal_weights = []
    if optimal_weights is not None:  # FIXME should this be doubled up for cross-spectra?
        columns_optimal_weights += getattr(optimal_weights, "columns", [])  # to compute optimal weights, e.g. for fnl

    if len(get_data_randoms) > 1:  # cross correlation
        # Double up parameters for RIC/AMR if not already provided as tuples
        if isinstance(ric_nbins, int):
            ric_nbins = (ric_nbins, ric_nbins)
        if isinstance(ric_regions, list) and isinstance(ric_regions[0], str):
            ric_regions = (ric_regions, ric_regions)
        if regression_maps is not None and isinstance(regression_maps, list) and isinstance(regression_maps[0], str):
            regression_maps = (regression_maps, regression_maps)
        if templates_paths_kwargs is not None and isinstance(templates_paths_kwargs, dict):
            templates_paths_kwargs = (templates_paths_kwargs, templates_paths_kwargs)
        if amr_regions_zranges is not None and isinstance(amr_regions_zranges, list) and isinstance(amr_regions_zranges[0], tuple) and isinstance(amr_regions_zranges[0][0], str):
            amr_regions_zranges = (amr_regions_zranges, amr_regions_zranges)

        all_regression_maps = list(set(regression_maps[0]) | set(regression_maps[1])) if regression_maps is not None else None
    else:
        all_regression_maps = regression_maps or None

    # Recover output and mesh information from the observable spectrum
    ellsout = ellsout or spectra[0].ells
    los = spectra[0].attrs["los"]  # this has to match with theory input
    if los in ["endpoint", "firstpoint"]:
        los = "local"
    mattrs = {name: spectra[0].attrs[name] for name in ["boxsize", "boxcenter", "meshsize"]}

    with create_sharding_mesh(meshsize=mattrs.get("meshsize", None)):
        if amr:  # Add photometric template values to the catalogs, if AMR is applied, as they are needed for the regression
            # _templates_paths_kwargs depends on the tracer, so do this before further changing get_data_randoms
            # load all regression maps (i.e. for both tracers in case of cross-correlation) ; this simplifies things greatly and useless ones will be dropped later to save memory
            for itracer, (_get_data_randoms, _templates_paths_kwargs) in enumerate(zip(get_data_randoms, templates_paths_kwargs, strict=True)):

                def wrap(f):
                    return lambda: _add_photometric_template_values(f(), all_regression_maps, _templates_paths_kwargs)

                get_data_randoms[itracer] = wrap(_get_data_randoms)

        # Split into "data" and randoms based on the provided ratio
        def wrap(f):
            return lambda: _split_data_randoms(f())

        get_data_randoms = [wrap(_get_data_randoms) for _get_data_randoms in get_data_randoms]

        if len(spectrum_regions_zranges) > 0:
            # Also get particles that might not be in the region selection
            def wrap(f):
                return lambda: _select_region_zrange_complement(f())

            get_extra_data_randoms = [wrap(_get_data_randoms) for _get_data_randoms in get_data_randoms]

            # Split catalogs into pk regions, if specified

            def wrap(f, spectrum_region):
                return lambda: _select_region_zrange(f(), spectrum_region)

            get_data_randoms = [
                wrap(_get_data_randoms, spectrum_region_zrange) for spectrum_region_zrange in spectrum_regions_zranges for _get_data_randoms in get_data_randoms
            ]  # [func1_region1, func2_region1, func3_region1 ... func1_region2, func2_region2, func3_region2 ...]
        else:
            get_extra_data_randoms = []

        all_particles = prepare_jaxpower_particles(
            *get_data_randoms,
            mattrs=mattrs,
            add_randoms=["IDS", "WEIGHT_FKP", "Z", *columns_optimal_weights] + (all_regression_maps if amr else []),
            add_data=["WEIGHT_FKP", "Z", *columns_optimal_weights] + (all_regression_maps if amr else []),
        )

        if get_extra_data_randoms:
            extra_particles = prepare_jaxpower_particles(
                *get_extra_data_randoms,
                mattrs=mattrs,
                add_randoms=["IDS", "WEIGHT_FKP", "Z", *columns_optimal_weights] + (all_regression_maps if amr else []),
                add_data=["WEIGHT_FKP", "Z", *columns_optimal_weights] + (all_regression_maps if amr else []),
                check=False,  # these particles will be outside mattrs but it doesn't matter
            )
        else:
            extra_particles = []

        all_randoms = [particles["randoms"] for particles in all_particles]
        all_data = [particles["data"] for particles in all_particles]
        # Extra data and randoms are needed for observational effects but shouldn't be in spectrum computations
        extra_data = [particles["data"] for particles in extra_particles]
        extra_randoms = [particles["randoms"] for particles in extra_particles]
        del all_particles, extra_particles

        # Make into len(spectrum_regions) catalogs if split into spectrum regions, otherwise one catalog
        nregion = len(spectrum_regions_zranges) if len(spectrum_regions_zranges) > 0 else 1
        nrandoms = len(all_randoms)
        ntracers = nrandoms // nregion
        all_randoms = [[all_randoms[ntracers * i + itracer] for itracer in range(ntracers)] for i in range(nregion)]
        all_data = [[all_data[ntracers * i + itracer] for itracer in range(ntracers)] for i in range(nregion)]
        # [[tracer1_region1, tracer2_region1, ...], [tracer1_region2, tracer2_region2, ...], ...]

        for iregion in range(nregion):
            for itracer in range(len(all_randoms[iregion])):
                # Randoms
                extra = all_randoms[iregion][itracer].extra
                if amr:
                    extra.update({"template_values": jnp.stack([extra.pop(map_name) for map_name in regression_maps[itracer]], axis=-1)})
                    for map in set(all_regression_maps) - set(regression_maps[itracer]):
                        del extra[map]  # remove maps not used for this tracer to save memory
                # extra already has weight_FKP, just remove from weights=indweights which contains FKP weights
                all_randoms[iregion][itracer] = all_randoms[iregion][itracer].clone(extra=extra, weights=_safe_divide(all_randoms[iregion][itracer].weights, extra["WEIGHT_FKP"]))

                # Data
                extra = all_data[iregion][itracer].extra
                if amr:
                    extra.update({"template_values": jnp.stack([extra.pop(map_name) for map_name in regression_maps[itracer]], axis=-1)})
                    for map in set(all_regression_maps) - set(regression_maps[itracer]):
                        del extra[map]  # remove maps not used for this tracer to save memory
                all_data[iregion][itracer] = all_data[iregion][itracer].clone(extra=extra, weights=_safe_divide(all_data[iregion][itracer].weights, extra["WEIGHT_FKP"]))
        del extra

        # Add extra data and randoms with WEIGHT_FKP = 0 to avoid them contributing to the window computation, but still have actual weights in RIC/AMR
        for itracer in range(ntracers):
            # Randoms
            extra = extra_randoms[itracer].extra
            if amr:
                extra.update({"template_values": jnp.stack([extra.pop(map_name) for map_name in regression_maps[itracer]], axis=-1)})
                for map in set(all_regression_maps) - set(regression_maps[itracer]):
                    del extra[map]  # remove maps not used for this tracer to save memory
            # Isolate weights and set WEIGHT_FKP to 0
            extra_randoms[itracer] = extra_randoms[itracer].clone(
                extra=extra | {"WEIGHT_FKP": jnp.zeros_like(extra["WEIGHT_FKP"])}, weights=_safe_divide(extra_randoms[itracer].weights, extra["WEIGHT_FKP"])
            )
            # Concatenate to first randoms catalog for this tracer
            all_randoms[0][itracer] = ParticleField.concatenate([all_randoms[0][itracer], extra_randoms[itracer]])

            # Data
            extra = extra_data[itracer].extra
            if amr:
                extra.update({"template_values": jnp.stack([extra.pop(map_name) for map_name in regression_maps[itracer]], axis=-1)})
                for map in set(all_regression_maps) - set(regression_maps[itracer]):
                    del extra[map]  # remove maps not used for this tracer to save memory
            # Isolate weights and set WEIGHT_FKP to 0
            extra_data[itracer] = extra_data[itracer].clone(
                extra=extra | {"WEIGHT_FKP": jnp.zeros_like(extra["WEIGHT_FKP"])}, weights=_safe_divide(extra_data[itracer].weights, extra["WEIGHT_FKP"])
            )
            # Concatenate to first data catalog for this tracer
            all_data[0][itracer] = ParticleField.concatenate([all_data[0][itracer], extra_data[itracer]])
        del extra, extra_randoms, extra_data

        if jax.process_index() == 0: logger.info("Catalogs ready, starting preparation...")

        # Prepare arguments for the window computation function
        ric_argss = []
        for itracer, (data_tracer, randoms_tracer) in enumerate(zip(zip(*all_data, strict=True), zip(*all_randoms, strict=True), strict=True)):
            ric_argss.append(prepare_RIC(data=data_tracer, randoms=randoms_tracer, regions=ric_regions[itracer], n_bins=ric_nbins[itracer], apply_to="randoms"))
        ric_argss = tuple(ric_argss)

        if amr:
            extra_effects = "RIC+AMR"
            amr_argss = []
            for itracer, (data_tracer, randoms_tracer) in enumerate(zip(zip(*all_data, strict=True), zip(*all_randoms, strict=True), strict=True)):
                amr_argss.append(prepare_AMR(data=data_tracer, randoms=randoms_tracer, regions_zranges=amr_regions_zranges[itracer], apply_to="randoms"))
            amr_argss = tuple(amr_argss)

            def delete_template_values(particles):
                extra = particles.extra
                del extra["template_values"]
                return particles.clone(extra=extra)

            all_data = [[delete_template_values(particles) for particles in data_region] for data_region in all_data]
            all_randoms = [[delete_template_values(particles) for particles in randoms_region] for randoms_region in all_randoms]
        else:
            extra_effects = "RIC"
            amr_argss = None

        # Turn into FKP fields
        fkp_fields = [
            [FKPField(data=d, randoms=r, attrs=mattrs) for d, r in zip(data_region, randoms_region, strict=True)]
            for data_region, randoms_region in zip(all_data, all_randoms, strict=True)
        ]

        del all_data, all_randoms

        if optimal_weights is None:
            if jax.process_index() == 0:
                logger.info("Using FKP weights, computing window for all ells at once.")
            # Using FKP weights which are symetrical, so this remains an autocorr
            binner = BinMesh2SpectrumPoles(fkp_fields[0][0].attrs, edges=spectra[0].get(0).edges("k"), ells=ellsout)  # TODO: check edges are ok
            # get norms from input spectrum
            fkp_norms = [jnp.concatenate([spectrum.get(ell).values("norm") for ell in ellsout], axis=0) for spectrum in spectra]

            # Renormalize data and randoms to input spectrum
            for iregion, (spectrum, fkps) in enumerate(zip(spectra, fkp_fields, strict=True)):
                # autocorrelation and FKP: component [0, 0]
                # autocorrelation and FKP: component [0, itracer]
                for itracer, fkp in enumerate(fkps):
                    # FIXME I think min(itracer, ntracers - 1) always equal to itracer ??
                    alphad = spectrum.get(0).attrs["wsum_data"][0][min(itracer, ntracers - 1)] / (fkp.data.weights * fkp.data.extra["WEIGHT_FKP"]).sum()
                    alphar = spectrum.get(0).attrs["wsum_randoms"][0][min(itracer, ntracers - 1)] / (fkp.randoms.weights * fkp.randoms.extra["WEIGHT_FKP"]).sum()
                    fkp_fields[iregion][itracer] = fkp.clone(
                        data=fkp.data.clone(weights=fkp.data.weights * alphad),
                        randoms=fkp.randoms.clone(weights=fkp.randoms.weights * alphar),
                    )

            # Needed list of lists for mutability before. Now switch to list of tuples to respect FM input signature
            fkp_fields = [tuple(fkp_region) for fkp_region in fkp_fields]

            ## FM based computations
            windows = {}

            # Shared window FM arguments
            window_fm_kw = {
                "mock_survey": mock_survey_catalog,
                "theory": theory,
                "nreal": n_realizations,
                "seeds": seeds,
                "batch_size": batch_size,
                "mock_survey_args": (*fkp_fields,),
                "static_argnames": ["los", "unitary_amplitude", "estimator_weights"],
                "tmpdir": None,  # No temporary output
                "survey_names": [f"{srz[0]}_{srz[1][0]}-{srz[1][1]}" for srz in spectrum_regions_zranges],
            }

            mock_survey_kwargs = {
                "los": los,
                "unitary_amplitude": unitary_amplitude,
                "nam_args": None,
                "fkp_norms": fkp_norms,
                "binner": binner,
                "estimator_weights": "WEIGHT_FKP",
                "data_regions": tuple(ric_arg.data_regions for ric_arg in ric_argss),
                "randoms_regions": tuple(ric_arg.randoms_regions for ric_arg in ric_argss),
            }

            if geo:
                if jax.process_index() == 0: logger.info("Computing geometry window with desiwinds...")
                _, windows["geometry"] = get_window_spikes(
                    **window_fm_kw,
                    mock_survey_kwargs=mock_survey_kwargs
                    | {"ric_args": None, "amr_args": None},
                )

            if ric:
                if jax.process_index() == 0:
                    logger.info("Computing total window (%s) with desiwinds...", extra_effects)
                _, windows[extra_effects] = get_window_spikes(
                    **window_fm_kw,
                    mock_survey_kwargs=mock_survey_kwargs | {"ric_args": ric_argss, "amr_args": amr_argss},
                )

            if jax.process_index() == 0: logger.info("desiwinds window computation finished.")

            for effect in windows:
                windows[effect] = {region_zrange: [windows[effect][ireal][idx] for ireal in range(n_realizations)] for idx, region_zrange in enumerate(spectrum_regions_zranges)}

        else:
            # Optimal weights: non symmetrical, so need to compute "cross-correlation" (same tracer, different weights) + not the same for all ells
            # If actual cross-spectra, need to compute A_w1 x B_w2 and A_w2 x B_w1 -> double the amount of spectra as well (and for each ell)
            # Proceed ell per ell and sum the windows at the end
            if jax.process_index() == 0: logger.info("Using optimal weights, computing windows for each ell separately.")

            # Double up RIC/AMR arguments even for auto-spectra to simplify the logic in the FM call
            if len(ric_argss) == 1:
                ric_argss = ric_argss * 2
                if amr:
                    amr_argss = amr_argss * 2

            # Prepare optimal weigts generators with same structure as fkp_fields
            optimal_weights_data = [
                lambda ell, fkp_region=fkp_region: optimal_weights(  # use default to avoid late binding in the loop
                    ell,
                    [
                        (
                            {column: fkp_field.data.extra[column] for column in ["Z", *columns_optimal_weights]}
                            | {"INDWEIGHT": fkp_field.data.weights * fkp_field.data.extra["WEIGHT_FKP"]}
                        )
                        for fkp_field in fkp_region
                    ],
                )
                for fkp_region in fkp_fields
            ]
            optimal_weights_randoms = [
                lambda ell, fkp_region=fkp_region: optimal_weights(  # use default to avoid late binding in the loop
                    ell,
                    [
                        (
                            {column: fkp_field.randoms.extra[column] for column in ["Z", *columns_optimal_weights]}
                            | {"INDWEIGHT": fkp_field.randoms.weights * fkp_field.randoms.extra["WEIGHT_FKP"]}
                        )
                        for fkp_field in fkp_region
                    ],
                )
                for fkp_region in fkp_fields
            ]

            def _attach_weights(fkp_region: tuple[FKPField], data_w1, data_w2, randoms_w1, randoms_w2):
                """
                Attach optimal weights to FKP fields in fields `'optimal_1'` and `'optimal_2'` for data and randoms.

                If the input fkp_region contains one tracer, both optimal weights are attached to the same tracer and the field is duplicated to have a length 2 output.
                If the input fkp_region contains two tracers, the first optimal weight is attached to the first tracer and the second optimal weight to the second tracer.
                """
                # These weights also contain real weights and FKP weights ; need to remove the real weights to isolate the "estimator weights" to apply at computation time in the FM
                if len(fkp_region) == 1:
                    return (
                        fkp_region[0].clone(
                            data=fkp_region[0].data.clone(
                                extra=fkp_region[0].data.extra
                                | {"weight_optimal_1": _safe_divide(data_w1, fkp_region[0].data.weights), "weight_optimal_2": _safe_divide(data_w2, fkp_region[0].data.weights)}
                            ),
                            randoms=fkp_region[0].randoms.clone(
                                extra=fkp_region[0].randoms.extra
                                | {
                                    "weight_optimal_1": _safe_divide(randoms_w1, fkp_region[0].randoms.weights),
                                    "weight_optimal_2": _safe_divide(randoms_w2, fkp_region[0].randoms.weights),
                                }
                            ),
                        ),
                    ) * 2
                elif len(fkp_region) == 2:
                    return (
                        fkp_region[0].clone(
                            data=fkp_region[0].data.clone(extra=fkp_region[0].data.extra | {"weight_optimal_1": _safe_divide(data_w1, fkp_region[0].data.weights)}),
                            randoms=fkp_region[0].randoms.clone(extra=fkp_region[0].randoms.extra | {"weight_optimal_1": _safe_divide(randoms_w1, fkp_region[0].randoms.weights)}),
                        ),
                        fkp_region[1].clone(
                            data=fkp_region[1].data.clone(extra=fkp_region[1].data.extra | {"weight_optimal_2": _safe_divide(data_w2, fkp_region[1].data.weights)}),
                            randoms=fkp_region[1].randoms.clone(extra=fkp_region[1].randoms.extra | {"weight_optimal_2": _safe_divide(randoms_w2, fkp_region[1].randoms.weights)}),
                        ),
                    )
                else:
                    raise ValueError(f"Unexpected number of tracers in fkp_region: {len(fkp_region)}")

            windows = {}
            if geo:
                windows["geometry"] = {ell: [] for ell in ellsout}
            if ric:
                windows[extra_effects] = {ell: [] for ell in ellsout}

            for ell in ellsout:
                binner = BinMesh2SpectrumPoles(fkp_fields[0][0].attrs, edges=spectra[0].get(ell).edges("k"), ells=[ell])  # TODO: check edges are ok
                # Recover FKP normalization for each region and this ell only from input spectrum
                fkp_norms = [jnp.concatenate([spectrum.get(ill).values("norm") for ill in [ell]], axis=0) for spectrum in spectra]

                for iopt, optweights in enumerate(zip(*[owd(ell) for owd in optimal_weights_data], *[owr(ell) for owr in optimal_weights_randoms], strict=True)):
                    # This loop will iterate once for auto and twice for cross (A_w1 x B_w2 and A_w2 x B_w1)
                    # _fkp_fields lists length 2 tuples (that might be one field duplicated for auto) with the optimal weights attached in the extra
                    _fkp_fields = [_attach_weights(fkp_region, *optweights[iregion], *optweights[nregion + iregion]) for iregion, fkp_region in enumerate(fkp_fields)]

                    # Renormalize data and randoms to input spectrum
                    for iregion, (spectrum, (fkp1, fkp2)) in enumerate(zip(spectra, _fkp_fields, strict=True)):
                        # OQE auto AxA: [[w1sum_A, w2sum_A]] | OQE cross AxB: [[w1sum_A, w2sum_B], [w2sum_A, w1sum_B]]
                        alphad1 = spectrum.get(ell).attrs["wsum_data"][iopt][0] / fkp1.data.clone(weights=fkp1.data.weights * fkp1.data.extra["weight_optimal_1"]).weights.sum()
                        alphad2 = spectrum.get(ell).attrs["wsum_data"][iopt][1] / fkp2.data.clone(weights=fkp2.data.weights * fkp2.data.extra["weight_optimal_2"]).weights.sum()
                        alphar1 = (
                            spectrum.get(ell).attrs["wsum_randoms"][iopt][0]
                            / fkp1.randoms.clone(weights=fkp1.randoms.weights * fkp1.randoms.extra["weight_optimal_1"]).weights.sum()
                        )
                        alphar2 = (
                            spectrum.get(ell).attrs["wsum_randoms"][iopt][1]
                            / fkp2.randoms.clone(weights=fkp2.randoms.weights * fkp2.randoms.extra["weight_optimal_2"]).weights.sum()
                        )
                        # Renormalize the optimal weights so that the final mesh is properly normalized
                        _fkp_fields[iregion] = (
                            fkp1.clone(
                                data=fkp1.data.clone(weights=fkp1.data.weights * alphad1),
                                randoms=fkp1.randoms.clone(weights=fkp1.randoms.weights * alphar1),
                            ),
                            fkp2.clone(
                                data=fkp2.data.clone(weights=fkp2.data.weights * alphad2),
                                randoms=fkp2.randoms.clone(weights=fkp2.randoms.weights * alphar2),
                            ),
                        )

                    # Shared window FM arguments
                    window_fm_kw = {
                        "mock_survey": mock_survey_catalog,
                        "theory": theory,
                        "nreal": n_realizations,
                        "seeds": seeds,
                        "batch_size": batch_size,
                        "mock_survey_args": _fkp_fields,  # list of tuples of FKP fields with different norm and optimal weights
                        "static_argnames": ["los", "unitary_amplitude", "estimator_weights"],
                        "tmpdir": None,  # No temporary output
                        "survey_names": [f"{srz[0]}_{srz[1][0]}-{srz[1][1]}" for srz in spectrum_regions_zranges],
                    }

                    mock_survey_kwargs = {
                        "los": los,
                        "unitary_amplitude": unitary_amplitude,
                        "nam_args": None,
                        "fkp_norms": [jnp.ones_like(fkp_norm) for fkp_norm in fkp_norms],  # will renormalize manually when summing cross corr windows
                        "binner": binner,  # one ell only
                        "estimator_weights": ("weight_optimal_1", "weight_optimal_2"),
                        "data_regions": tuple(ric_arg.data_regions for ric_arg in ric_argss),
                        "randoms_regions": tuple(ric_arg.randoms_regions for ric_arg in ric_argss),
                    }

                    if geo:
                        if jax.process_index() == 0:
                            logger.info("Computing geometry window for ell=%i, optimal weights combination %i with desiwinds...", ell, iopt)
                        _, _windows_fm_geo = get_window_spikes(**window_fm_kw, mock_survey_kwargs=mock_survey_kwargs | {"ric_args": None, "amr_args": None})
                        windows["geometry"][ell].append(_windows_fm_geo)

                    if ric:
                        if jax.process_index() == 0:
                            logger.info("Computing %s window for ell=%i, optimal weights combination %i with desiwinds...", extra_effects, ell, iopt)
                        _, _windows_fm = get_window_spikes(**window_fm_kw, mock_survey_kwargs=mock_survey_kwargs | {"ric_args": ric_argss, "amr_args": amr_argss})
                        windows[extra_effects][ell].append(_windows_fm)

                # Sum over weights iteration (ie for cross AxB, sum A_w1 x B_w2 and A_w2 x B_w1) to get the final window for this ell before summing over ells
                # Also renormalize now
                for effect in windows:
                    windows[effect][ell] = jax.tree.map(
                        (lambda *windows_and_norm: windows_and_norm[0].clone(value=np.sum([wd.value() for wd in windows_and_norm[:-1]], axis=0) / windows_and_norm[-1][..., None])),
                        *windows[effect][ell],
                        [fkp_norms] * n_realizations,
                        is_leaf=lambda x: isinstance(x, types.WindowMatrix),
                    )

            if jax.process_index() == 0: logger.info("desiwinds window computation finished.")

            # For each region, sum the windows over ells and apply control variate
            def _combine_ells(windows):
                observables = [window.observable for window in windows]
                observable = types.join(observables)
                value = np.concatenate([window.value() for window in windows], axis=0)
                return windows[0].clone(value=value, observable=observable)  # join multipoles

            for effect in windows:
                windows[effect] = {
                    region_zrange: [_combine_ells([windows[effect][ell][ireal][idx] for ell in ellsout]) for ireal in range(n_realizations)]
                    for idx, region_zrange in enumerate(spectrum_regions_zranges)
                }

        return windows


def run_preliminary_fit_mesh2_spectrum(data: types.Mesh2SpectrumPoles, window: types.WindowMatrix, select: dict=None, theory: str='rept', fixed=tuple(), out: types.Mesh2SpectrumPoles=None):
    """
    Compute a smooth theory spectrum to assume when building the covariance.

    Parameters
    ----------
    data : Mesh2SpectrumPoles or None
        Measured spectrum multipoles used to build the covariance and (optionally)
        to set priors / initialize the fit. If None, the function will still
        construct an analytic covariance from `window` but cannot use data-driven
        priors.
    window : WindowMatrix
        Window matrix describing mode-coupling of the estimator. The window's
        observable axes are matched to the `data` before fitting.
    select : dict, optional
        If provided, a selection is applied to `data` via `data.select(**select)`
        prior to fitting (e.g. to restrict k-ranges or multipoles).
    theory : str, optional
        Theory to use in the fit, one of ['rept', 'kaiser'].
    out : Mesh2SpectrumPoles, optional
        If provided, returns a clone of these power spectrum multipoles with best fit theory values.

    Returns
    -------
    out : Mesh2SpectrumPoles
    """
    # Import spectrum computation and covariance tools from jaxpower
    from jaxpower import MeshAttrs, compute_spectrum2_covariance
    # Select k-range for fitting (avoid very low k)
    smooth = data.select(k=(0.001, 10.))

    # Create mesh attributes from input data
    mattrs = MeshAttrs(**{name: data.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    # Compute Gaussian covariance (assuming Gaussian density field)
    covariance = compute_spectrum2_covariance(mattrs, data)  # Gaussian, diagonal covariance

    # Apply selection to data (restrict to fitting range)
    select = select or {'k': (0.02, 10.)}
    data = data.select(**select)
    # Match window to data range
    window = window.at.observable.match(data)
    # Restrict window theory to coverage of measurement
    window = window.at.theory.select(k=(0.001, 1.2 * next(iter(data)).coords('k').max()))
    # Match covariance to data range
    covariance = covariance.at.observable.match(data)
    # Extract effective redshift from window
    z = window.observable.get(ells=0).attrs['zeff']

    # Import clustering theory classes from desilike
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, KaiserTracerPowerSpectrumMultipoles, REPTVelocileptorsTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler

    # Select theory model (Kaiser or REPT with velocileptors)
    Theory = {'rept': REPTVelocileptorsTracerPowerSpectrumMultipoles, 'kaiser': KaiserTracerPowerSpectrumMultipoles}[theory]

    # Create fiducial theory template at measurement redshift
    template = FixedPowerSpectrumTemplate(fiducial='DESI', z=z)
    # Instantiate theory model with template
    theory = Theory(template=template)
    # Create observable: combines data, theory, and window function
    observable = TracerPowerSpectrumMultipolesObservable(data=data, window=window, theory=theory)
    # Create likelihood: Gaussian likelihood with computed covariance
    likelihood = ObservablesGaussianLikelihood(observable, covariance=covariance.value())
    # Fix specified parameters
    for param in fixed:
        likelihood.all_params[param].update(fixed=True)

    # Minimize likelihood to get best-fit theory
    profiler = MinuitProfiler(likelihood, seed=42)
    profiles = profiler.maximize()
    # Get best-fit parameters
    params = profiles.bestfit.choice(index='argmax', input=True)
    if out is None:
        # Build smooth theory spectrum from best-fit parameters
        poles = []
        for ill, ell in enumerate(theory.ells):
            if ell in smooth.ells:
                # Use original smooth data as template
                pole = smooth.get(ells=ell)
            else:
                # Create new pole with zero shot noise for missing multipoles
                pole = smooth.get(ells=0).clone(meta={"ell": ell})
                if ell != 0:
                    # Zero out shot noise for higher multipoles (only monopole has shot noise)
                    pole = pole.clone(num_shotnoise=np.zeros_like(pole.values("num_shotnoise")))
            # Evaluate theory at k-values from data
            theory.init.update(k=pole.coords("k"))
            value = theory(**params)[ill]
            pole = pole.clone(value=value)
            poles.append(pole)
        # Build spectrum object from theory poles
        smooth = types.Mesh2SpectrumPoles(poles, attrs=smooth.attrs)
    else:
        # Use provided output structure
        value = []
        for label, pole in out.items(level=1):
            # Evaluate theory at k-values
            theory.init.update(k=pole.coords('k'))
            value.append(theory(**params)[theory.ells.index(label['ells'])])
        smooth = out.clone(value=value)
    return smooth


def compute_covariance_mesh2_spectrum(*get_data_randoms, theory=None, fields=None, mattrs=None):
    r"""
    Compute the 2-point spectrum covariance with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data' and 'randoms' catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    theory : Mesh2SpectrumPoles
        Theory 2-point spectrum multipoles.
    fields : tuple, list, optional
        Field names.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'. If ``None``, default attributes are used.

    Returns
    -------
    covarance : CovarianceMatrix
        The computed 2-point spectrum covariance.
    """
    # Import covariance and window computation tools from jaxpower
    from jaxpower import create_sharding_mesh, compute_fkp2_covariance_window, interpolate_window_function, compute_spectrum2_covariance, FKPField
    # Use FFTLog for reliable correlation-to-spectrum conversion
    fftlog = True
    # Use default fields (1, 2, ...) if not provided
    if fields is None:
        fields = list(range(1, 1 + len(get_data_randoms)))

    results = {}
    # Set up distributed computation mesh
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        # Load and prepare particles
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'])
        # Create FKP fields for covariance window computation
        all_fkp = [FKPField(particles['data'], particles['randoms']) for particles in all_particles]
        mattrs = all_fkp[0].attrs
        # Set correlation binning parameters (finer than spectrum binning)
        kw = dict(edges={'step': mattrs.cellsize.min()}, basis='bessel') if fftlog else dict(edges={})
        # Add fields for cross-covariance and random splitting seed
        kw.update(los='local', fields=fields, split=[(42, fkp.randoms.extra['IDS']) for fkp in all_fkp])
        # Mesh painting parameters: TSC with interlacing
        kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
        # Compute covariance window function (correlation in configuration space)
        windows = compute_fkp2_covariance_window(all_fkp, **kw, **kw_paint)
        #if jax.process_index() == 0: windows.write(f'_tests/window_correlation.h5')
        if fftlog:
            # Very robust to this choice of FFTLog grid
            # Use logarithmic s-grid for interpolation
            coords = np.logspace(-2, 8, 8 * 1024)
            # Interpolate window functions to fine s-grid
            windows = windows.map(lambda window: interpolate_window_function(window, coords=coords), level=1)
        # Store raw correlation windows for diagnostics
        results['window_covariance_mesh2_correlation'] = windows

    # Convert correlation to power spectrum covariance matrix via FFTLog
    covariance = compute_spectrum2_covariance(windows, theory, flags=['smooth'] + (['fftlog'] if fftlog else []))
    # Update label names to match observable structure
    fields = covariance.observable.fields
    # Create observable tree with proper labels
    observable = types.ObservableTree(list(covariance.observable), observables=['spectrum2'] * len(fields), tracers=fields)
    covariance = covariance.clone(observable=observable)
    # Store in results dict
    results['raw'] = covariance
    return results


def compute_rotation_mesh2_spectrum(window: types.WindowMatrix, covariance: types.CovarianceMatrix, Minit: str='momt',
                                    data: types.Mesh2SpectrumPoles=None, theory: types.Mesh2SpectrumPoles=None, select: dict=None):
    """
    Compute the rotation to make the window matrix more diagonal.

    Parameters
    ----------
    window : WindowMatrix
        Window matrix.
    covariance : CovarianceMatrix
        Covariance of the measured spectrum.
    Minit : {'momt', ...}, optional
        Initialization method passed to rotation.setup(Minit=...). Defaults to 'momt'.
    data : Mesh2SpectrumPoles or None, optional
        Measured spectrum used to set priors for the rotation (if available).
    theory : Mesh2SpectrumPoles or None, optional
        Theory spectrum used together with `data` when setting priors.

    Returns
    -------
    rotation : WindowRotationSpectrum2
    """
    # Import rotation matrix computation from jaxpower
    from jaxpower import WindowRotationSpectrum2
    # Extract observable from window or data
    observable = window.observable
    if data is not None:
        # Use data as observable instead of window
        if select is not None:
            data = data.select(**select)
        observable = data
    # Match window observable to target observable (reorder/subset as needed)
    window = window.at.observable.match(observable)
    if theory is not None:
        def interpolate_pole(ref, pole):
            # Interpolate theory to match reference k-values
            return ref.clone(value=np.interp(ref.coords('k'), pole.coords('k'), pole.value()))

        # Interpolate theory to window theory k-values
        theory = window.theory.map(lambda pole, label: interpolate_pole(pole, theory.get(ells=label['ells'])), input_label=True, level=1)
    # Match covariance observable to target observable
    covariance = covariance.at.observable.match(observable)
    # Create rotation matrix object
    rotation = WindowRotationSpectrum2(window=window, covariance=covariance, xpivot=0.1)
    # Set up rotation matrix (initialize using 'momt' method: moment-based initialization)
    rotation.setup(Minit=Minit)
    # Fit rotation matrix to data (if provided)
    rotation.fit()
    if rotation.with_momt and data is not None:
        # To set up priors for rotation parameters from data
        rotation.set_prior(data=data, theory=theory)
    return rotation


def compute_box_mesh2_spectrum(*get_data, ells=(0, 2, 4), edges=None, los='z', cache=None, mattrs=None):
    r"""
    Compute the 2-point spectrum multipoles for a cubic box using :mod:`jaxpower`.

    Parameters
    ----------
    get_data : callables
        Functions that return dict of 'data' (optionally 'shifted') catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'.
    ells : list of int, optional
        List of multipole moments to compute. Default is (0, 2, 4).
    edges : dict, optional
        Edges for the binning; array or dictionary with keys 'start' (minimum :math:`k`), 'stop' (maximum :math:`k`), 'step' (:math:`\Delta k`).
        If ``None``, default step of :math:`0.001 h/\mathrm{Mpc}` is used.
        See :class:`jaxpower.BinMesh2SpectrumPoles` for details.
    los : {'x', 'y', 'z', array-like}, optional
        Line-of-sight direction. If 'x', 'y', 'z' use fixed axes, or provide a 3-vector.
    cache : dict, optional
        Cache to store binning class (can be reused if ``meshsize`` and ``boxsize`` are the same).
        If ``None``, a new cache is created.

    Returns
    -------
    spectrum : Mesh2SpectrumPoles
        The computed 2-point spectrum multipoles.
    """
    # Import tools for periodic box power spectrum computation
    from jaxpower import (create_sharding_mesh, FKPField, compute_fkp2_shotnoise, compute_box2_normalization, BinMesh2SpectrumPoles, compute_mesh2_spectrum, compute_fkp2_shotnoise)

    mattrs = mattrs or {}
    # Set up distributed computation across JAX devices
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        # Load and prepare particles (data + optional shifted for RSD distortions)
        all_particles = prepare_jaxpower_particles(*get_data, mattrs=mattrs)
        # Initialize or retrieve cached binning object
        if cache is None: cache = {}
        # Set default k-space binning step (0.001 h/Mpc)
        if edges is None: edges = {'step': 0.001}
        # Gather particle attributes (weights, mesh info)
        attrs = _get_jaxpower_attrs(*all_particles)
        # Store line-of-sight direction
        attrs.update(los=los)
        mattrs = all_particles[0]['data'].attrs

        # Define the binner for k-space binning
        key = 'bin_mesh2_spectrum_{}'.format('_'.join(map(str, ells)))
        bin = cache.get(key, None)
        # Create new binning if not cached or mesh changed
        if bin is None or not np.all(bin.mattrs.meshsize == mattrs.meshsize) or not np.allclose(bin.mattrs.boxsize, mattrs.boxsize):
            bin = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)
        # Store binning in cache
        cache.setdefault(key, bin)

        # Computing normalization for periodic box (simpler than survey: no randoms)
        all_data = [particles['data'] for particles in all_particles]
        norm = compute_box2_normalization(*all_data, bin=bin)

        # Computing shot noise from shifted or data catalogs
        # shifted if reconstruction
        all_fkp = [FKPField(particles['data'], particles['shifted']) if particles.get('shifted', None) is not None else particles['data'] for particles in all_particles]
        # Free memory
        del all_particles
        num_shotnoise = compute_fkp2_shotnoise(*all_fkp, bin=bin, fields=None)

        # Paint particles on mesh grid
        kw = dict(resampler='tsc', interlacing=3, compensate=True)
        # out='real' to save memory (store as real arrays)
        all_mesh = []
        for fkp in all_fkp:
            # Paint particle density on mesh
            mesh = fkp.paint(**kw, out='real')
            all_mesh.append(mesh - mesh.mean())
        # Free FKP field memory
        del all_fkp
        # JIT the mesh-based spectrum computation; helps with memory footprint
        jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'])
        #jitted_compute_mesh2_spectrum = compute_mesh2_spectrum
        # Compute power spectrum from painted meshes via FFT
        spectrum = jitted_compute_mesh2_spectrum(*all_mesh, bin=bin, los=los)
        # Attach normalization and shot noise
        spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
        # Propagate attributes to output
        spectrum = spectrum.map(lambda pole: pole.clone(attrs=attrs))
        spectrum = spectrum.clone(attrs=attrs)
        # Wait for computation to complete
        jax.block_until_ready(spectrum)
        if jax.process_index() == 0:
            logger.info('Mesh-based computation finished')
    return spectrum


def compute_window_box_mesh2_spectrum(spectrum: types.Mesh2SpectrumPoles, zsnap: float=None):
    r"""
    Compute the 2-point spectrum window for a box (i.e., binning window) with :mod:`jaxpower`.

    Parameters
    ----------
    spectrum : Mesh2SpectrumPoles
        Measured 2-point spectrum multipoles.

    Returns
    -------
    window : WindowMatrix
        The computed 2-point spectrum window.
    """
    # Compute binning window for periodic box power spectrum
    from jaxpower import create_sharding_mesh, MeshAttrs, BinMesh2SpectrumPoles, compute_mesh2_spectrum_window

    # Extract mesh attributes and multipoles from input spectrum
    mattrs = {name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    los = spectrum.attrs['los']
    ells = spectrum.ells
    pole = spectrum.get(0)
    # Set up distributed computation
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        mattrs = MeshAttrs(**mattrs)
        # Create binning from spectrum edges
        bin = BinMesh2SpectrumPoles(mattrs, edges=pole.edges('k'), ells=ells)
        #edgesin = np.linspace(bin.edges.min(), bin.edges.max(), 2 * (len(bin.edges) - 1))
        # Use input spectrum bins as theory bins
        edgesin = bin.edges
        # For box, window is just the binning matrix (no mode coupling from survey effects)
        window = compute_mesh2_spectrum_window(mattrs, edgesin=edgesin, ellsin=ells, los=los, bin=bin)
        observable = window.observable
        # Attach redshift/snapshot information if provided
        if zsnap is not None:
            observable = observable.map(lambda pole: pole.clone(attrs=pole.attrs | dict(zeff=zsnap, zsnap=zsnap)))
        window = window.clone(observable=observable)
    return window


def compute_covariance_box_mesh2_spectrum(theory: types.Mesh2SpectrumPoles=None, mattrs=None):
    r"""
    Compute the 2-point spectrum covariance for a box with :mod:`jaxpower`.

    Parameters
    ----------
    theory : Mesh2SpectrumPoles, optional
        Theory spectrum used together with `spectrum` when setting priors.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'. If ``None``, default attributes are used.

    Returns
    -------
    covarance : CovarianceMatrix
        The computed 2-point spectrum covariance.
    """
    # Compute Gaussian covariance for periodic box power spectrum
    from jaxpower import create_sharding_mesh, MeshAttrs, compute_spectrum2_covariance
    # Add zero shot noise to theory for covariance computation
    theory_sn = theory.map(lambda pole: pole.clone(num_shotnoise=pole.values('num_shotnoise') * 0.), level=2)
    mattrs = mattrs or {}
    # Set up distributed computation
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        mattrs = MeshAttrs(**mattrs)
        # Compute Gaussian, diagonal covariance
        covariance = compute_spectrum2_covariance(mattrs, theory_sn)  # Gaussian, diagonal covariance

        # Update label names to match observable structure
        fields = covariance.observable.fields
        # Create observable tree with proper labels
        observable = types.ObservableTree(list(covariance.observable), observables=['spectrum2'] * len(fields), tracers=fields)
        covariance = covariance.clone(observable=observable)
    return covariance
