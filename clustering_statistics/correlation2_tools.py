"""
Configuration-space 2-point clustering measurements.

Main functions
--------------
* `compute_particle2_correlation`: Measure the cutsky 2PCF from pair counts (includes jackknife utility).
* `compute_particle2_angular_upweights`: Derive angular upweights for fiber-collision mitigation.
* `compute_box_particle2_correlation`: Measure the 2PCF in cubic boxes.
"""

import logging
import itertools
from functools import partial
import warnings

import numpy as np
import jax

import lsstypes as types
from .tools import _format_bitweights


logger = logging.getLogger('particle2_correlation')


def compute_particle2_angular_upweights(*get_data):
    """
    Compute angular upweights (AUW) from fibered and parent data catalogs.

    Parameters
    ----------
    get_data : callables
        Functions returning dicts with 'fibered_data' and 'parent_data'.
        Catalogs must contain 'RA', 'DEC', 'INDWEIGHT', optionally 'BITWEIGHT'.

    Returns
    -------
    auw : ObservableTree
        Angular upweights as an ObservableTree with 'DD' leaf.
    """
    from cucount.jax import BinAttrs, create_sharding_mesh
    from lsstypes import ObservableLeaf, ObservableTree

    with create_sharding_mesh():
        all_particles = prepare_cucount_particles(*get_data, positions_type='rd')
        all_fibered_particles = [{'data': particles['fibered_data']} for particles in all_particles]
        all_parent_particles = [{'data': particles['parent_data']} for particles in all_particles]
        if jax.process_index() == 0:
            logger.info('All particles on the device')

        theta = 10**np.arange(-5, -1 + 0.1, 0.1)
        battrs = BinAttrs(theta=theta)

        counts_fibered = _compute_particle2_correlation_close_pair_correction(all_fibered_particles, battrs, auw=None, cut=None, normalize_randoms=False)
        counts_parent = _compute_particle2_correlation_close_pair_correction(all_parent_particles, battrs, auw=None, cut=None, normalize_randoms=False)

    kw = dict(theta=battrs.coords('theta'), theta_edges=battrs.edges('theta'), coords=['theta'])
    auw = {}
    for combinations in counts_fibered:
        auw[combinations] = ObservableLeaf(value=np.where(counts_fibered[combinations].value() == 0., 1., counts_parent[combinations].value() / counts_fibered[combinations].value()), **kw)
        auw[combinations + 'parent'] = counts_parent[combinations]
        auw[combinations + 'fibered'] = counts_fibered[combinations]

    return ObservableTree(list(auw.values()), pairs=list(auw.keys()))


def prepare_cucount_particles(*get_data_randoms, positions_type='pos', subsampler=None, jackknife=None, split_randoms=False, concatenate=False, wattrs=None):
    """
    Convert catalogs to :class:`cucount.Particles`.

    Parameters
    ----------
    get_data_randoms : callables
    positions_type : str, optional
        'pos' (default, uses 'POSITION') or 'rd' (uses 'RA', 'DEC').
    subsampler : optional
        Optional object with a ``label(positions)`` method.
    jackknife : dict, optional
        Jackknife configuration. If provided, build a KMeansSubsampler.
    split_randoms : bool, float
        If provided, ratio of randoms / data to split the (concatenated) randoms or shifted catalogs into.
    concatenate : bool
        If ``True``, return concatenated randoms or shifted catalogs.
    wattrs : WeightAttrs, optional
        Weight attributes passed to KMeansSubsampler.

    Returns
    -------
    all_particles : list of dict
    subsampler : optional
    """
    import numpy as np
    from cucount.jax import Particles

    if jackknife is None: jackknife = {}
    kw_jackknife = dict(jackknife)
    if kw_jackknife: kw_jackknife = {'mode': 'angular', 'nsplits': 60, 'nside': 512, 'random_state': 42} | kw_jackknife

    def get_pw(catalog):
        catalog = catalog.all_to_all()  # loadbalance
        if positions_type == 'rd':
            positions = (catalog['RA'], catalog['DEC'])
        else:
            positions = catalog['POSITION']
        weights = [catalog['INDWEIGHT']] + _format_bitweights(catalog['BITWEIGHT'] if 'BITWEIGHT' in catalog else None)
        return positions, weights

    def _is_list(catalog): return isinstance(catalog, (tuple, list))

    if kw_jackknife and subsampler is None:
        import cucount
        from cucount.jax import SplitAttrs, WeightAttrs
        from cucount.utils import KMeansSubsampler
        if wattrs is None: wattrs = WeightAttrs()

        jackknife_particles = []
        for _get_data_randoms in get_data_randoms:
            data = cucount.numpy.Particles(*get_pw(_get_data_randoms()['data'].gather(mpiroot=None)))
            jackknife_particles.append(data)

        jackknife_particles = cucount.numpy.Particles.concatenate(jackknife_particles)
        subsampler = KMeansSubsampler(jackknife_particles, wattrs=wattrs, **kw_jackknife)

    def _concatenate(catalog, collective=False):
        if _is_list(catalog):
            if len(catalog) == 1:
                catalog = catalog[0]
            else:
                if collective:
                    catalog = catalog[0].cconcatenate(catalog)
                else:
                    catalog = catalog[0].concatenate(catalog)
        return catalog

    def _split_catalog(catalog, split_randoms, data_size):
        from mpytools.random import MPIRandomState
        if data_size is None:
            raise ValueError('split_randoms is in terms of data size, so provide data')
        catalog = _concatenate(catalog, collective=True)
        csize = catalog.csize
        if isinstance(split_randoms, tuple):
            split_size, nsplits = split_randoms[0] * data_size, split_randoms[1]
            frac = (split_size * nsplits) / csize
            if frac > 1.:
                warnings.warn(f'catalog of randoms is {1. / frac:.2f} too small to perform {nsplits:d} nsplits with {split_randoms[0]:.2f} the data size')
            frac = min(frac, 1.)
        else:
            split_size = split_randoms * data_size
            nsplits = max(int(csize / split_size), 1)
            frac = 1.
        rng = MPIRandomState(seed=42, size=catalog.size)
        x = rng.uniform(0., 1.)
        toret = []
        for isplit in range(nsplits):
            mask = (x >= isplit * frac / nsplits) & (x < (isplit + 1) * frac / nsplits)
            toret.append(catalog[mask])
        return toret

    def get_all_particles(catalog, as_list=False, data_size=None):
        if as_list and not _is_list(catalog):
            catalog = [catalog]
        if as_list and split_randoms:
            catalog = _split_catalog(catalog, split_randoms, data_size)
        if _is_list(catalog):
            if concatenate:
                catalog = _concatenate_catalog(catalog)
            else:
                return [get_all_particles(c) for c in catalog]
        positions, weights = get_pw(catalog)
        splits = None if subsampler is None else subsampler.label(positions).astype('i8')
        return Particles(positions, weights=weights, splits=splits, positions_type=positions_type, exchange=True)

    all_particles = []
    for _get_data_randoms in get_data_randoms:
        catalogs = _get_data_randoms()
        data_size = catalogs['data'].csize if 'data' in catalogs else None
        particles = {}
        for name, catalog in catalogs.items():
            particles[name] = get_all_particles(catalog, as_list=name in ['randoms', 'shifted'], data_size=data_size)
        all_particles.append(particles)

    if subsampler is not None:
        return all_particles, subsampler
    return all_particles


def _guess_wattrs(get_data_randoms, auw=None):
    """Guess WeightAttrs from input catalogs (callable or Particles)."""
    from cucount.jax import WeightAttrs
    bitwise = angular = None
    if callable(get_data_randoms):
        catalogs = get_data_randoms()
        if 'BITWEIGHT' in catalogs['data']:
            bitwise = dict(weights=_format_bitweights(catalogs['data']['BITWEIGHT']))
            if jax.process_index() == 0:
                logger.info(f'Applying PIP weights {bitwise}.')
    else:  # Particles
        bitwise = get_data_randoms.get('bitwise_weight')

    if auw is not None:
        angular = dict(sep=auw.get('DD').coords('theta'), weight=auw.get('DD').value())
        if jax.process_index() == 0:
            logger.info(f'Applying AUW {angular}.')

    wattrs = WeightAttrs(bitwise=bitwise, angular=angular)
    return wattrs


def compute_particle2_correlation(*get_data_randoms, auw=None, cut=None, battrs: dict=None, zeff: dict=None, jackknife: dict=None, split_randoms: bool | float=False):
    """
    Compute two-point correlation function using :mod:`cucount.jax`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions returning dicts with 'data', 'randoms' (optionally 'shifted').
        Catalogs must contain 'POSITION', 'INDWEIGHT', optionally 'BITWEIGHT'.
    auw : ObservableTree, optional
        Angular upweights to apply.
    cut : bool, optional
        If provided, apply a theta-cut of (0, 0.05) degrees.
    battrs : dict, optional
        Bin attributes for cucount.jax.BinAttrs.
    zeff : dict, optional
        Effective redshift parameters.
    jackknife : dict, optional
        Jackknife configuration. If non-empty, use angular K-means jackknife splits.

    Returns
    -------
    correlation : Count2Correlation or Count2JackknifeCorrelation
    """
    from cucount.jax import BinAttrs, WeightAttrs, SelectionAttrs, SplitAttrs
    from cucount.types import count2
    from lsstypes import Count2Correlation, Count2JackknifeCorrelation
    from jaxpower import create_sharding_mesh
    from .spectrum2_tools import prepare_jaxpower_particles, compute_fkp_effective_redshift

    if zeff is None: zeff = {'boxpad': 1.1, 'cellsize': 10.}
    kw_zeff = dict(zeff)

    if jackknife is None: jackknife = {}
    kw_jackknife = dict(jackknife)

    def merge_randoms(catalog):
        if not isinstance(catalog, (tuple, list)): return catalog
        return catalog[0].concatenate(catalog)

    get_randoms = [lambda f=f: {'randoms': merge_randoms(f()['randoms'])} for f in get_data_randoms]

    with create_sharding_mesh(meshsize=kw_zeff.get('meshsize', None)):
        all_particles = prepare_jaxpower_particles(*get_randoms, mattrs=kw_zeff, add_randoms=['IDS'])
        all_randoms = [particles['randoms'] for particles in all_particles]
        seed = [(42, randoms.extra['IDS']) for randoms in all_randoms]
        zeff, norm_zeff = compute_fkp_effective_redshift(*all_randoms, order=2, split=seed,
                                                         resampler='cic', return_fraction=True)

    key = 'raw'
    with create_sharding_mesh():
        if battrs is None:
            battrs = dict(s=np.linspace(0., 180., 181), mu=(np.linspace(-1., 1., 201), 'midpoint'))
        battrs = BinAttrs(**battrs)

        sattrs = None
        if cut is not None:
            sattrs = SelectionAttrs(theta=(0.05, 180.))
            if jax.process_index() == 0: logger.info(f'Applying theta-cut {sattrs}.')
            key = 'cut'
        if auw is not None:
            key = 'auw'

        wattrs = _guess_wattrs(get_data_randoms[0], auw=auw)
        mattrs = None

        spattrs = None
        all_particles = prepare_cucount_particles(*get_data_randoms, jackknife=kw_jackknife, split_randoms=split_randoms, wattrs=wattrs)
        if isinstance(all_particles, tuple):
            all_particles, subsampler = all_particles
            spattrs = SplitAttrs(mode='jackknife', nsplits=subsampler.nsplits)

        if jax.process_index() == 0: logger.info('All particles on the device')

        def count2split(*particles, wattrs=None):
            kw = dict(battrs=battrs, mattrs=mattrs, sattrs=sattrs, spattrs=spattrs, wattrs=wattrs)
            nsplits = [len(p) if isinstance(p, list) else 0 for p in particles]
            if any(nsplits):
                nsplit = next(n for n in nsplits if n)
                particles = list(particles)
                for ip, particle in enumerate(particles):
                    if isinstance(particle, list):
                        assert len(particle) == nsplit
                    else:
                        particles[ip] = [particle] * nsplit
                counts = [count2(*p, **kw)['weight'] for p in zip(*particles)]
                return types.sum(counts)
            return count2(*particles, **kw)['weight']

        counts = {}

        if jax.process_index() == 0: logger.info('Computing DD term.')
        counts['DD'] = count2split(*[particles['data'] for particles in all_particles], wattrs=wattrs)

        for particles in all_particles:
            particles['data'] = particles['data'].clone(weights=wattrs(particles['data']))

        for combinations in ['DS', 'SD', 'SS']:
            particles, _combinations = _get_particle_combinations(combinations, all_particles, with_repeats=False)
            if jax.process_index() == 0:
                logger.info(f'Computing {_combinations} term.')
            counts[combinations] = count2split(*particles)

        if 'RR' not in counts:
            if all(particles.get('shifted', None) is None for particles in all_particles):
                counts['RR'] = counts['SS']
            else:
                if jax.process_index() == 0: logger.info('Computing RR term.')
                counts['RR'] = count2split(*[particles['randoms'] for particles in all_particles])

    correlation = (Count2JackknifeCorrelation if kw_jackknife else Count2Correlation)(estimator='landyszalay', **counts)
    correlation.attrs.update(zeff=zeff / norm_zeff, norm_zeff=norm_zeff)    
    return {key: correlation}



def _get_particle_combinations(combinations, all_particles, with_repeats=True):
    particles, _combinations, keys = [], '', []
    for icomb, comb in enumerate(combinations):
        key = {'D': 'data', 'S': 'shifted'}[comb]
        icomb = min(len(all_particles) - 1, icomb)
        particle = all_particles[icomb].get(key, None)
        if key == 'shifted':
            if particle is None:
                key, particle = 'randoms', all_particles[icomb]['randoms']
                _combinations += 'R'
            else:
                _combinations += 'S'
        else:
            _combinations += 'D'
        key = (icomb, key)
        if not with_repeats and key in keys:
            continue
        keys.append(key)
        particles.append(particle)
    return particles, _combinations


def compute_box_particle2_correlation(*get_data, battrs: dict=None, mattrs: dict=None, split_randoms: bool | float=False):
    """
    Compute periodic-box two-point correlation function using :mod:`cucount.jax`.

    Parameters
    ----------
    get_data : callables
        Functions returning dicts with 'data' and optionally 'shifted'.
    battrs : dict, optional
        Bin attributes for cucount.jax.BinAttrs.
    mattrs : dict, optional
        Mesh attributes, typically with 'boxsize' and 'boxcenter'.

    Returns
    -------
    correlation : Count2Correlation
    """
    from cucount.jax import BinAttrs, WeightAttrs, MeshAttrs, create_sharding_mesh
    from cucount.types import count2, count2_analytic
    from lsstypes import Count2Correlation

    with create_sharding_mesh():
        if battrs is None:
            battrs = dict(s=np.linspace(0., 180., 181), mu=(np.linspace(-1., 1., 201), 'z'))
        battrs = BinAttrs(**battrs)

        mattrs = mattrs or {}
        wattrs = WeightAttrs()

        all_particles = prepare_cucount_particles(*get_data, split_randoms=split_randoms)
        if jax.process_index() == 0: logger.info('All particles on the device')

        mattrs = MeshAttrs(*[particles['data'] for particles in all_particles], battrs=battrs, los=los, **mattrs)

        def count2split(*particles, wattrs=None):
            kw = dict(battrs=battrs, mattrs=mattrs, wattrs=wattrs)
            nsplits = [len(p) if isinstance(p, list) else 0 for p in particles]
            if any(nsplits):
                nsplit = next(n for n in nsplits if n)
                particles = list(particles)
                for ip, particle in enumerate(particles):
                    if isinstance(particle, list):
                        assert len(particle) == nsplit
                    else:
                        particles[ip] = [particle] * nsplit
                counts = [count2(*p, **kw)['weight'] for p in zip(*particles)]
                return types.sum(counts)
            return count2(*particles, **kw)['weight']

        if jax.process_index() == 0:
            logger.info('Computing DD term.')
        DD = count2split(*[particles['data'] for particles in all_particles], wattrs=wattrs)

        for particles in all_particles:
            particles['data'] = particles['data'].clone(weights=wattrs(particles['data']))

        if jax.process_index() == 0: logger.info('Computing analytic RR term.')
        RR = count2_analytic(battrs=battrs, mattrs=mattrs)

        counts = dict(DD=DD, RR=RR)

        for combinations in ['DS', 'SD', 'SS']:
            particles, _combinations = _get_particle_combinations(combinations, all_particles)
            if not particles:
                continue
            if jax.process_index() == 0:
                logger.info(f'Computing {combinations} term.')
            counts[combinations] = count2split(*particles)

    if all(name in counts for name in ['DS', 'SD', 'SS']):
        return Count2Correlation(estimator='landyszalay', **counts)
    return Count2Correlation(estimator='natural', DD=counts['DD'], RR=counts['RR'])


def compute_particle2_correlation_close_pair_correction(*get_data_randoms, correlation, battrs=None, auw=None, cut=None, jackknife=None,  split_randoms: bool | float=False, **kwargs):
    """Compute and apply close-pair corrections."""

    from cucount.jax import create_sharding_mesh, BinAttrs

    if jackknife is None: jackknife = {}
    kw_jackknife = dict(jackknife)

    with create_sharding_mesh() as sharding_mesh:
        if callable(get_data_randoms[0]):
            spattrs = None
            wattrs = _guess_wattrs(get_data_randoms[0], auw=auw)
            all_particles = prepare_cucount_particles(*get_data_randoms, jackknife=kw_jackknife, wattrs=wattrs, split_randoms=split_randoms)
            if isinstance(all_particles, tuple):
                all_particles, subsampler = all_particles
                spattrs = SplitAttrs(mode='jackknife', nsplits=subsampler.nsplits)
            if jax.process_index() == 0: logger.info('All particles on the device')

        if battrs is None:
            # Try to find battrs from lsstypes object
            edges = correlation.edges()
            ells = getattr(correlation.get('DD'), 'ells', [])
            d = {}
            los = 'midpoint'  # guessing line-of-sight
            for coord_name in ['s', 'theta']:
                coord_name = f'{name}{idim + 1}'
                if coord_name in edges:
                    edge = edges[coord_name]
                    edge = np.append(edge[:, 0], edge[-1, 1])
                    if name == 'mu':
                        edge = (edge, los)
                    d[name] = edge
            if ells:
                d['pole'] = (ells, los)
            battrs = BinAttrs(**d)
        else:
            battrs = BinAttrs(**battrs)
        results = {}
        results['raw'] = correlation
        corrections = {'auw': auw, 'cut': cut}
        for correction_name in corrections:
            if corrections[correction_name] is not None:
                correction = _compute_particle2_correlation_close_pair_correction(all_particles, battrs, **{correction_name: corrections[correction_name]}, spattrs=spattrs, normalize_randoms=False)
                results[correction_name] = _apply_particle2_correlation_close_pair_correction(correlation, correction)

    return results


def _apply_particle2_correlation_close_pair_correction(correlation, correction):
    """Apply additive corrections to a :class:`Count2Correlation`."""
    out = correlation.copy()

    def sum_counts(leaves):
        return leaves[0].clone(counts=sum(leaf.values('counts') for leaf in leaves), norm=leaves[0].values('norm'))

    _branches = []
    for label, branch in correlation.items(level=1):
        correction_count_name = label['count_names'].replace('S', 'R')
        if correction_count_name in correction:
            branch = types.tree_map(sum_counts, [branch, correction[correction_count_name]], level=None, is_leaf=lambda *args: False)
        _branches.append(branch)

    out._branches = _branches
    return out


def _compute_particle2_correlation_close_pair_correction(all_particles, battrs, spattrs=None, auw=None, cut=None, normalize_randoms: bool=True):
    """
    Compute close-pair corrections to two-point counts.

    Parameters
    ----------
    all_particles : list
        Output of :func:`prepare_cucount_particles`.
    battrs : BinAttrs or list
        Bin attributes for the three sides.
    auw : optional
        Angular upweighting object.
    cut : optional
        If provided, compute direct cut correction only.

    Returns
    -------
    correction : dict
        Additive correction counts keyed by count name.
    """
    from cucount.jax import BinAttrs, SelectionAttrs, WeightAttrs, get_sharding_mesh
    from cucount.types import count2

    sharding_mesh = get_sharding_mesh()

    input_all_particles = []
    for particles in all_particles:
        particles = dict(particles)
        if 'shifted' in particles:
            particles.pop('randoms')
        input_all_particles.append(particles)
    all_particles = input_all_particles

    kw_battrs = dict(battrs=battrs)

    sattrs = SelectionAttrs(theta=(0., 0.05))
    wattrs = WeightAttrs()

    if normalize_randoms:
        def normalize_randoms(data, randoms, wattrs=WeightAttrs()):
            data_weights, randoms_weights = wattrs(data), wattrs(randoms)
            return randoms.clone(weights=data_weights.sum() / randoms_weights.sum() * randoms_weights)

    def remove_phantom_particles(particles):
        if sharding_mesh.axis_names:
            particles = jax.jit(_identity_fn, out_shardings=jax.sharding.NamedSharding(sharding_mesh, spec=P(None)))(particles)
        weights = particles.get('individual_weight')[0]
        if weights is None:
            return particles
        return particles[weights != 0].clone(exchange=True)

    for particles in all_particles:
        #particles['data'] = remove_phantom_particles(particles['data'])
        if normalize_randoms:
            for name in ['shifted', 'randoms']:
                if name in particles:
                    particles[name] = normalize_randoms(particles['data'], particles[name], wattrs=wattrs)

    def count2split(*particles, wattrs=None, mattrs=None):
        kw = dict(battrs=battrs, wattrs=wattrs, sattrs=sattrs, mattrs=mattrs, spattrs=spattrs, norm=1.)
        nsplits = [len(p) if isinstance(p, list) else 0 for p in particles]
        if any(nsplits):
            for nsplit in nsplits:
                if nsplit: break
            particles = list(particles)
            for ip, particle in enumerate(particles):
                if isinstance(particle, list):
                    assert len(particle) == nsplit
                else:
                    particles[ip] = [particle] * nsplit
            counts = [count2(*p, **kw)['weight'] for p in zip(*particles)]
            return types.sum(counts)
        return count2(*particles, **kw)['weight']

    correction = {}
    if cut is not None:
        wattrs = WeightAttrs()
        with_randoms = any(name in all_particles[0] for name in ['randoms', 'shifted'])
        correction = {}
        for combinations in itertools.product(['D'] + (['S'] if with_randoms else []), repeat=2):
            particles, combinations = _get_particle_combinations(combinations, all_particles, with_repeats=False)
            counts = count2split(*particles, wattrs=wattrs)
            correction[combinations] = counts.clone(value=-counts.value())
        return correction

    bitwise = angular = None
    bitweights = all_particles[0]['data'].get('bitwise_weight')
    if bitweights:
        bitwise = dict(weights=bitweights)

    angular = None
    if auw is not None:
        combinations = 'DD'
        angular = dict(edges=[np.append(edge[:, 0], edge[-1, 1]) for edge in auw.get(combinations).edges().values()],
                       weight=auw.get(combinations).value())
        if jax.process_index() == 0:
            logger.info('Applying AUW.')

    if bitwise is not None or auw is not None:
        wattrs = WeightAttrs(bitwise=bitwise)
        # Set default negative_weight = individual_weight
        for i, particles in enumerate(all_particles):
            particles = particles['data']
            if not particles.get('negative_weight'):
                negative_weight = wattrs(particles)
                all_particles[i]['data'] = particles.clone(weights=particles.get('weights') + [negative_weight], index_value=particles.index_value.clone(negative_weight=1))
    
    wattrs = WeightAttrs(bitwise=bitwise, angular=angular)
    particles, combinations = _get_particle_combinations('DD', all_particles, with_repeats=False)
    correction['DD'] = count2split(*particles, wattrs=wattrs)
    return correction
