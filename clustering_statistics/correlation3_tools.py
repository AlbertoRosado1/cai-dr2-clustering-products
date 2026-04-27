"""
Configuration-space 3-point clustering measurements.

Main functions
--------------
* `compute_particle3_angular_upweights`: Derive angular upweights for fiber-collision mitigation.
"""

import logging
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp

import lsstypes as types
from .tools import _format_bitweights


logger = logging.getLogger('correlation3')


def compute_particle3_angular_upweights(*get_data):
    """
    Compute angular upweights (AUW) from fibered and parent data catalogs.

    Parameters
    ----------
    get_data : callables
        Functions that return dict of 'fibered_data', 'parent_data' catalogs. Each catalog must contain 'RA', 'DEC', 'INDWEIGHT', and optionally 'BITWEIGHT'.

    Returns
    -------
    auw : ObservableTree
        Angular upweights as an ObservableTree with 'DDD' leaf.
    """
    # Import cucount submodules for pair counting and particle handling
    from cucount.jax import Particles, BinAttrs, SelectionAttrs, WeightAttrs, setup_logging
    from cucount.jax import create_sharding_mesh
    from cucount.types import count3close
    from lsstypes import ObservableLeaf, ObservableTree

    # Use distributed mesh for computation across devices
    with create_sharding_mesh() as sharding_mesh:
        all_fibered_data, all_parent_data = [], []

        # Helper function to extract (RA, DEC) positions and weights from catalog
        def get_rdw(catalog):
            positions = (catalog['RA'], catalog['DEC'])
            # Combine individual weights with optional bitwise weights
            weights = [catalog['INDWEIGHT']] + _format_bitweights(catalog['BITWEIGHT'] if 'BITWEIGHT' in catalog else None)
            return positions, weights

        # Process each data source (for cross-correlations)
        for _get_data in get_data:
            _data = _get_data()
            # Create Particles objects for fibered and parent catalogs in celestial (RA, DEC) coordinates
            fibered_data = Particles(*get_rdw(_data['fibered_data']), positions_type='rd', exchange=True)
            parent_data = Particles(*get_rdw(_data['parent_data']), positions_type='rd', exchange=True)
            all_fibered_data.append(fibered_data)
            all_parent_data.append(parent_data)

        # Define angular separation bins (in degrees)
        theta = 10**np.arange(-4, np.log10(180.), 0.2)
        battrs = BinAttrs(theta=theta)
        sattrs = SelectionAttrs(theta=(0., 0.05))

        # Set up bitwise weight attributes (PIP weights)
        bitwise = None
        if all_fibered_data[0].get('bitwise_weight'):
            raise NotImplementedError
            bitwise = dict(weights=all_fibered_data[0].get('bitwise_weight'))
            if jax.process_index() == 0:
                logger.info(f'Applying bitwise weights {bitwise}.')

        # Compute triplet counts for fibered data with bitwise weights
        wattrs = WeightAttrs(bitwise=bitwise)
        kw = dict(battrs12=battrs, battrs13=battrs, battrs23=battrs)

        def count3close_resol(*all_particles, wattrs=None):
            theta_limits = [(0., 0.3), (0.3, 1.), (1., 5.), (5., 180.)]
            nsides = [None, 512, 128, 32]  # 55, 75, 50
            #nsides = [None, 1024, 256, 64]
            #theta_limits = [(0., 0.5), (0.5, 1.), (1., 5.), (5., 180.)]
            #nsides = [None, 512, 256, 64]  # 55, 75, 50 
            #theta_limits, nsides = theta_limits[sl], nsides[sl]
            #theta_limits = [(0., 0.01), (0.1, 5.), (5., 180.)]
            #nsides = [None, 8, 8]
            import healpy as hp

            _identity_fn = lambda x: x

            def digitize(nside, particles):
                from jax.sharding import PartitionSpec as P
                if sharding_mesh.axis_names:
                    particles = jax.jit(_identity_fn, out_shardings=jax.sharding.NamedSharding(sharding_mesh, spec=P(None)))(particles)
                pix = hp.vec2pix(nside, *particles.get('positions').T, nest=False)
                weights = wattrs(particles)
                npix = hp.nside2npix(nside)
                pix_weights = np.bincount(pix, weights=weights, minlength=npix)
                mask = pix_weights > 0
                pix_weights = pix_weights[mask]
                pix_positions = np.column_stack(hp.pix2vec(nside, np.flatnonzero(mask), nest=False))
                return Particles(pix_positions, pix_weights, exchange=False)

            all_particles = list(all_particles) + [all_particles[-1]] * (3 - len(all_particles))
            results = []
            for theta_limit, nside in zip(theta_limits, nsides):
                sattrs_limit = SelectionAttrs(theta=theta_limit)
                # 1) close-pair (1, 2)
                all_particles_resol = list(all_particles)
                if nside is not None:
                    all_particles_resol[2] = digitize(nside, all_particles_resol[2])
                result12 = count3close(*all_particles_resol, close_pair=(1, 2), sattrs12=sattrs, sattrs13=sattrs_limit, **kw)['weight']
                # 2) close-pair (1, 3), excluding (1, 2) close pairs
                all_particles_resol = list(all_particles)
                if nside is not None:
                    all_particles_resol[1] = digitize(nside, all_particles_resol[1])
                result13 = count3close(*all_particles_resol, close_pair=(1, 3), sattrs12=sattrs_limit, sattrs13=sattrs, veto12=sattrs, wattrs=wattrs, **kw)['weight']
                # 3) close-pair (2, 3), excluding (1, 2), (1, 3) close pairs
                all_particles_resol = list(all_particles)
                if nside is not None:
                    all_particles_resol[0] = digitize(nside, all_particles_resol[0])
                result23 = count3close(*all_particles_resol, close_pair=(2, 3), sattrs12=sattrs_limit, sattrs23=sattrs, veto12=sattrs, veto13=sattrs, wattrs=wattrs, shard_particle=2, **kw)['weight']
                results += [result12, result13, result23]
            result = results[0].clone(counts=sum(result.values('counts') for result in results), norm=results[0].values('norm'))
            return result

        DDDfibered = count3close_resol(*all_fibered_data, wattrs=wattrs)
        # Compute triplet counts for parent (unfiber-limited) data without bitwise weights
        DDDparent = count3close_resol(*all_parent_data, wattrs=wattrs)

    # Prepare output arrays with angular separation bins
    coords = ['theta1', 'theta2', 'theta3']
    kw = dict(coords=coords)
    for coord in coords:
        kw[coord] = battrs.coords('theta')
        kw[f'{coord}_edges'] = battrs.edges('theta')
    auw = {}
    # Angular upweights = ratio of parent to fibered pair counts (1 where no pairs)
    auw['DDD'] = ObservableLeaf(value=np.where(DDDfibered.value() == 0., 1., DDDparent.value() / DDDfibered.value()), **kw)
    auw['DDDparent'] = DDDparent
    auw['DDDfibered'] = DDDfibered
    # Wrap in ObservableTree for consistent data structure
    auw = ObservableTree(list(auw.values()), triplets=list(auw.keys()))
    return auw