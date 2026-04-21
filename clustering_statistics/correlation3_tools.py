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
    with create_sharding_mesh():
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
        theta = 10**np.arange(-5, np.log10(180.) + 0.1, 0.1)
        battrs = BinAttrs(theta=theta)
        sattrs = SelectionAttrs(theta=(0., 0.05))

        # Set up bitwise weight attributes (PIP weights)
        bitwise = None
        if all_fibered_data[0].get('bitwise_weight'):
            bitwise = dict(weights=all_fibered_data[0].get('bitwise_weight'))
            if jax.process_index() == 0:
                logger.info(f'Applying PIP weights {bitwise}.')

        # Compute pair counts for fibered data with bitwise weights
        wattrs = WeightAttrs(bitwise=bitwise)
        kw = dict(battrs12=battrs, battrs13=battrs, battrs23=battrs, sattrs12=sattrs, sattrs13=sattrs, wattrs=wattrs)
        DDDfibered = count3close(*all_fibered_data, **kw)['weight'].value()

        # Compute pair counts for parent (unfiber-limited) data without bitwise weights
        wattrs = WeightAttrs()
        DDDparent = count3close(*all_parent_data, **kw)['weight'].value()

    # Prepare output arrays with angular separation bins
    coords = ['theta1', 'theta2', 'theta3']
    kw = dict(coords=coords)
    for coord in coords:
        kw[coord] = battrs.coords('theta')
        kw[f'{coord}_edges'] = battrs.edges('theta')
    auw = {}
    # Angular upweights = ratio of parent to fibered pair counts (1 where no pairs)
    auw['DDD'] = ObservableLeaf(value=np.where(DDDfibered == 0., 1., DDDparent / DDDfibered), **kw)

    # Wrap in ObservableTree for consistent data structure
    auw = ObservableTree(list(auw.values()), triplets=list(auw.keys()))
    return auw
