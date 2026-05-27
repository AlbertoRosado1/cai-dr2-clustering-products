"""Registry for cosmology likelihood data products."""

from .bao import BAO_DR2_DATASETS, make_list
from .bbn import BBN_LIKELIHOODS
from .cmb import CMB_LIKELIHOODS
from .sn import SN_LIKELIHOODS


BAO_LIKELIHOODS = {
    name: {
        'name': name,
        'family': 'bao',
        'parameterization': 'background',
        **metadata,
    }
    for name, metadata in BAO_DR2_DATASETS.items()
}

LIKELIHOOD_REGISTRY = {
    **BAO_LIKELIHOODS,
    **SN_LIKELIHOODS,
    **BBN_LIKELIHOODS,
    **CMB_LIKELIHOODS,
}


def normalize_likelihoods(likelihoods=None, dataset=None, default='desi-bao-all', registry=LIKELIHOOD_REGISTRY):
    """Normalize one or more likelihood names to a list of strings."""
    if likelihoods is None:
        likelihoods = dataset
    elif dataset is not None:
        raise ValueError('Pass either likelihoods or dataset, not both.')
    if likelihoods is None:
        likelihoods = default
    names = make_list(likelihoods)
    unknown = [name for name in names if name not in registry]
    if unknown:
        raise ValueError('Unknown likelihood(s): {}. Known likelihoods are {}.'.format(
            ', '.join(unknown), ', '.join(sorted(registry))))
    return names


def get_likelihood_metadata(likelihoods=None, dataset=None, registry=LIKELIHOOD_REGISTRY):
    """Return metadata dictionaries for one or more likelihoods."""
    return [registry[name] for name in normalize_likelihoods(likelihoods=likelihoods, dataset=dataset,
                                                            registry=registry)]


def get_likelihood_families(likelihoods=None, dataset=None, registry=LIKELIHOOD_REGISTRY):
    """Return the likelihood families needed by a likelihood combination."""
    return sorted({metadata['family'] for metadata in get_likelihood_metadata(likelihoods=likelihoods,
                                                                             dataset=dataset,
                                                                             registry=registry)})


def get_parameterization(likelihoods=None, dataset=None, registry=LIKELIHOOD_REGISTRY):
    """Return the cosmological parameterization required by likelihoods."""
    parameterizations = {metadata.get('parameterization', 'background')
                         for metadata in get_likelihood_metadata(likelihoods=likelihoods,
                                                                 dataset=dataset,
                                                                 registry=registry)}
    if parameterizations == {'background'}:
        return 'background'
    if len(parameterizations) == 1:
        return parameterizations.pop()
    return 'general'
