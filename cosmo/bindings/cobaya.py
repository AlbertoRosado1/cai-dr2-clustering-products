"""Cobaya bindings for DESI cosmology likelihoods."""

from .bao import get_bao_data_path
from .registry import LIKELIHOOD_REGISTRY, get_likelihood_metadata


DEFAULT_BAO_LIKELIHOOD_PACKAGE = 'cosmo.cobaya_likelihoods.bao.desi_dr2'


def _resolve_config(config, python_path=None):
    """Return a shallow copy of a Cobaya likelihood config."""
    if config is None:
        return None
    return dict(config)

def _get_bao_cobaya_likelihood(metadata, likelihood_package=None, likelihood_path=None, python_path=None):
    """Return one Cobaya likelihood entry for a BAO data product."""
    if likelihood_package is None:
        likelihood_package = DEFAULT_BAO_LIKELIHOOD_PACKAGE
    likelihood_path = get_bao_data_path(likelihood_path)
    config = {
        'path': str(likelihood_path),
        'measurements_file': metadata['measurements_file'],
        'cov_file': metadata['cov_file'],
    }
    if python_path is not None:
        config['python_path'] = str(python_path)
    return f'{likelihood_package}.{metadata["likelihood"]}', config


def _get_external_cobaya_likelihoods(metadata, python_path=None):
    cobaya = metadata['cobaya']
    if isinstance(cobaya, str):
        return {cobaya: None}
    if isinstance(cobaya, dict):
        return {name: _resolve_config(config, python_path=python_path) for name, config in cobaya.items()}
    raise TypeError(f'Unsupported Cobaya metadata for {metadata.get("name")!r}: {type(cobaya).__name__}')


def get_cobaya_likelihoods(likelihoods=None, dataset=None, likelihood_package=None,
                           likelihood_path=None, python_path=None, registry=LIKELIHOOD_REGISTRY):
    """Return Cobaya likelihood entries for one or more likelihoods.

    Parameters
    ----------
    likelihoods : str, list, optional
        Likelihood names, e.g. ``'desi-bao-all'`` or a list such as
        ``['desi-bao-all', 'pantheonplus']``.
    dataset : str, list, optional
        Convenience alias for BAO-only calls.
    likelihood_package : str, optional
        Override the package that contains native Cobaya BAO wrapper classes.
    likelihood_path : str, Path, optional
        Directory containing BAO mean/covariance files.
    python_path : str, Path, optional
        Optional path added by Cobaya before importing likelihood classes.
    """
    output = {}
    for metadata in get_likelihood_metadata(likelihoods=likelihoods, dataset=dataset, registry=registry):
        family = metadata['family']
        if family == 'bao':
            name, config = _get_bao_cobaya_likelihood(metadata, likelihood_package=likelihood_package,
                                                      likelihood_path=likelihood_path,
                                                      python_path=python_path)
            output[name] = config
        elif 'cobaya' in metadata:
            output.update(_get_external_cobaya_likelihoods(metadata, python_path=python_path))
        else:
            raise ValueError(f'No Cobaya binding is registered for likelihood family {family!r}.')
    return output


def get_cobaya_priors(likelihoods=None, dataset=None, registry=LIKELIHOOD_REGISTRY):
    """Return info-level Cobaya priors registered for likelihood combinations."""
    output = {}
    for metadata in get_likelihood_metadata(likelihoods=likelihoods, dataset=dataset, registry=registry):
        output.update(metadata.get('prior', {}) or {})
    return output
