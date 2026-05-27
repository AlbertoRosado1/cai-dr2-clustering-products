"""Named likelihood-combination presets for DESI cosmology configs."""

from collections.abc import Iterable


LIKELIHOOD_COMBINATIONS = {
    'bao': ['desi-bao-all'],
    'bao-sn-pantheonplus': ['desi-bao-all', 'pantheonplus'],
    'bao-sn-union3': ['desi-bao-all', 'union3'],
    'bao-sn-desy5': ['desi-bao-all', 'desy5sn'],
    'bao-sn-desdovekie': ['desi-bao-all', 'desdovekie'],
    'bao-sn-pantheonplus-zmin0.1': ['desi-bao-all', 'pantheonplus-zmin0.1'],
    'bao-sn-union3-zmin0.1': ['desi-bao-all', 'union3-zmin0.1'],
    'bao-sn-desy5-zmin0.1': ['desi-bao-all', 'desy5sn-zmin0.1'],
    'bao-bbn': ['desi-bao-all', 'schoneberg2024-bbn'],
    'bao-bbn-fixed-nnu': ['desi-bao-all', 'schoneberg2024-bbn-fixed-nnu'],
    'bao-thetastar-fixed-nnu': ['desi-bao-all', 'planck2018-thetastar-fixed-nnu'],
    'bao-thetastar-varied-nnu': ['desi-bao-all', 'planck2018-thetastar-varied-nnu'],
    'bao-rdrag-fixed-nnu': ['desi-bao-all', 'planck2018-rdrag-fixed-nnu'],
    'bao-cmb-compressed-theta': ['desi-bao-all', 'CMB-compressed-theta'],
    'bao-cmb-compressed-r-la': ['desi-bao-all', 'CMB-compressed-R-lA'],
    'bao-cmb-compressed-theta-ombh2': ['desi-bao-all', 'CMB-compressed-theta-ombh2'],
    'bao-cmb-compressed-theta-ombh2-ombch2': ['desi-bao-all', 'CMB-compressed-theta-ombh2-ombch2'],
    'bao-sn-cmb-compressed-theta': ['desi-bao-all', 'pantheonplus', 'CMB-compressed-theta'],
    'bao-sn-cmb-compressed-r-la': ['desi-bao-all', 'pantheonplus', 'CMB-compressed-R-lA'],
    'cmb-spa': ['CMB-SPA'],
    'cmb-spa-tauprior': ['CMB-SPA-tauprior'],
    'bao-cmb-spa': ['desi-bao-all', 'CMB-SPA'],
    'bao-sn-desdovekie-cmb-spa': ['desi-bao-all', 'desdovekie', 'CMB-SPA'],
    'bao-planck-npipe': ['desi-bao-all', 'planck-NPIPE-highl-CamSpec-TTTEEE'],
    'bao-planck-npipe-lensing': ['desi-bao-all', 'planck-NPIPE-highl-CamSpec-TTTEEE', 'planckpr4lensing'],
    'bao-planck-npipe-ell-max-600': ['desi-bao-all', 'planck-NPIPE-highl-CamSpec-TTTEEE-ell-max-600'],
    'bao-planck-npipe-cuts-for-act': ['desi-bao-all', 'planck-NPIPE-highl-CamSpec-TTTEEE-cuts-for-act'],
    'bao-planck-npipe-sroll2-momento': ['desi-bao-all', 'planck2018-lowl-TTTEEE-sroll2-momento'],
}


def is_likelihood_combination(name):
    """Return whether *name* is a named likelihood-combination preset."""
    return isinstance(name, str) and name in LIKELIHOOD_COMBINATIONS


def get_likelihood_combination(name):
    """Return the likelihood list for a named preset."""
    try:
        return list(LIKELIHOOD_COMBINATIONS[name])
    except KeyError as exc:
        raise KeyError(f'Unknown likelihood combination {name!r}.') from exc


def _as_sequence(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def normalize_likelihood_combination(value):
    """Expand one preset/list/comma-separated value to likelihood names.

    Non-preset likelihood names are preserved. Presets can also appear inside a
    comma-separated value or explicit list, e.g. ``'bao,pantheonplus'``.
    """
    if is_likelihood_combination(value):
        return get_likelihood_combination(value)
    output = []
    for item in _as_sequence(value):
        if is_likelihood_combination(item):
            output.extend(get_likelihood_combination(item))
        else:
            output.append(item)
    # Preserve order but avoid duplicate entries introduced by combinations like
    # 'bao,pantheonplus'.
    deduped = []
    for item in output:
        if item not in deduped:
            deduped.append(item)
    if len(deduped) == 1:
        return deduped[0]
    return deduped


def normalize_likelihood_combinations(values):
    """Normalize multiple CLI/config likelihood-combination values."""
    values = values or ['bao']
    return [normalize_likelihood_combination(value) for value in values]
