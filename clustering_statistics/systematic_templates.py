
import numpy as np

import lsstypes as types
from .tools import get_stats_fn, base_stats_dir


def include_systematic_templates(window: types.WindowMatrix, templates: dict, effects: tuple | list=('auw',)):
    """Include systematic templates in the window matrix."""
    observable = window.observable
    windows, theories, _types = [], [], []
    for effect in effects:
        if effect == 'auw':
            diff = templates['auw'].match(observable).value() - templates['raw'].match(observable).value()
            # - sign, as applied on the theory side
            windows.append(- diff[:, None])
            theories.append(types.ObservableLeaf(value=np.array([1.]), scale=np.array([0.3])))
            _types.append(effect)
        elif effect == 'photo':
            diff_amr = templates['mock_amr'].clone(value=templates['mock_amr'].value() - templates['mock_noamr'].value())
            diff_amr = smooth_template(diff_amr, effect='amr')
            windows.append(diff_amr[:, None])
            theories.append(types.ObservableLeaf(value=np.array([1.]), scale=np.array([0.1])))
            _types.append('amr')
            if 'sys' in templates:
                diff_wsys = templates['sys'].match(observable).value()
                windows.append(diff_wsys[:, None])
            theories.append(types.ObservableLeaf(value=np.array([0.]), scale=np.array([0.3])))
            _types.append('sys')
        elif effect == 'ric':
            diff_ric = templates['mock_ric'].clone(value=templates['mock_ric'].value() - templates['mock_noric'].value())
            diff_ric = smooth_template(diff_ric, effect='ric')
            windows.append(diff_ric[:, None])
            theories.append(types.ObservableLeaf(value=np.array([1.]), scale=np.array([0.1])))
            _types.append('ric')
    theory = window.theory
    theory = types.ObservableTree([theory] + theories, types=['theory'] + _types)
    window = window.clone(value=np.concatenate([window.value()] + windows, axis=1), theory=theory)
    return window


def polyfit(x, y, order, cov=None):
    x, y, order = map(np.asarray, (x, y, sorted(order)))

    if cov is None:
        W = np.eye(len(x))
    else:
        W = np.linalg.inv(cov)
    X = x[:, None] ** order[None, :]
    A = X.T @ W @ X
    b = X.T @ W @ y
    coeff = np.linalg.solve(A, b)

    class PolyfitResult:
        def __init__(self, coeff, order):
            self.coeff = coeff
            self.order = order

        def __call__(self, x):
            x = np.asarray(x)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                basis = x[None, :] ** self.order[:, None]
                y = self.coeff @ basis
            y = np.where(np.isfinite(y), y, 0.0)
            return y

    return PolyfitResult(coeff, order)


def smooth_template(stat, effect='amr', order=None, klim=None, wkp=None):
    if effect not in ('amr', 'ric'):
        raise ValueError(effect)
    # same defaults for 'amr' and 'ric' for now
    isbk = stat.get(ells=stat.ells[0]).coords('k').ndim > 1
    if isbk:
        # use slightly less number of orders as the bispectrum template is noisier at large scales, use wkp=4 since cov(B) ~ 1/k^4 at small scales
        default = dict(order=[-5, -3, -2], klim=[0.01, 0.20, 2], wkp=4)
    else:
        # use more orders as the power spectrum template is very smooth, use wkp=2 since cov(P) ~ 1/k^2 at small scales
        default = dict(order=[-6, -5, -4, -3, -2, -1], klim=[0.01, 0.40, 5], wkp=2)
    if order is None: order = default['order']
    if klim is None: klim = default['klim']
    if wkp is None: wkp = default['wkp']

    kmin, kmax, nbin = klim
    tofit = stat.select(k=(kmin, kmax)).select(k=slice(0, None, nbin))
    values = []
    for ell in tofit.ells:
        pole = tofit.get(ells=ell)
        k = pole.coords('k')
        kout = stat.get(ells=ell).coords('k')
        if isbk:
            k, kout = k[..., 0], kout[..., 0]
        fit = polyfit(k, pole.value(), order=order, cov=np.diag(1/k**wkp))
        values.append(fit(kout))
    return stat.clone(value=np.hstack(values))


def get_template_mock_fns(tracer=None, zrange=None, region=None, kind='mesh2_spectrum', key='mock_amr', **kwargs):
    mock_stats_dir = base_stats_dir
    if kind not in ['mesh2_spectrum', 'mesh3_spectrum']:
        raise ValueError(kind)
    weight, extra = 'default-FKP', None
    if key in ['mock_amr', 'mock_noamr']:
        if kind == 'mesh2_spectrum':  # GLAM mocks
            version = 'glam-uchuu-bgs-altmtl' if 'BGS' in tracer else 'glam-uchuu-v2-altmtl'
            project = 'full_shape/imaging_systematics' if 'BGS' in tracer else 'full_shape/base'
            imocks = [imock for imock in range(150, 250) if imock not in [210, 243]]
        else:  # Abacus mocks
            version = 'abacus-hf-dr2-v2-altmtl'
            project = 'full_shape/imaging_systematics'
            imocks = list(range(25))
        weight = {'mock_amr': 'default-FKP', 'mock_noamr': 'default-FKP-noimsys'}[key]
    elif key in ['mock_ric', 'mock_noric']:  # Abacus mocks
        version = 'holi-v3-altmtl'
        if 'BGS' in tracer:
            version = 'holi-bgs-altmtl'
        project = 'full_shape/base'
        imocks = list(range(25))
        extra = {'mock_ric': None, 'mock_noric': 'reshuffle'}[key]
    else:
        raise ValueError(key)
    options = dict(project=project, version=version, tracer=tracer, zrange=zrange, region=region, weight=weight, auw=False,
                   kind=kind, basis='sugiyama-diagonal', extra=extra)
    return [get_stats_fn(mock_stats_dir, imock=imock, **options) for imock in imocks]


def get_smooth_template(tracer, zrange, region, effect='amr', kind='mesh2_spectrum'):
    weight = types.mean([types.read(fn) for fn in get_template_mock_fns(tracer, zrange, region, kind=kind, key=f'mock_{effect}')])
    noweight = types.mean([types.read(fn) for fn in get_template_mock_fns(tracer, zrange, region, kind=kind, key=f'mock_no{effect}')])
    diff = weight.clone(value=weight.value() - noweight.value())
    return smooth_template(diff, effect=effect)
