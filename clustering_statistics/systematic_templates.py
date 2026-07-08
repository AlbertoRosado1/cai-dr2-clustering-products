
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
            diff_amr = templates['amr'].match(observable).value()
            diff_wsys = templates['sys'].match(observable).value()
            windows.append(diff_amr[:, None])
            theories.append(types.ObservableLeaf(value=np.array([1.]), scale=np.array([0.01])))
            _types.append('amr')
            windows.append(diff_wsys[:, None])
            theories.append(types.ObservableLeaf(value=np.array([0.]), scale=np.array([0.3])))
            _types.append('sys')
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
    return PolyfitResult(coeff, order)


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


def smooth_pk(pk, order, klim, wkp=2):
    kmin, kmax, nbin = klim
    tofit = pk.select(k=(kmin, kmax)).select(k=slice(0, None, nbin))
    values = []
    for ell in tofit.ells:
        pole = tofit.get(ells=ell)
        fit = polyfit(k:=pole.coords('k'), pole.value(), order=order, cov=np.diag(1/k**wkp))
        values.append(fit(pk.get(ells=ell).coords('k')))
    return pk.clone(value=np.hstack(values))


def smooth_bk(bk, order, klim, wkp=4):
    kmin, kmax, nbin = klim
    tofit = bk.select(k=(kmin, kmax)).select(k=slice(0, None, nbin))
    values = []
    for ell in tofit.ells:
        pole = tofit.get(ells=ell)
        fit = polyfit(k:=pole.coords('k')[..., 0], pole.value(), order=order, cov=np.diag(1/k**wkp))
        values.append(fit(bk.get(ells=ell).coords('k')[..., 0]))
    return bk.clone(value=np.hstack(values))


def load_null_glam_weight_noweight(tracer, zrange, region, auw=False, kind='mesh2_spectrum'):
    mock_stats_dir = base_stats_dir

    version = 'glam-uchuu-bgs-altmtl' if 'BGS' in tracer else 'glam-uchuu-v2-altmtl'
    project = 'full_shape/imaging_systematics' if 'BGS' in tracer else 'full_shape/base'
    extra = dict(kind=kind)
    if kind == 'mesh3_spectrum':
        extra |= dict(basis='sugiyama-diagonal')

    mock_list = [imock for imock in range(150, 250) if imock not in [210, 243]]
    options = dict(project=project, version=version, tracer=tracer, zrange=zrange, region=region, auw=auw) | extra
    weight_fns = [get_stats_fn(mock_stats_dir, weight='default-FKP', imock=imock, **options) for imock in mock_list]
    noweight_fns = [get_stats_fn(mock_stats_dir, weight='default-noimsys-FKP', imock=imock, **options) for imock in mock_list]
    weight = types.mean([types.read(fn) for fn in weight_fns])
    noweight = types.mean([types.read(fn) for fn in noweight_fns])
    return weight, noweight


def load_null_abacus_weight_noweight(tracer, zrange, region, auw=False, kind='mesh2_spectrum'):
    mock_stats_dir = base_stats_dir

    version = 'abacus-hf-dr2-v2-altmtl'
    project = 'full_shape/imaging_systematics'
    extra = dict(kind=kind)
    if kind == 'mesh3_spectrum':
        extra |= dict(basis='sugiyama-diagonal')

    mock_list = [imock for imock in range(25)]
    options = dict(project=project, version=version, tracer=tracer, zrange=zrange, region=region, auw=auw) | extra
    weight_fns = [get_stats_fn(mock_stats_dir, weight='default-FKP', imock=imock, **options) for imock in mock_list]
    noweight_fns = [get_stats_fn(mock_stats_dir, weight='default-FKP-noimsys', imock=imock, **options) for imock in mock_list]
    weight = types.mean([types.read(fn) for fn in weight_fns])
    noweight = types.mean([types.read(fn) for fn in noweight_fns])
    return weight, noweight


def get_amr_template(tracer, zrange, region, kind='mesh2_spectrum'):
    if kind == 'mesh2_spectrum':
        glam_weight, glam_noweight = load_null_glam_weight_noweight(tracer, zrange, region, auw=False, kind=kind)
        diff_glam = glam_weight.clone(value=glam_weight.value() - glam_noweight.value())
        # use more orders as diff_glam is very smooth, use wkp=2 since cov(P) ~ 1/k^2 at small scales
        template = smooth_pk(diff_glam, order=[-6, -5, -4, -3, -2, -1], klim=[0.01, 0.40, 5], wkp=2)
        return template
    if kind == 'mesh3_spectrum':
        abacus_weight, abacus_noweight = load_null_abacus_weight_noweight(tracer, zrange, region, auw=False, kind=kind)
        diff_abacus = abacus_weight.clone(value=abacus_weight.value() - abacus_noweight.value())
        # use slightly less number of orders as diff_abacus is noisier at large scales, use wkp=4 since cov(B) ~ 1/k^4 at small scales
        template = smooth_bk(diff_abacus, order=[-5, -3, -2], klim=[0.01, 0.20, 2], wkp=4)
        return template
    raise ValueError(kind)
