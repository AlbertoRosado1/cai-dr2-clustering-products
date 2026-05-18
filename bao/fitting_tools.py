"""
Utilities for BAO fitting.

This module is a BAO-focused extraction of the reusable pieces from
``full_shape/fitting_tools.py``. It keeps the existing binning and theory
choices, but replaces campaign-specific hard-coded paths with configurable
helpers and repairs the incomplete BAO likelihood path.
"""

import logging
import os
from typing import Any, Dict, List, Tuple

from clustering_statistics.tools import get_stats_fn
from full_shape.fitting_tools import get_template as _full_shape_get_template
from full_shape.fitting_tools import get_theory as _full_shape_get_theory
from full_shape.fitting_tools import load_bins

logger = logging.getLogger("bao.fitting_tools")

VALID_MEASUREMENT_KINDS = {
    "particle2_correlation",
    "recon_particle2_correlation",
    "mesh2_spectrum",
    "recon_mesh2_spectrum",
}


def _require(module_name):
    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError as exc:
        raise ImportError(
            f"{module_name} is required for BAO fitting but is not available in the current environment."
        ) from exc


def get_measurement_fn(
    kind: str = "particle2_correlation",
    version: str = "dr2-v2",
    tracer: str = "LRG",
    region: str = "NGC",
    zrange=(0.8, 1.1),
    recon=None,
    cut=None,
    auw=None,
    nran=18,
    weight_type="default",
    base_dir=None,
    project="bao/base",
    imock=None,
    stats_dir=None,
    ext="h5",
    extra="",
    battrs=None,
    **_,
):
    """
    Build a measurement filename using ``clustering_statistics.tools.get_stats_fn``.

    This keeps BAO paths aligned with the repository-wide statistics layout:
    ``stats_dir / project / version / [mock{imock}/] <basename>.h5``.
    """
    canonical_kind = kind
    if canonical_kind.startswith("covariance_"):
        canonical_kind = canonical_kind[len("covariance_") :]
    if canonical_kind not in VALID_MEASUREMENT_KINDS:
        raise ValueError(
            "Unsupported measurement kind {!r}. Expected one of {}.".format(
                kind, sorted(VALID_MEASUREMENT_KINDS)
            )
        )
    if stats_dir is None:
        stats_dir = base_dir or os.getenv("BAO_DATA_DIR") or (os.getenv("SCRATCH", ".") + "/measurements")
    if recon and not kind.startswith("recon_"):
        raise ValueError("Use a recon_* measurement kind instead of passing recon separately.")

    catalog = {
        "version": version,
        "tracer": tracer,
        "region": region,
        "zrange": zrange,
        "weight": weight_type,
        "imock": imock,
    }
    kwargs = dict(
        stats_dir=stats_dir,
        project=project,
        kind=kind,
        auw=auw,
        cut=cut,
        extra=extra,
        ext=ext,
        catalog=catalog,
    )
    if battrs is not None:
        kwargs["battrs"] = battrs
    return str(get_stats_fn(**kwargs))


def get_effective_redshift(
    data_args,
    get_measurement=get_measurement_fn,
    zeff_kind=None,
):
    """
    Estimate ``z_eff`` from a measurement carrying the redshift metadata.

    This reuses the convention from ``full_shape/fitting_tools.py`` but makes
    the file locator injectable.
    """
    types = _require("lsstypes")
    kind = zeff_kind or data_args.get("zeff_kind") or data_args.get("kind") or "mesh2_spectrum"
    if kind not in VALID_MEASUREMENT_KINDS:
        raise ValueError(
            "Unsupported zeff measurement kind {!r}. Expected one of {}.".format(
                kind, sorted(VALID_MEASUREMENT_KINDS)
            )
        )
    measurement = types.read(get_measurement(**data_args, kind=kind))
    mono = measurement.theory.get(ells=0)
    zeff = getattr(mono, "z", mono._meta.get("z", None))
    if zeff is None:
        raise AttributeError("No z_eff found in the requested measurement.")
    return zeff


def load_bao_data(
    data_args,
    data_kind="recon_particle2_correlation",
    covariance_kind=None,
    get_measurement=get_measurement_fn,
):
    """
    Load BAO correlation-function data and covariance.

    The data source is intentionally configurable because the legacy full-shape
    loader was wired to a specific blinded-data campaign and to power-spectrum
    products.
    """
    types = _require("lsstypes")
    if data_kind not in VALID_MEASUREMENT_KINDS:
        raise ValueError(
            "Unsupported data measurement kind {!r}. Expected one of {}.".format(
                data_kind, sorted(VALID_MEASUREMENT_KINDS)
            )
        )
    if covariance_kind is None:
        covariance_kind = "covariance_{}".format(data_kind)
    data = types.read(get_measurement(**data_args, kind=data_kind))
    covariance = types.read(get_measurement(**data_args, kind=covariance_kind))
    return data, covariance


def get_template(
    task,
    z_eff=1.0,
    ells=(0, 2),
    cosmo=None,
    **_,
):
    """Return the BAO template used by the fit."""
    if "BAO" not in task:
        raise ValueError(f"Unsupported BAO task {task!r}.")
    return _full_shape_get_template(task, z_eff=z_eff, ells=ells, cosmo=cosmo, **_)


def get_theory(
    task,
    template=None,
    ells=(0, 2),
    smoothing_radius=15,
):
    """Return the damped-wiggles BAO correlation-function theory."""
    if "BAO" not in task:
        raise ValueError(f"Unsupported BAO task {task!r}.")
    if template is None:
        raise ValueError("template is required for BAO theory construction.")
    return _full_shape_get_theory(task, template=template, ells=ells, smoothing_radius=smoothing_radius)


def _extract_coords(data, coord_name):
    """Best-effort extraction of per-multipole coordinates from a measurement object."""
    try:
        return [pole.coords(coord_name) for pole in data]
    except TypeError:
        return [pole.coords(coord_name) for pole in list(data)]


def _build_xi_observable(data, covariance, theory, slim):
    galaxy = _require("desilike.observables.galaxy_clustering")
    observable_cls = getattr(galaxy, "TracerCorrelationFunctionMultipolesObservable")

    kwargs = {
        "data": data.value(concatenate=True),
        "covariance": covariance.value(),
        "ells": data.ells,
        "s": _extract_coords(data, "s"),
        "theory": theory,
        "slim": slim,
    }
    try:
        return observable_cls(**kwargs)
    except TypeError:
        # Older/newer desilike versions sometimes accept the measurement object directly.
        kwargs["data"] = data
        return observable_cls(**kwargs)


def get_observable_likelihood(
    task: str,
    data_args,
    fit_args,
    get_measurement=get_measurement_fn,
    data_loader=load_bao_data,
):
    """
    Build a BAO observable and Gaussian likelihood.

    Parameters mirror the old ``full_shape.fitting_tools.get_observable_likelihood``
    entry point, but this version supports BAO only.
    """
    if "BAO" not in task:
        raise ValueError(f"Unsupported fit type {task!r}; this module is BAO-only.")

    corr_type = fit_args["corr_type"]
    if corr_type != "xi":
        raise ValueError(f"BAO fitting requires corr_type='xi', got {corr_type!r}.")

    bins_type = fit_args.get("bins_type", "y3_bao")
    option = fit_args.get("option", "")
    tracer = data_args["tracer"]
    measurement_kind = fit_args.get("measurement_kind", "recon_particle2_correlation")
    if measurement_kind not in VALID_MEASUREMENT_KINDS:
        raise ValueError(
            "Unsupported measurement_kind {!r}. Expected one of {}.".format(
                measurement_kind, sorted(VALID_MEASUREMENT_KINDS)
            )
        )
    z_eff = fit_args.get("z_eff")
    if z_eff is None:
        z_eff = get_effective_redshift(
            data_args,
            get_measurement=get_measurement,
            zeff_kind=fit_args.get("zeff_kind", measurement_kind),
        )

    DESI = getattr(_require("cosmoprimo.fiducial"), "DESI")
    ObservablesGaussianLikelihood = getattr(
        _require("desilike.likelihoods"),
        "ObservablesGaussianLikelihood",
    )

    rmin, rmax, rbin, _ = load_bins(corr_type, bins_type)
    ells = (0,) if "1d" in option else (0, 2)
    slim = {ell: (rmin, rmax, rbin) for ell in ells}
    smoothing_radius = 30 if tracer == "QSO" else 15

    covariance_kind = fit_args.get("covariance_kind")
    data, covariance = data_loader(
        data_args,
        data_kind=measurement_kind,
        covariance_kind=covariance_kind,
        get_measurement=get_measurement,
    )
    template = get_template(task, z_eff=z_eff, ells=ells, cosmo=DESI())
    theory = get_theory(task, template=template, ells=ells, smoothing_radius=smoothing_radius)

    if 2 not in ells:
        for param in theory.init.params.select(basename="*l2_*"):
            param.update(fixed=True)
        for param in theory.init.params.select(basename="qap"):
            param.update(fixed=True)

    observable = _build_xi_observable(data, covariance, theory, slim)
    likelihood = ObservablesGaussianLikelihood(
        observables=observable,
        theory=theory,
        covariance=covariance.value(),
    )
    likelihood()
    return likelihood, observable, theory
