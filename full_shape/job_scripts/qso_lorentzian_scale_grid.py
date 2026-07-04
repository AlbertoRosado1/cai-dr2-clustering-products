"""Profile QSO Lorentzian box fits over a small grid of scale cuts.

This helper is intentionally separate from ``validation_box_mocks.py`` so the
standard validation entry point remains stable.  It supports two workflows:

* local profiling from the broad QSO cache already present on this machine;
* cluster-side ``build`` runs for configs whose data/covariance cache is absent.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from full_shape import setup_logging


DEFAULT_CACHE_DIR = Path(__file__).parent / "_cache"
DEFAULT_FITS_DIR = Path(__file__).parent / "fits_box_mocks" / "qso_lorentzian_scale_grid"
DEFAULT_RAW_DATA = (
    DEFAULT_CACHE_DIR
    / "prepared_stats"
    / "data_QSO-z1.475-S2+QSO-z1.475-S3-9b0dacf6.h5"
)
DEFAULT_RAW_COVARIANCE = (
    DEFAULT_CACHE_DIR
    / "prepared_stats"
    / "covariance_QSO-z1.475-S2+QSO-z1.475-S3+cov-mock-none-f34da1fa.h5"
)
DEFAULT_BROAD_SELECTS = {
    "mesh2_spectrum": [
        {"ells": 0, "k": [0.02, 0.20, 0.01]},
        {"ells": 2, "k": [0.02, 0.20, 0.01]},
    ],
    "mesh3_spectrum": [
        {"ells": (0, 0, 0), "k": [0.02, 0.20, 0.01]},
        {"ells": (2, 0, 2), "k": [0.02, 0.20, 0.01]},
    ],
}
DEFAULT_STATS_DIR = Path("/global/cfs/cdirs/desicollab/science/cai/desi-clustering/dr2/summary_statistics/box")
DEFAULT_COVARIANCE_STATS_DIR = Path(
    "/dvs_ro/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/mock_challenge/ezmock"
)


@dataclass(frozen=True)
class GridConfig:
    name: str
    stats: tuple[str, ...]
    p_ells: tuple[int, ...] = (0, 2)
    p_kmax: float = 0.20
    b_ells: tuple[tuple[int, int, int], ...] = ((0, 0, 0), (2, 0, 2))
    b_kmax: float = 0.20
    b202_kmax: float = 0.03


def _config_grid(kmax_values=None, include_p_ell4=False, include_b202_full=False) -> list[GridConfig]:
    """Return the baseline local grid covered by the default broad covariance."""
    p_kmax_values = tuple(kmax_values or (0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20))
    configs = []
    for p_kmax in p_kmax_values:
        tag = f"{p_kmax:.2f}".replace(".", "p")
        ell_sets = [("l0", (0,)), ("l02", (0, 2))]
        if include_p_ell4:
            ell_sets.append(("l024", (0, 2, 4)))
        for ell_label, p_ells in ell_sets:
            configs.append(GridConfig(
                name=f"P_{ell_label}_k{tag}",
                stats=("mesh2_spectrum",),
                p_ells=p_ells,
                p_kmax=p_kmax,
            ))
            if ell_label == "l0":
                continue
            configs.append(GridConfig(
                name=f"PB_{ell_label}_B000_k{tag}",
                stats=("mesh2_spectrum", "mesh3_spectrum"),
                p_ells=p_ells,
                p_kmax=p_kmax,
                b_ells=((0, 0, 0),),
                b_kmax=p_kmax,
            ))
            configs.append(GridConfig(
                name=f"PB_{ell_label}_B000B202_k{tag}",
                stats=("mesh2_spectrum", "mesh3_spectrum"),
                p_ells=p_ells,
                p_kmax=p_kmax,
                b_ells=((0, 0, 0), (2, 0, 2)),
                b_kmax=p_kmax,
                b202_kmax=0.03,
            ))
            if include_b202_full:
                configs.append(GridConfig(
                    name=f"PB_{ell_label}_B000B202full_k{tag}",
                    stats=("mesh2_spectrum", "mesh3_spectrum"),
                    p_ells=p_ells,
                    p_kmax=p_kmax,
                    b_ells=((0, 0, 0), (2, 0, 2)),
                    b_kmax=p_kmax,
                    b202_kmax=p_kmax,
                ))
    return configs


def _parse_names(names: list[str] | None, kmax_values=None, include_p_ell4=False, include_b202_full=False) -> list[GridConfig]:
    configs = _config_grid(
        kmax_values=kmax_values,
        include_p_ell4=include_p_ell4,
        include_b202_full=include_b202_full,
    )
    if not names:
        return configs
    by_name = {config.name: config for config in configs}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise ValueError(f"unknown config name(s): {missing}; available names are {sorted(by_name)}")
    return [by_name[name] for name in names]


def _selects_from_config(config: GridConfig) -> dict[str, list[dict]]:
    selects = {}
    if "mesh2_spectrum" in config.stats:
        selects["mesh2_spectrum"] = [
            {"ells": ell, "k": [0.02, float(config.p_kmax), 0.01]}
            for ell in config.p_ells
        ]
    if "mesh3_spectrum" in config.stats:
        bselects = []
        for ell in config.b_ells:
            kmax = config.b202_kmax if ell == (2, 0, 2) else config.b_kmax
            bselects.append({"ells": ell, "k": [0.02, float(kmax), 0.01]})
        selects["mesh3_spectrum"] = bselects
    return selects


def _apply_select(observable, select):
    if select is None:
        return observable
    labels = []
    for item in select:
        item = dict(item)
        label = {}
        for key in observable.labels(return_type="keys"):
            if key in item:
                label[key] = item.pop(key)
        labels.append(label)
        pole = observable.get(**label)
        for coord_name, limits in item.items():
            if len(limits) == 3:
                step = limits[2]
                edge = pole.edges(coord_name)[0]
                rebin = int(np.rint(np.mean(step / (edge[..., 1] - edge[..., 0]))) + 0.5)
                pole = pole.select(**{coord_name: slice(0, None, rebin)})
            pole = pole.select(**{coord_name: tuple(limits[:2])})
        observable = observable.at(**label).replace(pole)
    return observable.get(labels)


def _selector_matrix(selected, broad):
    import scipy as sp

    blocks = []
    for selected_piece, broad_piece in zip(selected, broad, strict=True):
        selected_ells = list(selected_piece.ells)
        broad_ells = list(broad_piece.ells)
        broad_offsets = {}
        offset = 0
        for ell in broad_ells:
            broad_offsets[ell] = offset
            offset += broad_piece.get(ells=ell).size

        matrix = np.zeros((selected_piece.size, broad_piece.size), dtype="f8")
        row_offset = 0
        for ell in selected_ells:
            if ell not in broad_offsets:
                raise ValueError(f"selected ell {ell!r} is absent from broad cached covariance")
            spole = selected_piece.get(ells=ell)
            bpole = broad_piece.get(ells=ell)
            bcoords = np.asarray(bpole.coords("k"))
            if bcoords.ndim == 1:
                bcoords = bcoords[:, None]
            blookup = {tuple(np.round(row, 12)): i for i, row in enumerate(bcoords)}
            scoords = np.asarray(spole.coords("k"))
            if scoords.ndim == 1:
                scoords = scoords[:, None]
            for irow, coord in enumerate(scoords):
                key = tuple(np.round(coord, 12))
                if key not in blookup:
                    raise ValueError(f"selected coordinate {key!r} for ell {ell!r} is absent from broad cached covariance")
                matrix[row_offset + irow, broad_offsets[ell] + blookup[key]] = 1.0
            row_offset += spole.size
        blocks.append(matrix)
    return sp.linalg.block_diag(*blocks)


def _build_stats_from_local_cache(config: GridConfig, options: dict, data_fn: Path, covariance_fn: Path):
    import lsstypes as types
    from full_shape import box_tools, tools

    raw_data = types.read(data_fn)
    raw_covariance = types.read(covariance_fn)

    selected_observables = []
    broad_observables = []
    labels = {"observables": [], "tracers": []}
    for observable_options in options["likelihoods"][0]["observables"]:
        stat = observable_options["stat"]["kind"]
        if stat == "mesh2_spectrum":
            label = {"observables": "spectrum2", "tracers": ("QSO", "QSO")}
        elif stat == "mesh3_spectrum":
            label = {"observables": "spectrum3", "tracers": ("QSO", "QSO", "QSO")}
        else:
            raise ValueError(f"unsupported statistic {stat!r}")
        selected = _apply_select(raw_data.get(**label), observable_options["stat"].get("select"))
        broad = _apply_select(raw_data.get(**label), DEFAULT_BROAD_SELECTS[stat])
        selected_observables.append(selected)
        broad_observables.append(broad)
        labels["observables"].append(label["observables"])
        labels["tracers"].append(label["tracers"])

    data = tools.pack_stats(selected_observables, **labels)
    broad_data = tools.pack_stats(broad_observables, **labels)
    covariance = raw_covariance.at.observable.match(data)
    window = types.WindowMatrix(
        value=_selector_matrix(selected_observables, broad_observables),
        observable=data,
        theory=broad_data,
        attrs={"mode": "local-cache-selector"},
    )

    covariance_options = options["likelihoods"][0]["covariance"]
    factor, metadata = tools._get_covariance_correction_factor(
        covariance,
        options["likelihoods"][0]["observables"],
        covariance_options,
    )
    covariance = covariance.clone(value=covariance.value() * factor)
    covariance.attrs["covariance_correction_factor"] = float(factor)
    for name, value in metadata.items():
        covariance.attrs[name] = value

    volume_scale = box_tools.get_covariance_volume_scale_factor(covariance_options.get("volume_rescaling"))
    covariance = covariance.clone(value=covariance.value() * volume_scale)
    covariance.attrs["volume_scale_factor"] = float(volume_scale)

    return types.GaussianLikelihood(observable=data, window=window, covariance=covariance)


def _build_options(config: GridConfig, args):
    from full_shape.job_scripts import validation_box_mocks

    old_kranges = validation_box_mocks.KRANGES
    validation_box_mocks.KRANGES = _selects_from_config(config)
    try:
        options = validation_box_mocks._build_run_options(
            stats=config.stats,
            tracer=args.tracer,
            version=args.version,
            zsnap=args.zsnap,
            imocks=args.imocks,
            los=args.los,
            hod=args.hod,
            cosmo=args.cosmo,
            stats_dir=args.stats_dir,
            covariance_stats_dir=args.covariance_stats_dir,
            theory_model=args.theory_model,
            prior_basis=args.prior_basis,
            cosmo_model=args.cosmo_params,
            template=args.template,
            sampler=args.sampler,
            nchains=args.nchains,
            resume=args.resume,
            emulator=False if args.no_emulator else None,
        )
    finally:
        validation_box_mocks.KRANGES = old_kranges
    if args.profiler_niterations is not None:
        options.setdefault("profiler", {}).setdefault("maximize", {})
        options["profiler"]["maximize"]["niterations"] = args.profiler_niterations
    return options


def _run_one(config: GridConfig, args):
    import functools

    from clustering_statistics import box_tools as clustering_box_tools
    from full_shape import tools
    from full_shape import run_fit_from_options
    from full_shape.tools import get_fits_fn

    options = _build_options(config, args)
    if args.local_cached_stats:
        stats = _build_stats_from_local_cache(config, options, args.cached_data, args.cached_covariance)
        options["likelihoods"][0]["stats"] = stats
        tools.rebin_spectrum3_window = lambda window, data=None: window
        tools.select_window_theory = lambda window, data: window

    fits_fn = functools.partial(get_fits_fn, fits_dir=args.fits_dir, extra=config.name)
    path_options = copy.deepcopy(options)
    for likelihood_options in path_options["likelihoods"]:
        likelihood_options.pop("stats", None)
    run_fit_from_options(
        args.todo,
        **options,
        get_stats_fn=clustering_box_tools.get_box_stats_fn,
        get_fits_fn=fits_fn,
        cache_dir=args.cache_dir,
        cache_mode=args.cache_mode,
    )
    return get_fits_fn(kind="profiles", fits_dir=args.fits_dir, extra=config.name, **path_options)


def _write_manifest(configs: list[GridConfig], args):
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "configs": [asdict(config) | {"selects": _selects_from_config(config)} for config in configs],
        "cache_dir": str(args.cache_dir),
        "fits_dir": str(args.fits_dir),
    }
    args.output_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--todo", type=str, nargs="*", default=["profile"], choices=["build", "profile", "sample"])
    parser.add_argument("--config", type=str, nargs="*", help="Optional config names from the built-in grid.")
    parser.add_argument("--kmax-values", type=float, nargs="*", help="Override the default P/B000 kmax grid.")
    parser.add_argument("--include-p-ell4", action="store_true",
                        help="Also include P ell0+2+4 and matching P+B configurations.")
    parser.add_argument("--include-b202-full", action="store_true",
                        help="Also include P+B configurations with B202 kmax matched to B000/P kmax.")
    parser.add_argument("--local-cached-stats", action="store_true", help="Use local cached broad data/covariance.")
    parser.add_argument("--cached-data", type=Path, default=DEFAULT_RAW_DATA)
    parser.add_argument("--cached-covariance", type=Path, default=DEFAULT_RAW_COVARIANCE)
    parser.add_argument("--cache-mode", default="rw", choices=["r", "rw"])
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_FITS_DIR / "grid_manifest.json")
    parser.add_argument("--profiler-niterations", type=int, default=None,
                        help="Override profiler.maximize niterations.")

    parser.add_argument("--tracer", default="QSO_lorentzian")
    parser.add_argument("--version", default="abacus-hf-v2")
    parser.add_argument("--zsnap", type=float, default=1.475)
    parser.add_argument("--theory_model", default="folpsD")
    parser.add_argument("--prior_basis", default="physical_aap")
    parser.add_argument("--cosmo_params", default="base")
    parser.add_argument("--sampler", default="emcee")
    parser.add_argument("--imocks", default="all")
    parser.add_argument("--los", default=None)
    parser.add_argument("--hod", default=None)
    parser.add_argument("--cosmo", default=None)
    parser.add_argument("--stats_dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--covariance_stats_dir", type=Path, default=DEFAULT_COVARIANCE_STATS_DIR)
    parser.add_argument("--fits_dir", type=Path, default=DEFAULT_FITS_DIR)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--template", default="direct")
    parser.add_argument("--nchains", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_emulator", action="store_true")
    parser.add_argument("--local-safe-threads", action="store_true")
    return parser


def main():
    args = _get_parser().parse_args()
    if args.local_safe_threads:
        for name, value in {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "MPLCONFIGDIR": "/private/tmp/matplotlib-cache",
            "OMPI_MCA_btl": "self",
        }.items():
            os.environ.setdefault(name, value)
    setup_logging()
    configs = _parse_names(
        args.config,
        kmax_values=args.kmax_values,
        include_p_ell4=args.include_p_ell4,
        include_b202_full=args.include_b202_full,
    )
    _write_manifest(configs, args)
    for config in configs:
        print(f"Running {config.name}: {config}")
        profile_fn = _run_one(config, args)
        print(f"Wrote {profile_fn}")


if __name__ == "__main__":
    main()
