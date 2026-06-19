"""
Produce saved catalog-level blinded catalogs, optionally through desipipe.

The blinding parameters are resolved by ``desiblind.TracerCatalogBlinder`` from
the sealed catalog secret file.  By default the YAML record stores provenance
and secret-file identity, not the blind parameter values.

Examples
--------
Dry-run path planning on an interactive node::

    srun -n 4 python clustering_statistics/job_scripts/desipipe_catalog_blinding.py \
        --interactive --dry-run --tracer LRG --version data-dr2-v2 \
        --modes bao rsd fnl --parameters_fn /path/to/catalog_blinding_2026_06.npy \
        --regions NGC --maxr 1

Remove ``--dry-run`` only after checking the printed output paths. By default
this script refuses to overwrite existing files and refuses to write under
``/global/cfs/cdirs/desi`` unless ``--allow-cfs-output`` is explicitly passed.
"""

import argparse
import logging
import os
from pathlib import Path

from mpi4py import MPI

from clustering_statistics import catalog_blinding
from clustering_statistics import tools
from clustering_statistics.tools import setup_logging


logger = logging.getLogger("catalog-blinding-jobs")


def _normalize_tracer(tracer, version=None):
    """Allow simple BGS/ELG aliases while leaving already-full names alone."""
    if tracer in ["BGS", "ELG"]:
        return tools.get_full_tracer(tracer, version=version)
    return tracer


def _make_blinding_options(
    modes=("bao",),
    metadata="sealed",
    parameters_fn=None,
    save_dir=None,
    output_version_suffix=None,
    fiducial_f=0.8,
    zeff=None,
    bias=None,
    rsd_smoothing_radius=15.0,
    fnl_smoothing_radius=30.0,
    fnl_method="data_weights",
):
    options = dict(
        modes=list(modes),
        metadata=metadata,
        fiducial_f=fiducial_f,
        rsd_smoothing_radius=rsd_smoothing_radius,
        fnl_smoothing_radius=fnl_smoothing_radius,
        fnl_method=fnl_method,
    )
    for key, value in dict(
        parameters_fn=parameters_fn,
        save_dir=save_dir,
        output_version_suffix=output_version_suffix,
        zeff=zeff,
        bias=bias,
    ).items():
        if value is not None:
            options[key] = value
    return options


def _default_output_cat_dir(version=None, cat_dir=None, modes=("bao",), output_version=None, output_root=None, blinding_options=None):
    input_version = version or (Path(cat_dir).name if cat_dir is not None else "custom")
    if output_version is None:
        output_version = catalog_blinding.output_version_from_options(input_version, blinding_options or {"modes": modes})
    if output_root is None:
        output_root = Path(os.getenv("SCRATCH", ".")) / "desi-clustering" / "blinded_catalogs"
    return Path(output_root) / output_version / "LSScats"


def _is_relative_to(path, parent):
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _ensure_safe_output_dir(output_cat_dir, allow_cfs_output=False):
    """Refuse accidental writes to collaboration CFS catalog areas."""
    output_cat_dir = Path(output_cat_dir).expanduser().resolve(strict=False)
    protected_roots = [Path("/global/cfs/cdirs/desi"), Path("/dvs_ro/cfs/cdirs/desi")]
    if not allow_cfs_output:
        for root in protected_roots:
            if _is_relative_to(output_cat_dir, root):
                raise ValueError(
                    f"Refusing to write blinded catalogs under protected collaboration path {root}: {output_cat_dir}. "
                    "Use $SCRATCH for tests, or pass --allow-cfs-output only for an intentional production write."
                )
    return output_cat_dir


def _safe_weight_blind(prepared_data):
    if "WEIGHT_BLIND" not in prepared_data:
        return None
    return prepared_data["WEIGHT_BLIND"]


def _copy_prepared_columns_to_raw(raw_data, prepared_data, modes):
    """Copy persisted RSD/fNL changes from prepared data back to raw-style data."""
    if len(raw_data) != len(prepared_data):
        raise ValueError(
            "Prepared data length differs from raw data length; saved catalog blinding currently expects no object filtering. "
            "Try default-FKP weights first."
        )
    if "rsd" in modes:
        for column in ["RA", "DEC", "Z"]:
            raw_data[column] = prepared_data[column]
    if "fnl" in modes:
        weight_blind = _safe_weight_blind(prepared_data)
        if weight_blind is None:
            raise ValueError("fNL blinding did not return WEIGHT_BLIND")
        raw_data["WEIGHT_BLIND"] = weight_blind
        raw_data["WEIGHT"] = raw_data["WEIGHT"] * weight_blind
    raw_data.attrs.update(getattr(prepared_data, "attrs", {}))
    return raw_data


def _split_and_write(
    catalog,
    kind,
    output_cat_dir,
    catalog_options,
    regions=("NGC", "SGC"),
    iran=None,
    output_ext="fits",
    mpicomm=None,
    dry_run=False,
    overwrite=False,
):
    mpicomm = mpicomm or MPI.COMM_WORLD
    written = []
    for region in regions:
        kwargs = dict(catalog_options) | dict(region=region, ext=output_ext, cat_dir=output_cat_dir, version=None)
        if kind == "randoms":
            kwargs["nran"] = [iran]
        fn = tools.get_catalog_fn(kind=kind, **kwargs)
        if isinstance(fn, list):
            fn = fn[0]
        written.append(str(fn))
        if dry_run:
            continue
        region_catalog = catalog[tools.select_region(catalog["RA"], catalog["DEC"], region)]
        catalog_blinding.write_catalog(fn, region_catalog, mpicomm=mpicomm, overwrite=overwrite)
    return written


def make_blinded_catalogs(
    tracer="LRG",
    version="data-dr2-v2",
    cat_dir=None,
    output_cat_dir=None,
    output_root=None,
    output_version=None,
    output_ext="fits",
    regions=("NGC", "SGC"),
    weight="default-FKP",
    analysis="full_shape",
    nran=None,
    minr=0,
    maxr=None,
    modes=("bao",),
    metadata="sealed",
    parameters_fn=None,
    save_dir=None,
    output_version_suffix=None,
    fiducial_f=0.8,
    zeff=None,
    bias=None,
    rsd_smoothing_radius=15.0,
    fnl_smoothing_radius=30.0,
    fnl_method="data_weights",
    dry_run=False,
    overwrite=False,
    allow_cfs_output=False,
    read_ext=None,
):
    """Read, blind, and save clustering catalogs for one tracer."""
    mpicomm = MPI.COMM_WORLD
    setup_logging(level=(logging.INFO if mpicomm.rank == 0 else logging.ERROR))

    tracer = _normalize_tracer(tracer, version=version)
    modes = catalog_blinding.get_blinding_modes({"modes": modes})
    blinding_options = _make_blinding_options(
        modes=modes,
        metadata=metadata,
        parameters_fn=parameters_fn,
        save_dir=save_dir,
        output_version_suffix=output_version_suffix,
        fiducial_f=fiducial_f,
        zeff=zeff,
        bias=bias,
        rsd_smoothing_radius=rsd_smoothing_radius,
        fnl_smoothing_radius=fnl_smoothing_radius,
        fnl_method=fnl_method,
    )
    params = catalog_blinding.get_blinding_parameters(blinding_options, tracer=tracer, mpicomm=mpicomm)

    fiducial_catalog = tools.propose_fiducial("catalog", tracer=tracer, analysis=analysis)
    if nran is None:
        nran = fiducial_catalog["nran"]
    if maxr is None:
        maxr = nran
    nrans = list(range(minr, maxr))
    if not nrans:
        raise ValueError("No random catalogs selected")

    catalog_options = fiducial_catalog | dict(version=version, cat_dir=cat_dir, tracer=tracer, weight=weight, nran=nrans, ext=read_ext)
    output_cat_dir = Path(output_cat_dir) if output_cat_dir is not None else _default_output_cat_dir(
        version=version,
        cat_dir=cat_dir,
        modes=modes,
        output_version=output_version,
        output_root=output_root,
        blinding_options=blinding_options,
    )
    if not dry_run:
        output_cat_dir = _ensure_safe_output_dir(output_cat_dir, allow_cfs_output=allow_cfs_output)

    output_catalog_options = dict(version=None, cat_dir=output_cat_dir, tracer=tracer, weight=weight, ext=output_ext, nran=nrans)
    planned = {"data": [], "randoms": []}
    for region in regions:
        planned["data"].append(str(tools.get_catalog_fn(kind="data", region=region, **output_catalog_options)))
        planned["randoms"].extend(map(str, tools.get_catalog_fn(kind="randoms", region=region, **output_catalog_options)))

    blinding_record = dict(
        input=dict(version=version, cat_dir=cat_dir, tracer=tracer, weight=weight, nran=nrans, ext=read_ext),
        output=dict(cat_dir=output_cat_dir, ext=output_ext, files=planned),
        regions=list(regions),
        blinding=catalog_blinding.blinding_attrs(params),
        modes=list(modes),
        metadata=metadata,
    )
    blinding_record_fn = output_cat_dir.parent / f"blinding_record_{tracer}.yaml"

    if dry_run:
        if mpicomm.rank == 0:
            logger.info("Dry run for %s", tracer)
            logger.info("Output catalog directory: %s", output_cat_dir)
            for kind, fns in planned.items():
                for fn in fns:
                    logger.info("Would write %s: %s", kind, fn)
            logger.info("Would write blinding record: %s", blinding_record_fn)
        return str(output_cat_dir)

    if any(token in weight.lower() for token in ["bitwise", "compntile"]):
        raise NotImplementedError(
            "Saved catalog blinding is currently validated for non-bitwise weights; validate bitwise/compntile before production use."
        )

    if mpicomm.rank == 0:
        logger.info("Reading data catalog for %s", tracer)
    raw_data = tools.read_catalog(kind="data", concatenate=True, keep_columns=True, region="ALL", mpicomm=mpicomm, **catalog_options)
    raw_data = catalog_blinding.apply_bao_blinding_to_catalogs(raw_data, params)

    raw_randoms = []
    for iran in nrans:
        if mpicomm.rank == 0:
            logger.info("Reading random catalog %s for %s", iran, tracer)
        random = tools.read_catalog(
            kind="randoms",
            concatenate=True,
            keep_columns=True,
            region="ALL",
            mpicomm=mpicomm,
            **(catalog_options | dict(nran=[iran])),
        )
        random = catalog_blinding.apply_bao_blinding_to_catalogs(random, params)
        raw_randoms.append(random)

    if any(mode in modes for mode in ["rsd", "fnl"]):
        if mpicomm.rank == 0:
            logger.info("Preparing catalogs for RSD/fNL blinding for %s", tracer)
        prepared_data = tools.prepare_catalog(
            raw_data,
            kind="data",
            binned_weight={},
            region=None,
            zrange=None,
            weight=weight,
            FKP_P0=catalog_options.get("FKP_P0", None),
        )
        prepared_randoms = [
            tools.prepare_catalog(
                random,
                kind="randoms",
                binned_weight={},
                region=None,
                zrange=None,
                weight=weight,
                FKP_P0=catalog_options.get("FKP_P0", None),
            )
            for random in raw_randoms
        ]
        prepared_data = catalog_blinding.apply_rsd_blinding(prepared_data, prepared_randoms, params, tracer=tracer)
        prepared_data, prepared_randoms = catalog_blinding.apply_fnl_blinding(prepared_data, prepared_randoms, params, tracer=tracer)
        raw_data = _copy_prepared_columns_to_raw(raw_data, prepared_data, modes)

    blinding_record["blinding"] = catalog_blinding.blinding_attrs(params)
    catalog_blinding.write_blinding_record(blinding_record_fn, blinding_record, mpicomm=mpicomm, overwrite=overwrite)
    _split_and_write(raw_data, "data", output_cat_dir, catalog_options | dict(nran=nrans), regions=regions, output_ext=output_ext, mpicomm=mpicomm, overwrite=overwrite)
    for iran, random in zip(nrans, raw_randoms, strict=True):
        _split_and_write(
            random,
            "randoms",
            output_cat_dir,
            catalog_options | dict(nran=[iran]),
            regions=regions,
            iran=iran,
            output_ext=output_ext,
            mpicomm=mpicomm,
            overwrite=overwrite,
        )

    return str(output_cat_dir)


def setup_queue(queue_name="catalog_blinding", max_workers=4, time="04:00:00", mpiprocs_per_worker=128, constraint="cpu"):
    from desipipe import Environment, Queue, TaskManager, setup_logging as desipipe_setup_logging

    desipipe_setup_logging()
    queue = Queue(queue_name)
    queue.clear(kill=False)
    output = f"./slurm_outputs/{queue_name}/slurm-%j.out"
    error = f"./slurm_outputs/{queue_name}/slurm-%j.err"
    environ = Environment("nersc-cosmodesi", command="module unload desi-clustering || true")
    tm = TaskManager(queue=queue, environ=environ)
    return tm.clone(
        scheduler=dict(max_workers=max_workers),
        provider=dict(
            provider="nersc",
            time=time,
            mpiprocs_per_worker=mpiprocs_per_worker,
            output=output,
            error=error,
            stop_after=1,
            constraint=constraint,
        ),
    )


def collect_argparser():
    parser = argparse.ArgumentParser(description="Produce saved catalog-level blinded DESI clustering catalogs.")
    parser.add_argument("--interactive", action="store_true", help="Run directly under the current allocation instead of creating desipipe tasks.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs without reading or writing catalogs.")
    parser.add_argument("--tracer", nargs="+", default=["LRG"], help="Tracer(s) to process.")
    parser.add_argument("--version", default="data-dr2-v2", help="Input catalog version registered in tools.get_catalog_fn.")
    parser.add_argument("--cat_dir", default=None, help="Input catalog directory, if not using a registered version.")
    parser.add_argument("--output_cat_dir", default=None, help="Output LSScats directory. If omitted, use SCRATCH/desi-clustering/blinded_catalogs/<version>/LSScats.")
    parser.add_argument("--output_root", default=None, help="Root for default output catalog directories.")
    parser.add_argument("--output_version", default=None, help="Output version directory name under output_root.")
    parser.add_argument("--regions", nargs="+", default=["NGC", "SGC"])
    parser.add_argument("--weight", default="default-FKP")
    parser.add_argument("--analysis", default="full_shape", choices=["full_shape", "png_local", "full_shape_protected"])
    parser.add_argument("--nran", type=int, default=None, help="Number of randoms if maxr is not specified.")
    parser.add_argument("--minr", type=int, default=0)
    parser.add_argument("--maxr", type=int, default=None, help="Exclusive upper random index, LSS-style.")
    parser.add_argument("--modes", nargs="+", default=["bao"], choices=["bao", "ap", "rsd", "fnl", "png", "local_png"])
    parser.add_argument("--metadata", default="sealed", choices=["open", "sealed"])
    parser.add_argument("--parameters_fn", default=None, help="Sealed desiblind catalog_blinding_2026_06.npy file.")
    parser.add_argument("--save_dir", default=None, help="Directory containing the default catalog blinding secret file.")
    parser.add_argument("--fiducial_f", type=float, default=0.8)
    parser.add_argument("--zeff", type=float, default=None, help="Effective redshift for RSD/fNL blinding. Defaults to desiblind fallback values by tracer.")
    parser.add_argument("--bias", type=float, default=None, help="Tracer bias for RSD/fNL blinding. Defaults to desiblind fallback values by tracer.")
    parser.add_argument("--rsd_smoothing_radius", type=float, default=15.0)
    parser.add_argument("--fnl_smoothing_radius", type=float, default=30.0)
    parser.add_argument("--fnl_method", default="data_weights")
    parser.add_argument("--read_ext", default=None)
    parser.add_argument("--output_ext", default="fits")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing output catalog/blinding-record files. Default is to fail if outputs exist.")
    parser.add_argument("--allow-cfs-output", action="store_true", help="Allow writing under /global/cfs/cdirs/desi. Use only for intentional production writes.")
    parser.add_argument("--queue-name", default="catalog_blinding")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--time", default="04:00:00")
    parser.add_argument("--mpiprocs-per-worker", type=int, default=128)
    parser.add_argument("--constraint", default="cpu")
    return parser.parse_args()


def _kwargs_from_args(args, tracer):
    keys = [
        "version",
        "cat_dir",
        "output_cat_dir",
        "output_root",
        "output_version",
        "regions",
        "weight",
        "analysis",
        "nran",
        "minr",
        "maxr",
        "modes",
        "metadata",
        "parameters_fn",
        "save_dir",
        "fiducial_f",
        "zeff",
        "bias",
        "rsd_smoothing_radius",
        "fnl_smoothing_radius",
        "fnl_method",
        "dry_run",
        "read_ext",
        "output_ext",
    ]
    kwargs = {key: getattr(args, key) for key in keys}
    kwargs["overwrite"] = args.overwrite
    kwargs["allow_cfs_output"] = args.allow_cfs_output
    kwargs["tracer"] = tracer
    return kwargs


if __name__ == "__main__":
    args = collect_argparser()
    setup_logging()
    if args.interactive:
        for tracer in args.tracer:
            make_blinded_catalogs(**_kwargs_from_args(args, tracer))
    else:
        tm = setup_queue(
            queue_name=args.queue_name,
            max_workers=args.max_workers,
            time=args.time,
            mpiprocs_per_worker=args.mpiprocs_per_worker,
            constraint=args.constraint,
        )
        app = tm.python_app(make_blinded_catalogs)
        for tracer in args.tracer:
            app(**_kwargs_from_args(args, tracer))
        if MPI.COMM_WORLD.rank == 0:
            logger.info("Created desipipe tasks in queue %s. Inspect with: desipipe tasks -q %s", args.queue_name, args.queue_name)
            logger.info("Spawn with: desipipe spawn -q %s --spawn", args.queue_name)
