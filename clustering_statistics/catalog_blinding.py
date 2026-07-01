"""Thin desi-clustering adapter for desiblind catalog-level blinding."""

import logging
from pathlib import Path

import numpy as np
from mpi4py import MPI

from desiblind import TracerCatalogBlinder


logger = logging.getLogger("catalog-blinding")


def get_blinding_modes(options):
    """Return normalized catalog-blinding modes from a config dictionary."""
    if not options:
        return ()
    return TracerCatalogBlinder.get_blinding_modes(options.get("modes", options.get("mode", options.get("kind", "bao"))))


def get_blinding_parameters(options, tracer="LRG", mpicomm=None):
    """Resolve sealed catalog blinding parameters for one tracer."""
    del mpicomm
    if not options:
        return None
    return TracerCatalogBlinder.from_options(options, tracer=tracer)


def apply_bao_blinding_to_catalogs(catalogs, params, zcol="Z"):
    return TracerCatalogBlinder.apply_bao_blinding_to_catalogs(catalogs, params, zcol=zcol)


def apply_rsd_blinding(data, randoms, params, tracer="LRG"):
    return TracerCatalogBlinder.apply_rsd_blinding(data, randoms, params, tracer=tracer)


def apply_fnl_blinding(data, randoms, params, tracer="LRG"):
    return TracerCatalogBlinder.apply_fnl_blinding(data, randoms, params, tracer=tracer)


def output_version(version, params):
    return TracerCatalogBlinder.output_version(version, params)


def output_version_from_options(version, options):
    return TracerCatalogBlinder.output_version_from_options(version, options)


def _safe_tracer_key(tracer):
    return str(tracer).replace("-", "_").replace(".", "_").replace("/", "_")


def blinding_attrs(params):
    """Return attrs for one tracer or a tracer->params mapping."""
    if not params:
        return {}
    if "modes" in params:
        return TracerCatalogBlinder.blinding_attrs(params)

    attrs = {}
    active = {tracer: value for tracer, value in params.items() if value}
    if not active:
        return attrs
    if len(active) == 1:
        return TracerCatalogBlinder.blinding_attrs(next(iter(active.values())))
    attrs["catalog_blinding_tracers"] = ",".join(map(str, active))
    attrs["catalog_blinding"] = ";".join(
        f"{tracer}:{','.join(value['modes'])}" for tracer, value in active.items()
    )
    metadata = sorted({value.get("metadata", "sealed") for value in active.values()})
    attrs["catalog_blinding_metadata"] = ",".join(metadata)
    for tracer, value in active.items():
        prefix = f"catalog_blinding_{_safe_tracer_key(tracer)}"
        for key, attr_value in TracerCatalogBlinder.blinding_attrs(value).items():
            attrs[f"{prefix}_{key.removeprefix('catalog_blinding_')}"] = attr_value
    return attrs


def _to_builtin(value):
    """Return a plain-Python representation of DESI option values for YAML output."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(val) for val in value]
    return value


def catalog_for_writing(catalog, remove_columns=("POSITION", "INDWEIGHT", "BITWEIGHT")):
    """Return a shallow copy without transient measurement-only columns."""
    catalog = catalog.copy()
    for column in remove_columns:
        if column in catalog:
            del catalog[column]
    return catalog


def write_catalog(filename, catalog, mpicomm=None, group="LSS", remove_columns=("POSITION", "INDWEIGHT", "BITWEIGHT"), overwrite=False):
    """Gather and write a catalog in the same style as existing job scripts."""
    mpicomm = mpicomm or getattr(catalog, "mpicomm", MPI.COMM_WORLD)
    filename = Path(filename)
    if mpicomm.size > 1:
        catalog = catalog.gather(mpiroot=0)
    if mpicomm.rank == 0:
        filename.parent.mkdir(parents=True, exist_ok=True)
        if filename.exists():
            if overwrite:
                filename.unlink()
            else:
                raise FileExistsError(f"{filename} already exists; pass overwrite=True only after checking this is safe")
        catalog = catalog_for_writing(catalog, remove_columns=remove_columns)
        kwargs = {"group": group} if str(filename).endswith((".h5", ".hdf5")) else {}
        catalog.write(filename, **kwargs)
        logger.info("Wrote %s", filename)
    mpicomm.Barrier()
    return filename


def write_blinding_record(filename, record, mpicomm=None, overwrite=False):
    """Write a sealed YAML record for blinded catalog products."""
    import yaml

    mpicomm = mpicomm or MPI.COMM_WORLD
    filename = Path(filename)
    if mpicomm.rank == 0:
        filename.parent.mkdir(parents=True, exist_ok=True)
        if filename.exists():
            if overwrite:
                filename.unlink()
            else:
                raise FileExistsError(f"{filename} already exists; pass overwrite=True only after checking this is safe")
        with filename.open("w") as file:
            yaml.safe_dump(_to_builtin(record), file, sort_keys=False)
        logger.info("Wrote %s", filename)
    mpicomm.Barrier()
    return filename
