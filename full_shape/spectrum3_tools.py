import logging

import jax


logger = logging.getLogger('spectrum3')


def compute_mesh3_spectrum(*particles,
                            basis='scoccimarro', ells=[0, 2], edges=None, los='local', mattrs=None,
                            buffer_size=0, cache=None):
    from jaxpower import (FKPField, compute_fkp3_normalization, compute_fkp3_shotnoise, BinMesh3SpectrumPoles, compute_mesh3_spectrum)

    attrs = _get_jaxpower_attrs(*particles)
    mattrs = particles[0][0].attrs
    # Define the binner
    if cache is None: cache = {}
    bin = cache.get('bin_mesh3_spectrum', None)
    if edges is None: edges = {'step': 0.01 if 'scoccimarro' in basis else 0.005}
    if bin is None: bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis=basis, ells=ells, buffer_size=buffer_size)
    cache.setdefault('bin_mesh3_spectrum', bin)

    # Computing normalization
    all_fkp = [FKPField(data, randoms) for (data, randoms, _) in particles]
    norm = compute_fkp3_normalization(*all_fkp, bin=bin, split=42, cellsize=10)

    # Computing shot noise
    all_fkp = [FKPField(data, shifted) for (data, _, shifted) in particles]
    num_shotnoise = compute_fkp3_shotnoise(*all_fkp, bin=bin)

    kw = dict(resampler='tsc', interlacing=3, compensate=True)
    num_shotnoise = compute_fkp3_shotnoise(*all_fkp, los=los, bin=bin, **kw)
    jitted_compute_mesh3_spectrum = jax.jit(compute_mesh3_spectrum, static_argnames=['los'], donate_argnums=[0])

    spectrum = jitted_compute_mesh3_spectrum(*[fkp.paint(**kw, out='complex') for fkp in all_fkp], los=los, bin=bin)
    spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise, attrs=attrs)
    return spectrum