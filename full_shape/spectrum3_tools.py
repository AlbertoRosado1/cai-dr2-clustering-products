import logging

import jax

from spectrum2_tools import _get_jaxpower_attrs


logger = logging.getLogger('spectrum3')


def compute_mesh3_spectrum(*particles,
                            basis='sugiyama-diagonal', ells=[(0, 0, 0), (2, 0, 2)], edges=None, los='local',
                            buffer_size=0, cache=None):
    from jaxpower import (FKPField, compute_fkp3_normalization, compute_fkp3_shotnoise, BinMesh3SpectrumPoles, compute_mesh3_spectrum)

    attrs = _get_jaxpower_attrs(*particles)
    mattrs = particles[0][0].attrs
    # Define the binner
    if cache is None: cache = {}
    bin = cache.get('bin_mesh3_spectrum', None)
    if edges is None: edges = {'step': 0.01 if 'scoccimarro' in basis else 0.005}
    print(edges)
    if bin is None: bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis=basis, ells=ells, buffer_size=buffer_size)
    cache.setdefault('bin_mesh3_spectrum', bin)

    # Computing normalization
    all_fkp = [FKPField(data, randoms) for (data, randoms, _) in particles]
    norm = compute_fkp3_normalization(*all_fkp, bin=bin, split=42, cellsize=10)

    # Computing shot noise
    all_fkp = [FKPField(data, shifted if shifted is not None else randoms) for (data, randoms, shifted) in particles]
    kw = dict(resampler='tsc', interlacing=3, compensate=True)
+   kw = dict(resampler='tsc', interlacing=3, compensate=True)
    num_shotnoise = compute_fkp3_shotnoise(*all_fkp, los=los, bin=bin, **kw)

    jax.block_until_ready((norm, num_shotnoise))
    if jax.process_index() == 0:
        logger.info('Normalization and shotnoise computation finished')

    jitted_compute_mesh3_spectrum = jax.jit(compute_mesh3_spectrum, static_argnames=['los'], donate_argnums=[0])

    # out='real' to save memory
    spectrum = jitted_compute_mesh3_spectrum(*[fkp.paint(**kw, out='real') for fkp in all_fkp], los=los, bin=bin)
    spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise, attrs=attrs)

    jax.block_until_ready(spectrum)
    if jax.process_index() == 0:
        logger.info('Mesh-based computation finished')

    return spectrum