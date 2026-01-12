import logging

import jax


logger = logging.getLogger('spectrum2')


def prepare_jaxpower_particles(*get_data_randoms, mattrs=None, backend='mpi', **kwargs):
    from jaxpower import get_mesh_attrs, ParticleField

    all_data, all_randoms, all_shifted = [], [], []
    for _get_data_randoms in get_data_randoms:
        # data, randoms (optionally shifted) are tuples (positions, weights)
        data, randoms, *shifted = _get_data_randoms()
        all_data.append(data)
        all_randoms.append(randoms)
        if shifted:
            all_shifted.append(shifted)

    if all_shifted:
        assert len(all_shifted) == len(data), 'Give as many shifted randoms as data/randoms'

    # Define the mesh attributes; pass in positions only
    mattrs = get_mesh_attrs(*[data[0] for data in all_data + all_shifted + all_randoms], check=True, **(mattrs or {}))
    if jax.process_index() == 0:
        logger.info(f'Using mesh {mattrs}.')

    all_particles = []
    for i, (data, randoms) in enumerate(zip(all_data, all_randoms)):
        data = ParticleField(*data, attrs=mattrs, exchange=True, backend=backend, **kwargs)
        randoms = ParticleField(*randoms, attrs=mattrs, exchange=True, backend=backend, **kwargs)
        if all_shifted:
            shifted = ParticleField(*shifted, attrs=mattrs, exchange=True, backend=backend, **kwargs)
        else:
            shifted = None
        all_particles.append((data, randoms, shifted))
    if jax.process_index() == 0:
        logger.info(f'All particles on the device')

    return all_particles


def _get_jaxpower_attrs(*particles):
    mattrs = particles[0][0].attrs
    # Creating FKP fields
    attrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    for i, (data, randoms, shifted) in enumerate(particles):
        attrs[f'size_data{i:d}'], attrs[f'wsum_data{i:d}'] = data.size, data.sum()
        attrs[f'size_randoms{i:d}'], attrs[f'wsum_randoms{i:d}'] = randoms.size, randoms.sum()
        if shifted is not None:
            attrs[f'size_shifted{i:d}'], attrs[f'wsum_shifted{i:d}'] = shifted.size, shifted.sum()
    return attrs


def compute_mesh2_spectrum(*particles, cut=None, auw=None,
                           ells=(0, 2, 4), edges=None, los='firstpoint', cache=None):

    from jaxpower import (FKPField, compute_fkp2_normalization, compute_fkp2_shotnoise, BinMesh2SpectrumPoles, compute_mesh2_spectrum,
                          BinParticle2SpectrumPoles, BinParticle2CorrelationPoles, compute_particle2, compute_particle2_shotnoise)

    attrs = _get_jaxpower_attrs(*particles)
    mattrs = particles[0][0].attrs
    # Define the binner
    if cache is None: cache = {}
    bin = cache.get('bin_mesh2_spectrum', None)
    if edges is None: edges = {'step': 0.001}
    if bin is None: bin = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)
    cache.setdefault('bin_mesh2_spectrum', bin)

    # Computing normalization
    all_fkp = [FKPField(data, randoms) for (data, randoms, _) in particles]
    norm = compute_fkp2_normalization(*all_fkp, bin=bin, cellsize=10)

    # Computing shot noise
    all_fkp = [FKPField(data, shifted if shifted is not None else randoms) for (data, randoms, shifted) in particles]
    num_shotnoise = compute_fkp2_shotnoise(*all_fkp, bin=bin)

    jax.block_until_ready((norm, num_shotnoise))
    if jax.process_index() == 0:
        logger.info('Normalization and shotnoise computation finished')

    jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
    #jitted_compute_mesh2_spectrum = compute_mesh2_spectrum
    kw = dict(resampler='tsc', interlacing=3, compensate=True)
    # out='real' to save memory
    spectrum = jitted_compute_mesh2_spectrum(*[fkp.paint(**kw, out='real') for fkp in all_fkp], bin=bin, los=los)
    spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise, attrs=attrs)

    results = {'raw': spectrum}

    jax.block_until_ready(spectrum)
    if jax.process_index() == 0:
        logger.info('Mesh-based computation finished')

    if cut is not None:
        sattrs = {'theta': (0., 0.05)}
        #pbin = BinParticle2SpectrumPoles(mattrs, edges=bin.edges, xavg=bin.xavg, sattrs=sattrs, ells=ells)
        pbin = BinParticle2CorrelationPoles(mattrs, edges={'step': 0.1}, sattrs=sattrs, ells=ells)
        from jaxpower.particle2 import convert_particles
        all_particles = [convert_particles(fkp.particles) for fkp in all_fkp]
        close = compute_particle2(*all_particles, bin=pbin, los=los)
        close = close.clone(num_shotnoise=compute_particle2_shotnoise(*all_particles, bin=pbin), norm=norm)
        close = close.to_spectrum(spectrum)
        results['cut'] = spectrum.clone(value=spectrum.value() - close.value())

    if auw is not None:
        from cucount.jax import WeightAttrs
        from jaxpower.particle2 import convert_particles
        sattrs = {'theta': (0., 0.1)}
        all_data = [convert_particles(fkp.data, weights=[fkp.data.weights] * 2, exchange_weights=False, index_value=dict(individual_weight=1, negative_weight=1)) for fkp in all_fkp]
        wattrs = WeightAttrs(angular=dict(sep=auw.get('DD').coords('theta'), weight=auw.get('DD').value()) if auw is not None else None)
        pbin = BinParticle2SpectrumPoles(mattrs, edges=bin.edges, xavg=bin.xavg, sattrs=sattrs, wattrs=wattrs, ells=ells)
        DD = compute_particle2(*all_data, bin=pbin, los=los)
        DD = DD.clone(num_shotnoise=compute_particle2_shotnoise(*all_data, bin=pbin), norm=norm)
        results['auw'] = spectrum.clone(value=spectrum.value() + DD.value())

    jax.block_until_ready(results)
    if jax.process_index() == 0:
        logger.info(f'Particle-based calculation finished')

    if len(results) == 1:
        return next(iter(results.values()))
    return results