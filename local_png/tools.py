import numpy as np
import matplotlib.pyplot as plt
import logging


logger = logging.getLogger('PNG fitting tools')


def read_data(data_dir='.', mocks_dir=None, 
              tracer='LRG', zrange=(0.4, 1.1), weight_type='default-fkp-oqe', region='GCcomb', 
              add_ic=False, aladr1=False, weight_type_mocks=None, **kwargs):
    """ 
    Read the data from the clustering statistics output. This is a wrapper of clustering_statistics.tools.get_stats_fn.
    """
    import lsstypes
    from clustering_statistics.tools import get_stats_fn

    # Read the data:
    pk = lsstypes.read(get_stats_fn(kind='mesh2_spectrum', stats_dir=data_dir, tracer=tracer, zrange=zrange, weight=weight_type, region=region))

    # Read the window matrix:
    if add_ic and aladr1:
        logger.info('Reading the window with integral constraint contribution (DR1 style) ...')
        window = lsstypes.read(get_stats_fn(kind='window_mesh2_spectrum', stats_dir=data_dir, tracer=tracer, zrange=zrange, weight=weight_type, region=region, extra='with_ic'))
    elif add_ic and not aladr1:
        logger.info('Reading the window with integral constraint contribution (DR2 style)...') 
        window = lsstypes.read(get_stats_fn(kind='window_mesh2_spectrum', stats_dir=data_dir, tracer=tracer, zrange=zrange, weight=weight_type, region=region, extra='RIC+AMR'))
    else:
        logger.info('Reading the window without integral constraint contribution...')
        window = lsstypes.read(get_stats_fn(kind='window_mesh2_spectrum', stats_dir=data_dir, tracer=tracer, zrange=zrange, weight=weight_type, region=region))

    # Read the analytical covariance matrix:
    try: 
        cov = lsstypes.read(get_stats_fn(kind='covariance_mesh2_spectrum', stats_dir=data_dir, tracer=tracer, zrange=zrange, weight=weight_type, region=region))
    except:
        logger.info('Do not find the analytical covariance matrix. Please provide mocks_dir to estimate the covariance matrix from mocks.')
        cov = None

    # Read the mocks:
    mocks = None
    if mocks_dir is not None: 
        weight_type_mocks = weight_type_mocks or weight_type
        if 'nmocks' in kwargs:
            nmocks = kwargs['nmocks']
            logger.info(f"Reading {nmocks} mocks for tracer {tracer} with weight type {weight_type_mocks}.")
        else:
            nmocks = 1000 if weight_type_mocks == 'default-fkp-oqe' else 100

        fns_mock = [get_stats_fn(kind='mesh2_spectrum_poles', stats_dir=mocks_dir, project='holi-v3-altmtl', tracer=tracer, region=region, zrange=zrange, 
                                 weight=weight_type_mocks, imock=imock) for imock in range(nmocks)]    
        mocks = [lsstypes.read(fn) for fn in fns_mock]

    return pk, window, cov, mocks


def rebin_data(pk, window, cov, mocks, tracer='LRG', kmin=1e-3, kmax=0.08, kpivot=[1e-2, 2e-2], nrebin=[2,2], use_ell2=True, rebin_ell2=True, **kwargs):
    """ 
    Rebin the data with k > kpivot by a factor nrebin. The quadrupole is rebinned again by a factor nrebin for the full range.
    Then, select data in the k range [kmin, kmax]. If use_ell2 is False, we only keep the monopole.
    Finally, we match the size of the window and covariance to the size of the power spectrum.
    Return the rebinned power spectrum, window and covariance.
    """
    if not isinstance(kpivot, (list, tuple)): kpivot = [kpivot]
    if not isinstance(nrebin, (list, tuple)): nrebin = [nrebin]
    assert len(kpivot) == len(nrebin), "kpivot and nrebin should have the same length."
    
    # Let's rebin the power spectrum : 
    for pivot, rebin in zip(kpivot, nrebin):
        pk = pk.map(lambda pole: pole.at(k=(pivot, 1.)).select(k=slice(0, None, rebin)))
        
    if rebin_ell2:
        # Rebin the quadrupole again but for the full range.
        pk = pk.at(2).at(k=(0, 1e-2)).select(k=slice(0, None, 2))  
        pk = pk.at(2).select(k=slice(0, None, 2))  

    # Let's select the k range and ells:
    kmin_ell2, kmax_ell2 = kwargs.get('kmin_ell2', kmin), kwargs.get('kmax_ell2', kmax)
    pk = pk.at(0).select(k=(kmin, kmax))
    pk = pk.at(2).select(k=(kmin_ell2, kmax_ell2))
    pk = pk.get(ells=[0]) if not use_ell2 else pk.get(ells=[0, 2])
 
    # Match the size of wmatrix and covariance: 
    if window is not None: window = window.at.observable.match(pk)
    
    if cov is not None: 
        tracers = tuple(tracer.split('x')) 
        if len(tracers) == 1: tracers *= 2
        tracers = (tracers[0][:3], tracers[1][:3])  # LRG_zcmb -> LRG, ELGnotqso -> ELG, ... 
        cov = cov.at.observable.at(observables='spectrum2', tracers=tracers).match(pk)

    if mocks is not None:
        mocks = [mock.match(pk) for mock in mocks]
    
    logger.debug(f'After rebinning and k range selection: {pk.get(0).k.shape[0]} and {pk.get(2).k.shape[0] if use_ell2 else "Not used"} data points.')

    return pk, window, cov, mocks


def fix_likelihood_bias_and_damping(likelihood, tracer, zeffs, derived_cross_bias=True, nickname=None, available_tracers=None, **kwargs):
    """Apply bias and damping parameter relations between the paramters of the likelihood both for ell=0/2 and auto/cross power spectrum.

    Parameters
    ----------
    likelihood : desilike likelihood
        Likelihood whose parameters are updated in place.
    tracer : str
        name of the tracers: 'LRGxLRG', 'LRGxQSO', ... 
    nickname : str, optional
        Suffix inserted between the tracer shortname and '_ell{ell}' in the cross-correlation theory
        parameter names (e.g. nickname='LRGxELG' -> 'ELG_LRGxELG_ell0'). Must match the nickname used in
        get_obervable_and_likelihood. Default is None (no suffix, legacy behaviour).
    available_tracers : list or set, optional
        List of tracers for which complete data (including auto-correlations) is available.
        Used to determine which cross-correlation biases can be derived. If None, assumes all
        auto-correlations are available.

    Returns
    -------
    likelihood
        Same likelihood object, updated in place.
    """

    def _rescale_bias_params(likelihood, tracer, zeff):
        """ 
        Fix the bias parameters in the likelihood according to the redshift dependence of the bias.
        Only modifies the parameter if both source and target exist in the likelihood.

        Args:
            likelihood: The likelihood object.
            tracer: The tracer for which to fix the bias parameters. tracer[0] is source, tracer[1] is target.
            zeff: The effective redshifts.
        """
        from clustering_statistics import tools        
        source_param_name = f"{tracer[0]}.b1"
        target_param_name = f"{tracer[1]}.b1"
        
        # Both parameters must exist to create a derived relationship
        if source_param_name not in likelihood.all_params or target_param_name not in likelihood.all_params:
            logger.debug(f"Skipping derived relationship: {source_param_name} -> {target_param_name} (missing parameter)")
            return
            
        # b(z) = alpha * (1 + z)**2 + beta
        alpha, beta = tools.bias(1, tracer=tracer[0][:3], return_params=True)  
        factor = (alpha * (1 + zeff[1])**2 + beta) / (alpha * (1 + zeff[0])**2 + beta)
        likelihood.all_params[target_param_name].update(derived='{' + source_param_name + '}' + f' * {factor}')
        logger.debug(f"Derived relationship: {target_param_name} = {source_param_name} * {factor}")

    tracers = tracer.split('x')

    # Auto-correlation: link ell2 bias / damping to ell0.
    if tracers[0] == tracers[1]:
        if len(zeffs[tracer]) > 1:
            zeff = [zeffs[tracer][ell] for ell in [0, 2]]
            _rescale_bias_params(likelihood, tracer=[f"{tracers[0]}_ell0", f"{tracers[0]}_ell2"], zeff=zeff)
            # logger.warning('we neglect the redshift dependence of the damping term, for now') 
            param_name_ell0, param_name_ell2 = f"{tracers[0]}_ell0.sigmas",f"{tracers[0]}_ell2.sigmas"
            likelihood.all_params[param_name_ell2].update(derived='{' + param_name_ell0 + '}')
            logger.debug(f"Derived damping: {param_name_ell2} = {param_name_ell0}")

    # Cross-correlation: 
    # if derived_cross_bias is False: let free only one cross-correlation bias and fix the other one to break degeneracy.
    # if derived_cross_bias is True: By default, link the bias of the cross-correlation to their auto-correlation counterparts. 
    #                                If the auto-correlation is not available in available_tracers, use one of the cross-correlation bias as default to link #                                all the others one to him.
    if tracers[0] != tracers[1]:
        cross_suffix = f'_{nickname}' if nickname is not None else ''
        for i, tt in enumerate(tracers):
            if derived_cross_bias and (available_tracers is not None):
                # if auto-tracer is available in available_tracers:
                if 'x'.join([tt, tt]) in available_tracers:
                    # derived the bias from the auto-correlation bias, taking into account the different effective redshifts of the auto and cross correlation.
                    zeff = [zeffs['x'.join([tt, tt])][0], zeffs[tracer][0]]
                    _rescale_bias_params(likelihood, tracer=[f"{tt}_ell0", f'{tt}{cross_suffix}_ell0'], zeff=zeff)
                else:
                    # determine default tracer to link the bias parameters, if auto-correlation data is not available.
                    default_tracer = sorted([tracer for tracer in available_tracers if tt in tracer.split('x')])[0]
                    if tracer == default_tracer:
                        logger.debug(f'This parameter is free ({tt}, {tracer}), and it will be used as default to link the other cross-correlation bias parameters.')
                    else:
                        logger.debug(f'This parameter is free ({tt}, {tracer}), but it will be linked to {default_tracer} bias parameters to break degeneracy, since auto-tracer data for {tt} is not available.')
                        zeff = [zeffs[default_tracer][0], zeffs[tracer][0]]
                        _rescale_bias_params(likelihood, tracer=[f"{tt}_{default_tracer}_ell0", f'{tt}{cross_suffix}_ell0'], zeff=zeff)

            else:
                # let free the cross-correlation bias, but fix one of the two biases to break degeneracy.
                # the first linear bias parameter can be set with kwargs.          
                if i == 0:      
                    default_b1 = kwargs.get(f"{tt}{cross_suffix}_ell0.b1", 1)
                    likelihood.all_params[f"{tt}{cross_suffix}_ell0.b1"].update(value=default_b1, fixed=True) 
    
            if len(zeffs[tracer]) > 1:
                zeff = [zeffs[tracer][ell] for ell in [0, 2]]
                _rescale_bias_params(likelihood, tracer=[f"{tt}{cross_suffix}_ell0", f"{tt}{cross_suffix}_ell2"], zeff=zeff)
                # logger.warning('we neglect the redshift dependence of the damping term, for now')
                # Note: the first damping term is fixed to 0:
                if i == 1: likelihood.all_params[f"{tt}{cross_suffix}_ell2.sigmas"].update(derived='{' + f"{tt}{cross_suffix}_ell0.sigmas" + '}')



def get_observable_and_likelihood(pk, window, cov, tracer='LRG', zeffs={'LRGxLRG': {0: 0.7, 2: 0.7}}, p={'LRG': 1., 'ELG': 1., 'QSO': 1.4}, fix_fnl=False, engine='class', scale_covariance=1, nickname=None, **kwargs):
    """
    Get the observable and likelihood for a given tracer. Each multipole is treated as a different observable, but they share the same parameters in the theory.

    Args:
        pk: lsstypes Mesh2SpectrumPoles observable containing the power spectrum multipoles.
        window: lsstypes WindowMatrix matches the power spectrum multipoles.
        cov: lsstypes CovarianceMatrix matches the power spectrum multipoles.
        tracer: name of the tracer used to define the parameters in the theory and avoid duplicates when combining the different observables.
        p (dict, optional): Value of p parameter. Defaults to {'LRG': 1., 'ELG': 1., 'QSO': 1.4}.
        fix_fnl (bool, optional): If true do not fit for $f_{\rm NL}^{\rm loc}$. Defaults to False.
        engine (str, optional): Solver for perturbation theory computation ('class', 'camb' or either that works in cosmoprimo). Defaults to 'class'.
        nickname (str, optional): Suffix inserted between the tracer shortname and '_ell{ell}' in cross-correlation theory
            parameter names for cross-correlations. Use this to avoid parameter name collisions when combining
            multiple cross-correlations that share a tracer (e.g. LRGxELG and ELGxQSO both have ELG).
            With nickname='LRGxELG', ELG parameters become 'ELG_LRGxELG_ell0.b1'. Default is None.
        
    Kwargs:
         kwargs[f"{tt}_{nickname}_ell0.b1"] (or f"{tt}_ell0.b1" if no nickname) value used to fix
         one of the two b1 in the cross-correlation, otherwise fixed to 1.

    Returns:
       observables: list of monopole and quadrupole (if used) desilike observables.
       likelihood: desilike likelihood to run profer or MCMC on.
    """
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from cosmoprimo.fiducial import DESI

    tracers = tuple(tracer.split('x'))
    if len(tracers) == 1: tracers *= 2
    tracers = (tracers[0][:3], tracers[1][:3])  # LRG_zcmb -> LRG, ELGnotqso -> ELG, ...
    cross_correlation = (tracers[0] != tracers[1])

    observables = []
    for ell in pk.ells:  
        cross_suffix = f'_{nickname}' if (nickname is not None and cross_correlation) else ''
        if cross_correlation: 
            tracers_theo = [f'{tt}{cross_suffix}_ell{ell}' for tt in tracers]
        else:
            tracers_theo = [f'{tt}_ell{ell}' for tt in tracers]
        if not cross_correlation: tracers_theo = tracers_theo[:1]
        logger.info(f'{tracers_theo=}, {ell=}, zeff={zeffs["x".join(tracers)][ell]:2.4}')

        # extract only the mulitpole ell: 
        data = pk.get(ells=[ell])
        wmatrix = window.at.observable.match(data)

        # Define Template and Theory:
        template = FixedPowerSpectrumTemplate(z=zeffs["x".join(tracers)][ell], fiducial=DESI(engine=engine))
        theory = PNGTracerPowerSpectrumMultipoles(template=template, mode="b-p", tracers=tracers_theo)
        # Fix some parameters:
        theory.params['fnl_loc'].update(value=0.0, fixed=fix_fnl)
        if ell == 2: 
            theory.params[f"{'x'.join(tracers_theo)}.sn0"].update(value=0, fixed=True)
        for tracer in tracers_theo:
            theory.params[f'{tracer}.p'].update(value=p[tracer.split('_')[0]], fixed=True)
        # Use only one damping term for the cross-correlation
        if cross_correlation: 
            theory.params[f"{tracers_theo[0]}.sigmas"].update(value=0, fixed=True)

        # Don't forget to give different name for the observable in order to stack them together in the likelihood:
        name = 'pk_' + 'x'.join(tracers) + f'_ell{ell}'
        observables += [TracerPowerSpectrumMultipolesObservable(name=name, data=data, window=wmatrix, theory=theory)] 

    if isinstance(cov, list):
        import lsstypes
        logger.info('Using mocks to estimate the covariance matrix.')
        covariance = lsstypes.cov([mock.match(pk) for mock in cov]).value()
        correction_covariance = {'correction': 'hartlap-percival2014', 'nobs': len(cov)}
    else:
        logger.info('Using analytical covariance matrix.')
        covariance = cov.at.observable.get(tracers=tracers).value()
        correction_covariance = None

    likelihood = ObservablesGaussianLikelihood(observables=observables, covariance=covariance, 
                                               correct_covariance=correction_covariance, scale_covariance=scale_covariance)
    fix_likelihood_bias_and_damping(likelihood, tracer='x'.join(tracers), zeffs=zeffs, derived_cross_bias=False, nickname=nickname, **kwargs)
    likelihood()

    return observables, likelihood


def combine_analytical_covariances(pks, covs, order=['LRGxLRG', 'LRGxQSO', 'QSOxQSO'], fiducial=None):
    """ 
    Combine the analytical covariance matrices for the auto power spectra and the cross power spectra into a single covariance matrix for the combined data vector.
    
    #'fiducial should be a dictionary containing the redshift ranges for each tracer, (see propose_fiducial() function)'

    Remarks: 
        * The off-diagonal blocks are estimated from the cross-correlation covariance matrices, and rescaled to account for the difference in effective volume between the auto and cross power spectra. 
        * The rescaling is mandatory to avoid non definite positive covariance matrices.
        * Effective volumes are pre-computed for specific redshift ranges and tracers, and are hard-coded in the function... -> Update if needed.
    """
    from lsstypes import ObservableTree

    veffs = {'LRG_z0.4-1.0': 9.41, 'LRG_z0.4-1.1': 11.201, 'LRG_z0.8-1.0': 4.075, 'LRG_z0.8-1.1': 5.866,
             'ELG_z0.8-1.0': 3.637, 'ELG_z0.8-1.1': 5.663, 'ELG_z0.8-1.6': 16.011,
             'QSO_z0.8-1.0': 1.129, 'QSO_z0.8-1.1': 1.831, 'QSO_z0.8-1.6': 6.494, 'QSO_z0.8-3.5': 14.722, 
             'LRGxQSO_z0.8-1.0': 2.143, 'LRGxQSO_z0.8-1.1': 3.262,
             'ELGxLRG_z0.8-1.0': 3.849, 'ELGxLRG_z0.8-1.1': 5.751,
             'ELGxQSO_z0.8-1.6': 10.151}

    def _extract_offdiag_block(pks, covs, tracer1, tracer2):
        """ 
        Extract the off-diagonal block from the covariance matrices for tracer1 x tracer2 (for instance: LRGxLRG x LRGxQSO).
        We look for the block in all the covariance matrices available, and we match the observables to extract the correct block in the correct order ! 
        If the block is not found, we return a block of zeros.
        """
        block = np.zeros((pks[tracer1].size, pks[tracer2].size))
        for source_key, cov in covs.items(): 
            try:   
                local_tracers = [tuple(tracer1.split('x')), tuple(tracer2.split('x'))]
                local_obs = ObservableTree([pks[tracer1], pks[tracer2]], observables=['spectrum2', 'spectrum2'], tracers=local_tracers)
                block = cov.at.observable.get(observables=['spectrum2', 'spectrum2'], tracers=local_tracers).at.observable.match(local_obs)
                # extract only the off-diagonal block (always the lower-left block, because of the .match() order):
                block = block.value()[:pks[tracer1].size, pks[tracer1].size:]
                # If we get here, the block was found and matched successfully!
                logger.debug(f'Found block for {tracer1} x {tracer2} in covariance "{source_key}".')
            except Exception:
                pass
        return block
    
    def _get_rescale_factor(tracer1, tracer2):
        """ 
        Get the rescale factor for the off-diagonal block between tracer1 and tracer2 (for instance: LRGxLRG x LRGxQSO).
        Redshift range for the full-range and cross-range of tracer are determined from the fiducial dictionary, and the corresponding effective volumes are pre-computed in hardcoded veffs dictionary.
        The rescale factor is given by (V_eff_cross / V_eff1) * (V_eff_cross / V_eff2), where V_eff_cross is the effective volume of the cross-correlation, and V_eff1 and V_eff2 are the effective volumes of the auto-correlations for tracer1 and tracer2.

        Remark: 
            * if tracer1 = 'LRGxQSO' then V_eff_cross = V_eff1 -> no rescaling for this part. 
            * if three or more tracers are in tracer1 and tracer2, we raise a ValueError. We neglect this case because we do not have any estimation of the covariance for this case. 
        """
        tt11, tt12 = tracer1.split('x')
        tt21, tt22 = tracer2.split('x')

        if tt11 == tt12:
            zrange = fiducial[tt11]['zrange']
        else: 
            zrange = fiducial[tracer1]['zrange']
        veff1 = veffs[f'{tt11}_z{zrange[0]}-{zrange[1]}']

        if tt21 == tt22:
            zrange = fiducial[tt21]['zrange']
        else: 
            zrange = fiducial[tracer2]['zrange']
        veff2 = veffs[f'{tt21}_z{zrange[0]}-{zrange[1]}']

        # From which cross-correlation matrix the off-diagonal block is estimated:
        tracer_cross = 'x'.join(np.unique([tt11, tt12, tt21, tt22]))
        if len(tracer_cross.split('x')) > 2:
            raise ValueError(f'Unexpected tracer_cross: {tracer_cross} from tracer1: {tracer1} and tracer2: {tracer2}')
        zrange = fiducial[tracer_cross]['zrange']
        veff1_cross, veff2_cross = veffs[f'{tracer_cross}_z{zrange[0]}-{zrange[1]}'], veffs[f'{tracer_cross}_z{zrange[0]}-{zrange[1]}']
    
        return veff1_cross / veff1 * veff2_cross / veff2
    
    missing_pks = [label for label in order if label not in pks]
    if missing_pks:
        raise ValueError(f'Missing pk blocks for {missing_pks}. Available: {list(pks.keys())}')

    observable_tot = ObservableTree([pks[label] for label in order], observables=['spectrum2'] * len(order), tracers=order)

    matrix = np.zeros((observable_tot.size, observable_tot.size))
    offsets = np.concatenate(([0], np.cumsum([observable_tot.get(tracers=order[i]).size for i in range(len(order))])))
    missing_offdiag = []

    for i in range(len(order)):
        for j in range(len(order)):
            # print(f'  Pair (i, j) = ({i}, {j}): {order[i]} x {order[j]}:', offsets[i], offsets[i + 1], offsets[j], offsets[j + 1])
            if i == j:
                block = covs[order[i]].at.observable.get(observables='spectrum2', tracers=tuple(order[i].split('x')))
                block = block.at.observable.match(observable_tot.get(observables='spectrum2', tracers=order[i])).value()
            else:
                block = _extract_offdiag_block(pks, covs, order[i], order[j])

                # Do not forget to rescale the off-diagonal block to account for the difference in effective volume between the auto and cross power. 
                # This is mandatory because we use a larger volume for the auto while the cross block is estimated from the commom volume.
                try: 
                    rescale_factor = _get_rescale_factor(order[i], order[j])
                except ValueError:
                    rescale_factor = 0
                logger.debug(f'Rescale factor for block {order[i]} x {order[j]}: {rescale_factor:.3f}')
                block = rescale_factor * block 

                if np.sum(block) == 0:
                    missing_offdiag.append(f'No block found or no rescaling available for {order[i]} x {order[j]}')
            
            # Insert the block into the matrix:
            matrix[offsets[i]:offsets[i + 1], offsets[j]:offsets[j + 1]] = block

    logger.debug('Missing off-diagonal blocks:', missing_offdiag)

    covariance_tot = covs[order[0]].clone(observable=observable_tot, value=matrix)
    logger.debug(covariance_tot.observable)
    try: 
        np.linalg.cholesky(covariance_tot.value())
    except Exception as e:
        logger.warning('Covariance matrix is not positive definite:', e)
        
    return covariance_tot


def build_total_likelihood(order, pks, observables, covs, zeffs, fiducial, scale_covariance=1):
    """ 
    Build the total likelihood for the combined data vector, by stacking the observables and using the combined covariance matrix.
    Were are using order to stack the observable in the same order as the covariance matrix. 
    Note: pks, observables, covs, zeffs, fiducial are dictionaries with keys corresponding to the labels in order.
    Note2: fiducial is used only to extract the redshift ranges for the different tracers, which are needed to rescale the off-diagonal blocks of the covariance matrix in combine_analytical_covariances.
    """
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from tools import combine_analytical_covariances, fix_likelihood_bias_and_damping

    total_observables = [observable for tracer in order for observable in observables[tracer]]
    logger.debug([ff.name for ff in total_observables])

    if isinstance(covs[order[0]], list):
        logger.info('Using mocks to estimate the covariance matrix.')
        covariance = np.cov(np.transpose([np.concatenate([covs[tt][i].match(pks[tt]).value() for tt in order]) for i in range(len(covs[order[0]]))]))
        correction_covariance = {'correction': 'hartlap-percival2014', 'nobs': len(covs[order[0]])}
    else:
        logger.info('Using analytical covariance matrix.')
        covariance = combine_analytical_covariances(pks, covs, order=order, fiducial=fiducial).value()
        correction_covariance = None
    
    total_likelihood = ObservablesGaussianLikelihood(observables=total_observables, covariance=covariance, 
                                                     correct_covariance=correction_covariance,scale_covariance=scale_covariance)
    for tracer in order: 
        # We do not link the damping term from the cross-correlation and the auto-correlation
        # Because they are different effective redshifts and we do not know the a priori.
        fix_likelihood_bias_and_damping(total_likelihood, tracer=tracer, zeffs=zeffs, derived_cross_bias=True, nickname=tracer, available_tracers=order)
    total_likelihood()

    return total_likelihood


def run_profiler(likelihood, fn_output=None, sigfigs=2):
    """ 
    Run the iminuit profiler on the likelihood, the results are saved in a text file if output_name is provided. fn_output should be a .txt file. 
    """
    from desilike.profilers import MinuitProfiler

    profiler = MinuitProfiler(likelihood, seed=7)
    profiler.maximize(niterations=10)
    logger.info(f'\n{profiler.profiles.to_stats(tablefmt="pretty")}')

    if fn_output is not None:
        to_save = profiler.profiles.to_stats(tablefmt='list', sigfigs=sigfigs, params=profiler.profiles.choice().bestfit.params())[0]
        np.save(fn_output, to_save)

        # for latex table:
        #_ = profiler.profiles.to_stats(fn=fn_output)
        #np.savetxt(fn_output.replace('.txt', '_list.txt'), profiler.profiles.to_stats(tablefmt='list')[0], fmt='%s')

    return profiler


def run_mcmc(likelihood, fn_output='tmp/mcmc_output_*.npy', extend_chains=False, nchains=1, max_iterations=1e5, check_every=1000):
    """Run the MCMC sampler on the likelihood, the results are saved in a text file if fn_output is provided. 

    Args:
        likelihood: Desilike Likelihood object to run the MCMC on.
        fn_output (str, optional): Where the chains will be saved (need to have *). Defaults to 'tmp/mcmc_output_*.npy'.
        extend_chains (bool, optional): If True, it will extend the existing chains (saved in fn_output) by running new iterations. Defaults to False.
        nchains (int, optional): Number of chains to run. Defaults to 1.
        max_iterations (int, optional): Maximum number of iterations to run. Defaults to 1e5.
        check_every (int, optional): How often to check the convergence + save the current state of the chains. Defaults to 1000.

    """
    from desilike.samplers import EmceeSampler
    chains = [fn_output.replace('*', f'{i}') for i in range(nchains)] if extend_chains else nchains

    sampler = EmceeSampler(likelihood, seed=31, chains=chains, save_fn=fn_output)  
    sampler.run(max_iterations=max_iterations, check_every=check_every)

    return sampler


def plot_observables(observables, ylims=None, show=True, fn_output=None):
    """ 
    Plot the observables (power spectrum multipoles) with their theory predictions and residuals.

    Parameters
    ----------
    observables : dict
        Mapping tracer -> observables.
    ylims : sequence, optional
        Y-axis limits for the monopole and quadrupole panels.
    profile : object, optional
        Profiler or profile-like object. If provided, an extra column is added to display
        its summary table.
    """
    fig, axs = plt.subplots(2, 2, figsize=(6, 4), sharex=True, sharey=False, gridspec_kw={'height_ratios': (3, 1)}, squeeze=True)
    fig.subplots_adjust(hspace=0.1)

    translator = {'LRGxLRG': 'L', 'ELGxELG': 'E', 'QSOxQSO': 'Q', 'LRGxELG': 'LxE', 'LRGxQSO': 'LxQ', 'ELGxQSO': 'ExQ'}

    for tracer in observables.keys():
        for obs in observables[tracer]:
            j = 1 if 'ell2' in obs.name else 0

            wtheory = obs.data.clone(value=obs.flattheory)
           
            data_pole = obs.data.get()
            wtheory_pole = wtheory.get()
            x = data_pole.coords('k')
            std = obs.covariance.at.observable.get().std()

            scale = 1.
            axs[0, j].errorbar(x, scale * data_pole.value(), yerr=scale * std, linestyle='none', marker='o', markersize=4, label=rf'{translator.get(tracer)}')
            axs[0, j].loglog(x, scale * wtheory_pole.value(), ls='-', c='k')

            axs[1, j].plot(x, (data_pole.value() - wtheory_pole.value()) / std)
            axs[1, j].set_ylim(-4, 4)
            for offset in [-2., 2.]: axs[1, j].axhline(offset, color='k', linestyle='--')
            for offset in [-1., 1.]: axs[1, j].axhline(offset, color='lightgray', linestyle=':')

    if ylims is not None:
        axs[0, 0].set_ylim(*ylims[0])
        axs[0, 1].set_ylim(*ylims[1])
    else:
        axs[0, 0].set_ylim(1e4, 8e4)
        axs[0, 1].set_ylim(2e3, 5e4)

    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[0, 0].set_title(r'$\ell = 0$', fontsize=10)
    axs[0, 1].set_title(r'$\ell = 2$', fontsize=10)
    axs[0, 0].set_ylabel(r'$P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{3}$]')
    axs[1, 0].set_ylabel(r'$\Delta P_{\ell} / \sigma (P_{\ell})$')
    axs[1, 0].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
    axs[1, 1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')

    plt.tight_layout()
    if fn_output is not None: plt.savefig(fn_output)
    if show:
        plt.show()
    else:
        plt.close()


def get_getdist_plotter(fig_width_inch=5, fontsize=14, legend_fontsize=12, axes_labelsize=12, axes_fontsize=14, line_lables=True):
    """Wrapper around getdist to get a plotter with the desired settings."""
    from getdist import plots as gdplt

    plotter = gdplt.get_subplot_plotter()
    plotter.settings.fig_width_inch = fig_width_inch
    plotter.settings.fontsize = fontsize
    plotter.settings.legend_fontsize = legend_fontsize
    plotter.settings.axes_labelsize = axes_labelsize
    plotter.settings.axes_fontsize = axes_fontsize
    plotter.settings.line_labels = line_lables
    plotter.settings.alpha_filled_add = 0.8
    plotter.settings.legend_loc = 'upper right'
    plotter.settings.figure_legend_ncol = 1
    plotter.settings.legend_colored_text = False

    return plotter


def plot_triangle(chains, params, legend_labels=None, xlabels=[r'$f_{\rm NL}^{\rm loc}$', r'$b_1$', r'$s_{n,0}$', r'$\Sigma_s$'], 
                  contour_colors=None, filled=True, contour_ls=None, g=None, fn_output=None, return_fig=False):
    """ Wrapper around getdist's plot_triangle to plot the contours of the different chains."""
    from desilike.samples import plotting
    
    if g is None: g = get_getdist_plotter()
    plotting.plot_triangle(chains, params, legend_labels=legend_labels, 
                           contour_colors=contour_colors, filled=filled, contour_ls=contour_ls, 
                           g=g, show=False, fn=None)

    if xlabels is not None:
        for i in range(len(xlabels)):
            g.subplots[len(xlabels)-1, i].set_xlabel(xlabels[i])
            if i > 0:
                g.subplots[i, 0].set_ylabel(xlabels[i])

    if fn_output is not None: plt.savefig(fn_output)
    
    if return_fig:
        return g.fig
    else:
        plt.show()


def run_profiling_one_mock(mocks, windows, covs, tracer, imock=0, kmin=1e-3, analytical_covariance=True, 
                           force_profiling=False, base_dir=None, fiducial=None, extra_fn='', return_profiler=False):
    """Run the profiler on a single mock realisation and save the result to disk.

    Parameters
    ----------
    mocks : list
        List of mock power spectrum observables.
    window : lsstypes WindowMatrix
        Window matrix for the tracer.
    cov : lsstypes CovarianceMatrix
        Analytical covariance matrix for the tracer.
    tracer : str
        Tracer name (e.g. 'LRGxLRG').
    imock : int, optional
        Index of the mock to fit. Default is 0.
    kmin : float, optional
        Minimum k value for the fit. Default is 1e-3.
    analytical_covariance : bool, optional
        If True, use the analytical covariance. If False, use mock covariance. Default is True.
    force_profiling : bool, optional
        If True, rerun even if output file already exists. Default is False.
    base_dir : str, optional
        Base directory. Profile outputs are written under the corresponding profiles directory.
    """
    import os

    kwargs = {'LRG_LRGxQSO_ell0.b1': 2.15, 'LRG_LRGxELG_ell0.b1': 2.15, 'ELG_ELGxQSO_ell0.b1': 1.2, 'scale_covariance': 1}

    fn_profile = base_dir + f"mock{imock}/bestfit_{tracer}_{'analytical_cov' if analytical_covariance else 'mock_cov'}_kmin-{kmin}{extra_fn}.npy"
    if (os.path.isfile(fn_profile) and force_profiling) or (not os.path.isfile(fn_profile)):
        os.makedirs(os.path.dirname(fn_profile), exist_ok=True)
        
        tracers = tracer.split('-')

        obs, lik, zeffs = {}, {}, {}
        pks = {}
        for tt in tracers:
            mocks_tt = mocks[tt].copy()  # Avoid modifying the original mock observable.

            pk = mocks_tt.pop(imock).select(k=(kmin, 1))  # remove the used mock from the covariance matrix.
            pks[tt] = pk
            
            window = windows[tt].at.observable.match(pk)

            zeffs[tt] = {ell: window.observable.get(ell).attrs['zeff'] for ell in pk.ells}  # Keep only the zeff for the used multipoles.

            if analytical_covariance:
                covariance = covs[tt].at.observable.at(observables='spectrum2', tracers=tuple(tt.split("x"))).match(pk)
            else:
                # print(len(mocks_tt))
                covariance = [mm.match(pk) for mm in mocks_tt]

            obs[tt], lik[tt] = get_observable_and_likelihood(pk, window, covariance, tt, zeffs, engine='camb', fix_fnl=False, nickname=tt, **kwargs)

        if len(tracers) > 1:
            lik = build_total_likelihood(tracers, pks, obs, covs if analytical_covariance else mocks, zeffs, fiducial)
        else:
            obs, lik = obs[tracers[0]], lik[tracers[0]]

        profiler = run_profiler(lik, fn_output=fn_profile, sigfigs=5)

        if (kmin == 1e-3) and analytical_covariance and (len(tracers) == 1):
            ylims = [(2e3, 4e4), (2e3, 4e4)] if tracer in ['ELGxELG', 'ELGxQSO'] else None
            fn_obs = base_dir + f"mock{imock}/bestfit_{tracer}_analytical_cov.png"
            plot_observables({tracer: obs}, ylims=ylims, fn_output=fn_obs, show=False)

        if return_profiler:
            return profiler