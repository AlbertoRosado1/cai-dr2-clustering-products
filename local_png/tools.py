import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger('PNG fitting tools')


def read_data(stats_dir='.', tracer='LRG', zrange=(0.4, 1.1), weight_type='default-fkp-oqe', region='GCcomb', add_ic=False, **kwargs):
    """ 
    Read the data from the clustering statistics output. This is a wrapper of clustering_statistics.tools.get_stats_fn.
    """
    import lsstypes
    from clustering_statistics import tools
    import functools 
    
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir, tracer=tracer, zrange=zrange, weight=weight_type, region=region)

    # Read the data:
    pk = lsstypes.read(get_stats_fn(kind='mesh2_spectrum'))
    if add_ic:
        window = lsstypes.read(get_stats_fn(kind='window_mesh2_spectrum', extra='with_ic'))
    else:
        window = lsstypes.read(get_stats_fn(kind='window_mesh2_spectrum'))
    cov = lsstypes.read(get_stats_fn(kind='covariance_mesh2_spectrum'))

    return pk, window, cov


def rebin_data(pk, window, cov, tracer='LRG', kmin=1e-3, kmax=0.08, kpivot=2e-2, nrebin=2, use_ell2=True, **kwargs):
    """ 
    Rebin the data with k > kpivot by a factor nrebin. The quadrupole is rebinned again by a factor nrebin for the full range.
    Then, select data in the k range [kmin, kmax]. If use_ell2 is False, we only keep the monopole.
    Finally, we match the size of the window and covariance to the size of the power spectrum.
    Return the rebinned power spectrum, window and covariance.
    """
    # Let's rebin the power spectrum : 
    # print(f'Original k shape (ell=0): {pk.get(0).k.shape[0]}')
    pk = pk.map(lambda pole: pole.at(k=(kpivot, 1.)).select(k=slice(0, None, nrebin)))
    pk = pk.at(2).select(k=slice(0, None, nrebin))  # Rebin the quadrupole again but for the full range.
    # kpivot, nrebin = 4e-2, 2
    # pk = pk.map(lambda pole: pole.at(k=(kpivot, 1.)).select(k=slice(0, None, nrebin)))

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
    
    logger.debug(f'After rebinning and k range selection: {pk.get(0).k.shape[0]} and {pk.get(2).k.shape[0] if use_ell2 else "Not used"} data points.')

    return pk, window, cov


def fix_likelihood_bias_and_damping(likelihood, tracer, zeffs, derived_cross_bias=True, **kwargs):
    """Apply bias and damping parameter relations between the paramters of the likelihood both for ell=0/2 and auto/cross power spectrum.

    Parameters
    ----------
    likelihood : desilike likelihood
        Likelihood whose parameters are updated in place.
    tracer : str
        name of the tracers: 'LRGxLRG', 'LRGxQSO', ... 

    Returns
    -------
    likelihood
        Same likelihood object, updated in place.
    """
    from clustering_statistics import tools

    def _rescale_bias_params(likelihood, tracer, zeff):
        """ 
        Fix the bias parameters in the likelihood according to the redshift dependence of the bias.

        Args:
            likelihood: The likelihood object.
            tracer: The tracer for which to fix the bias parameters.
            zeff: The effective redshifts.
        """
        from clustering_statistics import tools        
        # b(z) = alpha * (1 + z)**2 + beta
        alpha, beta = tools.bias(1, tracer=tracer[0][:3], return_params=True)  
        factor = (alpha * (1 + zeff[1])**2 + beta) / (alpha * (1 + zeff[0])**2 + beta)
        likelihood.all_params[f"{tracer[1]}.b1"].update(derived='{' + f"{tracer[0]}.b1" + '}' + f' * {factor}')

    tracers = tracer.split('x')

    # Auto-correlation: link ell2 bias / damping to ell0.
    if tracers[0] == tracers[1]:
        if len(zeffs[tracer]) > 1:
            zeff = [zeffs[tracer][ell] for ell in [0, 2]]
            _rescale_bias_params(likelihood, tracer=[f"{tracers[0]}_ell0", f"{tracers[0]}_ell2"], zeff=zeff)
            # logger.warning('we neglect the redshift dependence of the damping term, for now') 
            likelihood.all_params[f"{tracers[0]}_ell2.sigmas"].update(derived='{' + f"{tracers[0]}_ell0.sigmas" + '}')

    # Cross-correlation: derive cross biases from auto biases or let it free.
    if tracers[0] != tracers[1]:
        for i, tt in enumerate(tracers):
            if derived_cross_bias:
                # derived the bias from the auto-correlation bias, taking into account the different effective redshifts of the auto and cross correlation.
                zeff = [zeffs['x'.join([tt, tt])][0], zeffs[tracer][0]]
                _rescale_bias_params(likelihood, tracer=[f"{tt}_ell0", f'{tt}_cross_ell0'], zeff=zeff)
            else:
                # let free the cross-correlation bias, but fix one of the two biases to break degeneracy.
                # the first linear bias parameter can be set with kwargs.          
                if i == 0:      
                    default_b1 = kwargs.get(f"{tt}_cross_ell0.b1", 1)
                    likelihood.all_params[f"{tt}_cross_ell0.b1"].update(value=default_b1, fixed=True)    

            if len(zeffs[tracer]) > 1:
                zeff = [zeffs[tracer][ell] for ell in [0, 2]]
                _rescale_bias_params(likelihood, tracer=[f"{tt}_cross_ell0", f"{tt}_cross_ell2"], zeff=zeff)
                # logger.warning('we neglect the redshift dependence of the damping term, for now')
                # Note: the first damping term is fixed to 0:
                if i == 1: likelihood.all_params[f"{tt}_cross_ell2.sigmas"].update(derived='{' + f"{tt}_cross_ell0.sigmas" + '}')


def get_obervable_and_likelihood(pk, window, cov, tracer, zeffs, p={'LRG': 1., 'ELG': 1., 'QSO': 1.4}, fix_fnl=False, engine='class', **kwargs):
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
        
    Kwargs:
         kwargs[f"LRG_cross_ell0.b1"] value used to fix one of the two b1 in the cross-correlation otherwise fix it to 1 (the damping term is set to 0).

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
        if cross_correlation: 
            tracers_theo = [tracer + f'_cross_ell{ell}' for tracer in tracers]
        else:
            tracers_theo = [tracer + f'_ell{ell}' for tracer in tracers]
        if not cross_correlation: tracers_theo = tracers_theo[:1]
        logger.info(f'{tracers_theo=}, {ell=}, zeff={zeffs["x".join(tracers)][ell]:2.4}')

        # extract only the mulitpole ell: 
        data = pk.get(ells=[ell])
        wmatrix = window.at.observable.match(data)
        covariance = cov.at.observable.get(observables='spectrum2', tracers=tracers)  # for the cross-covariance
        covariance = covariance.at.observable.match(data)

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
        observables += [TracerPowerSpectrumMultipolesObservable(name=name, data=data, window=wmatrix, covariance=covariance, theory=theory)] 

    likelihood = ObservablesGaussianLikelihood(observables=observables, covariance=cov.at.observable.get(tracers=tracers).value(), scale_covariance=1)
    fix_likelihood_bias_and_damping(likelihood, tracer='x'.join(tracers), zeffs=zeffs, derived_cross_bias=False, **kwargs)
    likelihood()

    return observables, likelihood


def run_profiler(likelihood, fn_output=None):
    """ 
    Run the iminuit profiler on the likelihood, the results are saved in a text file if output_name is provided. fn_output should be a .txt file. 
    """
    from desilike.profilers import MinuitProfiler

    profiler = MinuitProfiler(likelihood, seed=7)
    profiler.maximize(niterations=20)
    logger.info(f'\n{profiler.profiles.to_stats(tablefmt="pretty")}')

    if fn_output is not None:
        _ = profiler.profiles.to_stats(fn=fn_output)
        np.savetxt(fn_output.replace('.txt', '_list.txt'), profiler.profiles.to_stats(tablefmt='list')[0], fmt='%s')

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


def plot_observables(observables):
    """ 
    Plot the observables (power spectrum multipoles) with their theory predictions and residuals. 
    """
    fig, axs = plt.subplots(2, 2,  figsize=(6, 4), sharex=True, sharey=False, gridspec_kw={'height_ratios': (3, 1)}, squeeze=True)
    fig.subplots_adjust(hspace=0.1)

    for tracer in observables.keys():
        for obs in observables[tracer]:
            j = 1 if 'ell2' in obs.name else 0

            wtheory = obs.data.clone(value=obs.flattheory)
           
            data_pole = obs.data.get()
            wtheory_pole = wtheory.get()
            x = data_pole.coords('k')
            std = obs.covariance.at.observable.get().std()

            scale = 1.
            axs[0, j].errorbar(x, scale * data_pole.value(), yerr=scale * std, linestyle='none', marker='o', markersize=4, label=rf'{tracer}')
            axs[0, j].loglog(x, scale * wtheory_pole.value(), ls='-', c='k')

            axs[1, j].plot(x, (data_pole.value() - wtheory_pole.value()) / std)
            axs[1, j].set_ylim(-4, 4)
            for offset in [-2., 2.]: axs[1, j].axhline(offset, color='k', linestyle='--')
            for offset in [-1., 1.]: axs[1, j].axhline(offset, color='lightgray', linestyle=':')

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
    plt.show()


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

    veffs = {'LRG_z0.4-1.1': 11.201, 'LRG_z0.8-1.1': 5.866, 
             'ELG_z0.8-1.1': 5.663, 'ELG_z0.8-1.6': 16.011, 
             'QSO_z0.8-1.1': 1.831, 'QSO_z0.8-1.6': 6.494, 'QSO_z0.8-3.5': 14.722}

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
        veff1_cross, veff2_cross = veffs[f'{tt11}_z{zrange[0]}-{zrange[1]}'], veffs[f'{tt21}_z{zrange[0]}-{zrange[1]}']
    
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
    try: 
        np.linalg.cholesky(covariance_tot.value())
    except Exception as e:
        logger.warning('Covariance matrix is not positive definite:', e)

    return covariance_tot
