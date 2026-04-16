"""
    Compute the integral constraint contribution to the window matrix based on mocks.
    Author: Alberto Rosado-Marin (Ohio University) / Edmond Chaussidon (LBL)
"""

import numpy as np
import matplotlib.pyplot as plt

class LeastSquares:
    """
    Generic least-squares cost function with error.
    """
    from iminuit import Minuit

    errordef = Minuit.LEAST_SQUARES  # for Minuit to compute errors correctly

    def __init__(self, model, x, y, inv_cov):
        self.model = model  # model predicts y for given x
        self.x = x
        self.y = y
        self.inv_cov = inv_cov

    def __call__(self, *par):  # we must accept a variable number of model parameters
        ym = self.model(self.x, *par)
        diff = self.y + ym
        return np.trace(np.dot(diff, np.dot(self.inv_cov, diff.T)))


class WindowIC:
    """
    Include the Integral Constraint contribution to the window matrix by fitting a model to the difference between mocks with and without IC, and then building a new window matrix with the fitted IC contribution. It follows the method described in Chaussidon et al. 2024 (https://arxiv.org/abs/2411.17623).
    """
    def __init__(self, wmatrix, mocks_noic, mocks_ic, mocks_cov, ellsin=[0,2]):
        self.ells = mocks_noic[0].ells

        # Work only with on the ellsin (for instance if we want to build the IC contribution only from ellin=0,ellin=2 and neglect the quadrupole contribution):
        self.ellsin = ellsin

        self.kmin_out, self.kmax_out = 1e-3, 1e-1
        self.pk_obs = mocks_noic[0].select(k=(self.kmin_out,self.kmax_out))

        self.wmatrix = wmatrix

        self.set_power(mocks_noic, mocks_ic)
        self.set_wmatrix(wmatrix)
        self.set_covariance(mocks_cov)

    def set_power(self, mocks_noic, mocks_ic):
        self.power_noic = np.asarray([pk.match(self.pk_obs).get(self.ells).value() for pk in mocks_noic])
        shotnoise = [np.concatenate([pk.match(self.pk_obs).get(ell).values('shotnoise') for ell in self.ells]) for pk in mocks_noic]
        self.power_noic_w_sn = self.power_noic + shotnoise
        self.power_ic = np.asarray([pk.match(self.pk_obs).get(self.ells).value() for pk in mocks_ic])

    def set_wmatrix(self, wmatrix):
        # Sub select the theory (ie) the input of the window matrix:
        wmatrix = wmatrix.at.theory.select(k=(1.7e-3, 3e-1))
        # Match the observable pk (ie) k bins and observed ells.
        wmatrix = wmatrix.at.observable.match(self.pk_obs)
        # wmatrix = wmatrix.at.theory.select(k=slice(0, None, 2)) # One could rebin the entry.
        self.kin, self.kout = wmatrix.theory.get(ells=0).k, wmatrix.observable.get(ells=0).k
        W = wmatrix.value()
        self.W = W[:self.kout.size*len(self.ells), :self.kin.size*len(self.ellsin)]

    def set_covariance(self, mocks_cov):
        poles = np.asarray([pk.match(self.pk_obs).get(self.ells).value() for pk in mocks_cov])
        self.C = np.cov(poles.T)
        self.C_inv = np.linalg.inv(self.C)

    def f(self, params, kin=None, kout=None):
        A = params[:len(self.ells)*len(self.ellsin)]
        sigma = params[len(self.ells)*len(self.ellsin):]

        return np.block([[A[i + len(self.ells)*j] * np.exp(-((kout**2)[None, :] + (kin**2)[:, None]) / sigma[i + len(self.ells)*j]**2) for i in range(len(self.ells))] for j in range(len(self.ellsin))])

    def model(self, P, *params):
        return np.dot(self.W, np.dot(self.f(params, kin=self.kin, kout=self.kout), P.T)).T 
        
    def fit(self, ncall=5, initial_params=None, hard_limits=False):
        """ 
        Fit the model to the difference between mocks with and without IC, using a least-squares cost function and Minuit as the minimizer. The best-fit parameters are stored in self.bestfit_params.
        """
        from iminuit import Minuit
        from tqdm import tqdm

        lsq = LeastSquares(self.model, self.power_noic_w_sn, (self.power_ic - self.power_noic), self.C_inv)
        
        name_params = [f"x{i}" for i in range(0, 2*len(self.ells)*len(self.ellsin))]

        params = [1e-4]+[0 for i in range(len(self.ells)*len(self.ellsin)-1)] + [1 for i in range(len(self.ells)*len(self.ellsin))]
        params = initial_params if initial_params is not None else params

        print(f"ncall={ncall}, initial_params={params}")
        m = Minuit(lsq, name=name_params, **{name: params[i] for i, name in enumerate(name_params)})
        if hard_limits:
            # Add hard limit for the A parameters to avoid unphysical values (for instance, if A is too large, the IC contribution can be larger than the signal itself and lead to numerical issues in the minimization). The sigma parameters are left free to explore a wide range of values.
            for i, name in enumerate(name_params):
                m.limits[name] = (-1, 1) if i <  len(self.ells)*len(self.ellsin) else (None, None)
        #m.errordef =  errordef
        for _ in tqdm(range(ncall)):
            m.migrad()
        print("Valid minimum?", m.valid)
        
        count, iterations = 0, 3
        while not m.valid:
            print(f'Entering while loop because of INVALID minimum [{count + 1} / {iterations}]')
            m.migrad()
            count += 1
            if count >= iterations:
                break     
        print("Valid minimum?", m.valid)

        print(m.fmin)

        self.bestfit_params = [m.values[name] for name in name_params]
        return m
    
    def plot_validation(self, figsize=(6, 2.7), title='', save_fn=None, ylim=None, params=None):
        kk = self.kout
        pkdiff = (self.power_ic - self.power_noic)
        nmocks = self.power_ic.shape[0]

        params = self.bestfit_params if params is None else params
        plt.figure(figsize=figsize)
        for ill, _ in enumerate(self.ells):
            plt.subplot(1, len(self.ells), 1+ill)

            plt.fill_between(kk, pkdiff[:,ill*kk.size:(ill+1)*kk.size].mean(axis=0) - pkdiff[:,ill*kk.size:(ill+1)*kk.size].std(axis=0), 
                                pkdiff[:,ill*kk.size:(ill+1)*kk.size].mean(axis=0) + pkdiff[:,ill*kk.size:(ill+1)*kk.size].std(axis=0), 
                                color='lightgray', alpha=0.6, label=r'$\pm~\sigma~\Delta P$' if ill == 0 else None)
            
            plt.plot(kk, pkdiff[:,ill*kk.size:(ill+1)*kk.size].mean(axis=0), c='dodgerblue', ls='-', label=f'mean over\n{nmocks} mocks' if ill == 0 else None)
            plt.plot(kk, - self.model(self.power_noic_w_sn, *params)[:,ill*kk.size:(ill+1)*kk.size].mean(axis=0), c='orangered', label='model' if ill == 0 else None)

            if ylim is not None: plt.ylim(ylim)            
            if ill != 0: plt.yticks([])
            plt.xscale('log')
            if ill == 0: plt.legend(title=title, loc='lower right')
            plt.xlabel(r"$k$ $[h \mathrm{ Mpc}^{-1}]$")
            if ill == 0: plt.ylabel(r"$\left(P_{\ell}^{\mathrm{no~IC}} - P_{\ell}^{\mathrm{IC}}\right)(k)$")
        plt.tight_layout()
        if save_fn is not None: plt.savefig(save_fn, bbox_inches='tight', pad_inches=0.1, dpi=200, facecolor='white')
        plt.show()

    def build_wmatrix_with_ic(self, save_fn=None):  
        kkin, kkout = self.wmatrix.theory.get(ells=0).k, self.wmatrix.observable.get(ells=0).k

        nout = kkout.size * len(self.ells)
        nin = kkin.size * len(self.ellsin)

        # Work only with on the ellsin (for instance if we want to build the IC contribution only from ellin=0,ellin=2 and neglect the quadrupole contribution):
        WW = self.wmatrix.value()[:nout, :nin]
        WW = np.ma.array(WW, mask=np.isnan(WW))

        f = self.f(self.bestfit_params, kin=kkin, kout=kkout)
        f_mask = np.ma.array(f, mask=np.isnan(f))

        WW_ic = np.zeros_like(self.wmatrix.value())
        WW_ic[:nout, :nin] = WW - np.ma.dot(WW, np.ma.dot(f_mask, WW)).data

        wmatrix_ic = self.wmatrix.clone(value=WW_ic)
        if save_fn is not None: wmatrix_ic.write(save_fn)
        return wmatrix_ic