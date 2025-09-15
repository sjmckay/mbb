### Class to implement a modified blackbody SED, with the capability to fit the SED with various options
# and plot a simple version of it, as well as save the results to a file. 
# Author: Stephen McKay, Spring 2023


# Imports

import matplotlib.pyplot as plt
import numpy as np

import emcee
import corner

import warnings

from copy import deepcopy

# from astropy.table import Table, QTable
# from astropy.io import fits
import astropy.units as u
import astropy.constants as con
from astropy.constants import c, k_B, h
from astropy.cosmology import FlatLambdaCDM, Cosmology
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.30) 

from functools import partial

import multiprocessing as mp
from multiprocessing import Pool, cpu_count

NCORES = cpu_count()-2

Tcmb0 = 2.75

NWALKERS = 180
NITER = 2000
NBURN = 300
STEPSIZE = 1e-7

CURRENT_Z = 0

LLO = 8
LHI = 1000



from .mbb_funcs import mbb_func, planckbb #mbb_fun_ot, mbb_fun_go, mbb_fun_go_pl, mbb_fun_ot_pl

class ModifiedBlackbody:
    """Class to represent a modified blackbody (or MBB).
    
    This class can be used to encapsulate a single MBB model, or to perform an SED fit to photometry. The results can be easily plotted \
        or updated as needed, and various parameters/statistics can be extracted. The models are based off of \
        `Casey et al. (2012) <https://doi.org/10.1111/j.1365-2966.2012.21455.x>`_.

    Args:
        L (float): log10 of luminosity in solar units. If fitting data, this will set the initial guess for the fit.
        T (float): dust temperature in K. If fitting data, this will set the initial guess for the fit.
        beta (float): dust emissivity spectral index. If fitting data, this will set the initial guess for the fit.
        z (float): Redshift of this galaxy.
        alpha (float, optional): mid-IR power-law slope.
        l0 (float,optional): opacity turnover wavelength in microns.
        opthin (bool): Whether or not the model should assume optically thin dust emission.
        pl (pool): Whether or not the model should include a MIR power law (as in Casey+ 2012)

    Note: By default, ModifiedBlackbody assumes a flat :math:`\Lambda` CDM cosmology with :math:`\Omega_m = 0.3` and :math:`\Omega_\Lambda = 0.7`.\
         If you wish to change this, the code allows you to set the ``cosmo`` attribute of the ModifiedBlackbody to an instance of \
         ``astropy.cosmology.Cosmology`` after it is created. 
    """

    def __init__(self, L, T, beta, z, alpha=2.0,l0=200.,opthin=True, pl=False):
        self.L = L
        self.T = T 
        self.beta = beta 
        self.z = z
        self._cosmo = cosmo
        self._pl = pl
        self._opthin=opthin
        self.alpha=alpha
        self.l0=l0
        self._model = self._select_model()
        self.N = 11
        Lcurr = np.log10(self.get_luminosity((8,1000)).value)
        while((Lcurr > (L+0.0001)) | (Lcurr < (L-0.0001))):
            self.N = self.N + 0.1*(L-Lcurr)
            Lcurr = np.log10(self.get_luminosity((8,1000)).value)
        self.L = np.round(Lcurr,2)
        self._to_vary = None
        self._fit_result = None
        self._phot=None
        self._priors = None
    

    #read-only attributes
    @property
    def pl(self):
        return self._pl
    
    @property
    def opthin(self):
        return self._opthin
    
    @property
    def model(self):
        return self._model
    
    @property
    def fit_result(self):
        return self._fit_result

    @property
    def dust_mass(self):
        return self._compute_dust_mass()
    
    @property 
    def phot(self):
        return self._phot

    @property 
    def priors(self):
        return self._priors
    
    @property
    def cosmo(self):
        return self._cosmo
    
    @cosmo.setter
    def cosmo(self, new_cosmo):
        if isinstance(new_cosmo, Cosmology):
            self._cosmo = new_cosmo
        else:
            raise ValueError(f"'new_cosmo' must be of type astropy.cosmology.Cosmology, got {type(new_cosmo)}")

    def fit(self, phot, nwalkers=400, niter=500, ncores = NCORES, stepsize=1e-7,params=['L','beta','T'],priors=None,restframe=False,pool=None):
        """Fit photometry

        Fit a modified blackbody to photometry.
        Updates the parameters of this MBB model to the resulting parameters of the fit, and populates the ``fit_result`` attribute \
            of the ModifiedBlackbody with the fit results. Note that the final values of the MBB model will be set to \
            the converged (maximum likelihood) value from the fit, but depending on the posterior distribution, you may prefer \
            to use the median of the posterior as the final values for the parameters; this can be obtained with the ``post_percentile()`` method.

        The ``fit_result`` attribute is a dictionary containing the following:
            - ``sampler``: an ``emcee.EnsembleSampler`` representing the chain of walker values from the fit.
            - ``chi2``: the raw chi-squared value at the end of the fitting process.
            - ``n_params``: the number of fitted parameters.
            - ``n_bands``: the number of bands in the fit (the length of ``phot``)

        Example:

        .. code-block:: python
        
           from mbb import ModifiedBlackbody as MBB
           m = MBB(L=12, beta=1.8, T=35, z=2.5, alpha=2.0, opthin=True, pl = True)
           phot = ([450, 850],[0.005, 0.0021],[0.0006,0.00032]) #wl, flux, error
           
           # fit for redshift
           result = m.fit(phot=phot,niter=100,params=['L','z'],
                          restframe=False, priors = {'z':dict(mu=2.5,sigma=1.0)}) 
           # could equivalently use 'm.fit_result' instead of 'result'

           reduc_chi2 = result['chi2']/(result['n_bands']-result['n_params']) # reduced chi-squared

        Args:
            phot (array-like): wavelengths and photometry, arranged as a 3 x N array (wavelength, flux, error). 
            Wavelengths should be given as rest-frame values.
            nwalkers (int): how many walkers should be used in the MCMC fit. 
            niter (int): how many iterations to run in the fit.
            ncores (int): how many CPU cores to use in multiprocessing. If ``pool`` is not ``None``, this is ignored. Set to 1 to not use multiprocessing. \
                Default is number of available CPUs - 2.
            stepsize (float): stepsize used to randomize the initial walker values. 
            params (list): list of parameter names, e.g., [``L``, ``beta``, ``T``, ``z``, ``alpha``, ``l0``] to vary in the fit. The rest will be fixed. \
                ``L`` should generally be included since it represents the normalization of the model. 
            priors (dict): Priors to use in the Bayesian fitting. This should be a dictionary with keys corresponding \
                to the elements of ``params``. For each key, the corresponding value can be either (1) a function that takes the value of the parameter and \
                returns a number between 0.0 and 1.0, or (2) a dictionary with keys 'mu' and 'sigma', in which case the code will use these to \
                generate a Gaussian prior on that parameter. If ``None``, or for any params not included in ``priors``, flat (uniform) priors will be assumed. \
                Currently, ``T`` is constrained to be between 5 K and 120 K, ``beta`` is between 0.1 and 5.0, and ``z`` is between 0.1 and 12.
            restframe (bool): whether wavelengths in ``phot`` are given in the rest frame (default is observed frame)
            pool (multiprocessing.pool.Pool): an optional pool to pass to the sampler for multiprocessing; otherwise fit() will generate one internally. \
                Can be faster to use an external Pool if many fits are being performed, also may be useful depending on the OS/kernel being used.
        """
        phot = np.asarray(phot).reshape(3,-1) # make sure x,y,yerr are in proper shape
        if restframe: self._phot = (phot[0],phot[1],phot[2]) # emcee takes args as a list
        else: self._phot = (phot[0]/(1.0+self.z),phot[1],phot[2])
        self._priors = priors

        # replace L with N (under the hood, we use N to normalize the model)
        if 'L' in params: 
            i = params.index('L')
            params[i] = 'N'

        initdict = self._fit_param_dict()
        try:
            init = [initdict[key] for key in params]
        except KeyError:
            raise ValueError(f'Varied parameters must be one of {list(initdict.keys())}')
        ndim=len(init)

        #set up initial parameters
        fixed = initdict.copy()
        for key in params: fixed.pop(key)
        self._to_vary = params
        p0 = [np.array(init) + stepsize * np.random.randn(ndim) for i in range(nwalkers)]

        #run the MCMC fit
        sampler = self._run_fit(p0=p0, nwalkers=nwalkers, niter=niter, ncores=ncores, lnprob=self._lnprob, 
            ndim=ndim, to_vary = params, fixed = fixed, pool=pool)
        self._fit_result = {'sampler':sampler}

        #get 16,50,84 percentiles of fitted parameters and update
        med_params = self._get_params_spread()
        updated = initdict
        for i, key in enumerate(params):
            updated[key] = med_params[1][i]
        self._update_N(**updated)

        #save chi2 and fitted parameter results in easy to access format
        yprime = self.eval(self._phot[0],z=0).value
        self._fit_result['chi2'] = np.nansum( (self._phot[1]-yprime)**2/self._phot[2]**2 )
        self._fit_result['n_params'] = ndim
        self._fit_result['n_bands'] = len(self._phot[0])
        return deepcopy(self._fit_result)

    def reset(self):
        '''Clear the fit results.
        
        Clear the ModifiedBlackbody fit results, priors, and photometry. Current values of parameters (``L``, ``beta``, etc) will remain unchanged.
        
        '''
        self._to_vary = None
        self._fit_result = None
        self._phot = None
        self._priors = None


    def _get_chain_for_parameter(self, param,sample_by=1):
        '''retrieve the chain of walker values for a given parameter that was varied in the fit'''
        if self.fit_result != None:
            if param == 'L' and 'N' in self._to_vary: 
                params = self._fit_param_dict()
                lirs = []
                chain = self.fit_result['sampler'].get_chain(flat=True)[::sample_by,:]            
                for i in range(len(chain)):
                    p = params.copy()
                    for j, key in enumerate(self._to_vary): p[key] = chain[i][j]
                    lirs.append(np.log10(self._integrate_mbb(**p,wllimits=(8,1000)).value))
                return np.asarray(lirs)
            elif param in self._to_vary:
                chain = self.fit_result['sampler'].get_chain(flat=True)[:,:]
                where = np.where(np.array(self._to_vary) == param)[0]
                return np.squeeze(chain[:,where])
            else:
                raise KeyError(f'Cannot get chain for "{param}" since it was not varied in the fit.')
        return None
    

    def post_percentile(self,param,q=[16,50,84],sample_by=1):
        '''Determine the posterior percentile values of a given fit parameter.

        Example: 

        .. code-block:: python
            
           m.posterior('beta', q = [16,50,84]) # get median and 16th--84th percentile interval
        
        Args:
            param (str): name of parameter, element of the 'params' argument passed to ModifiedBlackbody.fit()
            q (array-like of float): percentile or sequence of percentiles to compute from the posterior distribution
            sample_by (int): sample every ``sample_by`` values in the posterior chain, helps with speed (especially for ``L``, which must be computed).
            
        Returns:
            float or array: the percentiles of the posterior distribution for parameter 'param'
        '''
        if self.fit_result != None:
            try:
                chain = self._get_chain_for_parameter(param,sample_by=sample_by)
                return np.nanpercentile(chain, q=q)
            except Exception as e:
                raise Exception(f"Unable to get posterior for parameter {param}: failed with error '{e}'")
        else: raise AttributeError(f'No fit has been run yet, so no posterior for {param} exists.')


    def posterior(self,param,sample_by=1):
        '''Determine the posterior chain of a given fit parameter.

        Args:
            param (str): name of parameter, element of the 'params' argument passed to ModifiedBlackbody.fit()
            sample_by (int): sample every ``sample_by`` values in the posterior chain, helps with speed (especially for ``L``, which must be computed).
            
        Returns:
            array: the chain of values in the posterior distribution for parameter 'param'
        '''
        if self.fit_result != None:
            try:
                chain = self._get_chain_for_parameter(param,sample_by=sample_by)
                return chain
            except Exception as e:
                raise Exception(f"Unable to get posterior for parameter {param}: failed with error '{e}'")
        else: raise AttributeError(f'No fit has been run yet, so no posterior for {param} exists.')
            

    def update(self, L=None, T=None, beta=None,z=None,alpha=None,l0=None):
        """ update modified blackbody parameters (not the underlying model)."""
        self._update_N(T=T, beta=beta,z=z,alpha=alpha,l0=l0)
        if L: #update N and L consistently
            Lcurr = np.log10(self.get_luminosity((8,1000)).value)
            while((Lcurr > (L+0.0001)) | (Lcurr < (L-0.0001))):
                self.N = self.N * (L/Lcurr)
                Lcurr = np.log10(self.get_luminosity((8,1000)).value)
            self.L = np.round(Lcurr,2)


    def _update_N(self, N=None, T=None, beta=None,z=None,alpha=None,l0=None):
        """ update modified blackbody parameters (not model), using N rather than luminosity (used in fitting). """
        if N: self.N = N
        if beta: self.beta = beta
        if T: self.T=T
        if z: 
            if self._phot != None: # update rest-frame wavelengths
                self._phot = (self._phot[0]*(1.0+self.z)/(1.0+z),self._phot[1],self._phot[2])
            self.z = z
        if alpha: self.alpha = alpha
        if l0: self.l0 = l0
        self.L = np.log10(self.get_luminosity((8,1000)).value) #update L consistently with N


    def plot_sed(self, obs_frame=False,ax=None):
        """plot the rest-frame form of this mbb just for basic visualization. It is recommended 
        to use a separate, more detailed plotting function for figures.

        Args:
            obs_frame (bool): whether to plot against observed-frame wavelengths (default is rest frame).s
            ax (matplotlib.pyplot.Axes): axes to plot the model on. 
        Returns:
            matplotlib.pyplot.figure: the current figure
            matplotlib.axes.Axes: the current axes
        """
        if ax is None: fig, ax = plt.subplots(figsize=(5,4),dpi=120) 
        else: fig = ax.get_figure()
        x = np.logspace(1,4,500)
        if self.fit_result != None:
            nsamples = 200
            y, lb,ub = self._get_model_spread(x)
        else: y = self.eval(x)
        if obs_frame == True:
            x *= (1.+self.z)
            ax.set(xlabel = r'$\lambda$ observed-frame [$\mu$m]', ylabel = 'Flux [mJy]')
        else:
            ax.set(xlabel = r'$\lambda$ rest-frame [$\mu$m]', ylabel = 'Flux [mJy]')
        ax.plot(x,y*1000, ls='-',linewidth=0.7,color='k')
        if self.fit_result != None: 
            ax.fill_between(x,lb*1000,ub*1000,color='steelblue',alpha=0.3)

        if self._phot != None:
            #initialize fitting arrays
            if obs_frame == True:
                fit_wl = self._phot[0] * (1+self.z)
            else:
                fit_wl = self._phot[0] 
            fit_flux = 1000*self._phot[1] #mJy
            fit_err = 1000*self._phot[2]
            # check for nondetections and or incorrect input
            mask = (fit_wl < 0) | (fit_flux < 0) | (fit_err < 0)
            fit_wl = fit_wl[~mask]
            fit_flux = fit_flux[~mask]
            fit_err = fit_err[~mask]
            ax.errorbar(fit_wl, fit_flux, fit_err, 
                        c='r', ls='', marker = 'o', ms = 3,
                        elinewidth=0.5, capsize = 1.5, ecolor = 'r')
        ax.set(xscale='log', yscale='log')
        ax.set(xlim = (x.min(), x.max()*0.2), ylim=(1e-2,2e2))
        ax.annotate(f'z = {np.round(float(self.z),2)}', xy=(0.02, 0.93), xycoords = 'axes fraction')
        ax.annotate(r'$\beta$ '+f'= {np.round(self.beta,2)}', xy=(0.02, 0.86), xycoords = 'axes fraction')
        ax.annotate(r'$T$ '+f'= {np.round(self.T,1)} K', xy=(0.02, 0.79), xycoords = 'axes fraction')
        return fig, ax
    

    def plot_corner(self,**kwargs):
        """ 
        Plot a corner plot showing the results from the MCMC fit to the data. All kwargs will be passed on to `corner.corner()`.

        Returns:
            matplotlib.pyplot.figure: the current figure
        """
        data = self.fit_result['sampler'].get_chain(flat=True)[::10,:]
        labels = {'T':r'$T$','beta':r'$\beta$','N':r'$L_{\rm IR}$','z':r'$z$','alpha':r'$\alpha$','l0':r'$\lambda_0$'}
        if 'N' in self._to_vary:
            lirs = self._get_chain_for_parameter('L',sample_by=10)
            lirs = lirs.reshape(len(lirs),1)
            whereN = np.where(np.array(self._to_vary) == 'N')[0]
            data[:,whereN] = lirs
        fig = corner.corner(
        data, 
        labels=[labels[key] for key in self._to_vary], 
        quantiles=(0.16,0.5,0.84),
        show_titles=True,
        **kwargs
        )
        return fig


    def eval(self, wl,z=None):
        """Evaluate MBB at wavelength
        
        Return evaluation of this MBB's function if observed at the given wavelengths wl
        shifted to redshift z, in Jy. Set z=0 to get rest-frame evaluation. Default is to 
        give observed-frame flux.

        Args:
            wl (float): wavelength(s) in micron
            z (float): redshift at which the model should be evaluated.

        Returns:
            float: value of mbb at the wavelength ``wl``
        """
        params = self._fit_param_dict()
        if z!=None:
            params['z']=z
        return self._eval_mbb(wl, **params)


    def _eval_mbb(self, wl, N, T, beta, z=0,alpha=2,l0=200):
        """Return evaluation of this MBB's function but with variable parameters. See docs for eval()"""
        return self._model(wl/(1+z),N=N,beta=beta,T=T, z=z,alpha=alpha,l0=l0)*u.Jy


    def get_peak_wavelength(self):
        '''Get the peak (rest-frame) wavelength of this ModifiedBlackbody, in microns.'''
        x = np.logspace(1,3,5000)
        y = self.eval(x,z=0)
        peak = np.nanargmax(y)
        peak_wl = x[peak] * u.micron
        return peak_wl


    def get_luminosity(self, wllimits=(8,1000)):
        """get integrated luminosity for the current MBB state between wavelength limits
         in microns.

         Args:
            wllimits (tuple): rest-frame wavelength limits in microns (lo, hi) between which to integrate

         Returns:
            float: the luminosity integrated between rest-frame wavelength limits given by ``wllimits``
         """

        return self._integrate_mbb(**self._fit_param_dict(), wllimits=wllimits)


    def _integrate_mbb(self,N,T,beta,z=0,alpha=2,l0=200,wllimits=(8,1000)):
        """
        integrate a model with given N, beta, T between wllimits in rest-frame. See docs for get_luminosity,
        which is essentially a wrapper for this function.

        """

        if len(wllimits) == 2 and wllimits[0] < wllimits[1]:
            nulow = (con.c/(wllimits[1]*u.um)).to(u.Hz)
            nuhigh = (con.c/(wllimits[0]*u.um)).to(u.Hz)
            nu = np.logspace(np.log10(nulow.value), np.log10(nuhigh.value), 1000) * u.Hz
            dnu = nu[1:] - nu[0:-1]
            DL = self._cosmo.luminosity_distance(z)
            lam = nu.to(u.um, equivalencies=u.spectral()).value  
            lum = np.sum(4*np.pi*DL**2 * self._eval_mbb(lam[:-1],N,T,beta, alpha=alpha,l0=l0) * dnu)/(1+z)
            return lum.to(u.Lsun)
        else: raise ValueError(f"wllimits must be in the form (l_low, l_high) with l_low < l_high, received {wllimits}")
    

    def _compute_dust_mass(self,):
        '''
        Compute dust mass for this ModifiedBlackbody.
        '''
        l0= 850.
        DL = self._cosmo.luminosity_distance(self.z)
        kappa_B_T = 0.15*u.m**2/u.kg * 1e26 * planckbb(l0, T=self.T) #kappa coeff: 
                                                                        # 0.0469 taken from Traina+2024/Draine+14 at 850um
                                                                        # 0.15 from Casey+12 at 850um
        Snu = self.eval(l0,z=0).value
        dustmass = Snu * DL**2 / kappa_B_T / (1.+self.z)
        return dustmass.to(u.Msun)


    def _run_fit(self, p0,nwalkers,niter,ndim,lnprob,ncores=NCORES,to_vary=['N','beta','T'], fixed=None,pool=None):
        """
        Function to handle the actual MCMC fitting routine of this ModifiedBlackbody's internal model.

        Args:
            p0: initial parameter array (usually [N, T, beta])
            nwalkers: number of walkers to use in MCMC run
            niter: number of iterations
            ndim: dimensionality (usally len(p0))
            lnprob: function used to determine logarithmic probability
            ncores: number of CPU cores to use
            to_vary: parameter names to vary in fit.
            fixed: dictionary of fixed parameters (keys=names, values=values)
            pool: optional multiprocessing.pool.Pool to pass to EnsembleSampler
        Returns:
            Dictionary with keys ``sampler``,``pos``,``prob``, and ``state``, which encode the results of the fit.
            ``sampler`` is the actual chain of parameter values from the MCMC run. 
            ``pos``, ``prob``, and ``state`` are the output of the ``run_mcmc`` function from the ``emcee.EnsembleSampler <https://emcee.readthedocs.io/en/stable/user/sampler/>``_
        """
        close_pool = False
        if pool is None:
            if ncores > 1:
                pool = mp.get_context("spawn").Pool(processes=ncores)
                close_pool = True
        try:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, parameter_names=to_vary, kwargs=fixed)
            print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, NBURN,progress=True)
            sampler.reset()
            print("Running fitter...")
            state = sampler.run_mcmc(p0, niter,progress=True)
            print("Done\n")
        finally:
            if close_pool and pool is not None:
                pool.close()
                pool.join()
        return sampler
    

    def _fit_param_dict(self):
        """
        Convenience function to return this ModifiedBlackbody's fitting parameters (so N instead of L) as a dictionary
        """
        return {'N':self.N,'T':self.T,'beta':self.beta,'z':self.z, 'alpha':self.alpha, 'l0':self.l0}
    

    def _get_model_spread(self, lam, nsamples=200):
        """
        Function to get the median, 16th, and 84th percentile of the ModifedBlackbody spectrum (posterior)
        
        Args:
            lam (array): wavelength in microns at which to sample the posterior spectrum
            nsamples (int): number of samples to draw from the posterior sampler

        Returns:
            tuple of arrays (float, float, float): the median, 16th, and 84th percentile of the spectrum.
        """
        models = []
        flattened_chain = self.fit_result['sampler'].get_chain(flat=True)
        draw = np.floor(np.random.uniform(0,len(flattened_chain),
            size=nsamples)).astype(int)
        params = flattened_chain[draw]
        p = self._fit_param_dict()
        for t in params:
            for i,key in enumerate(self._to_vary): # which parameters did we vary
                p[key] = t[i] # replace that parameter with fitted parameter
            mod = self._model(lam,**p)
            models.append(mod)
        spread = np.nanstd(models, axis=0)
        lb,med_model,ub = np.nanpercentile(models,[16,50,84],axis=0)
        return med_model, lb, ub


    def _get_params_spread(self):
        """Get the median, 16th, and 84th percentile of all the fit parameters.
        """
        params = self.fit_result['sampler'].get_chain(flat=True)
        params_res = np.nanpercentile(params,[16,50,84],axis=0)
        return params_res


    def _select_model(self):
        """
        choose which of the modifed blackbody models (include MIR power law? optically thin?) is appropriate 
        based on the ModifiedBlackbody initialization arguments pl = True/False and opthin = True/False.
        Previously this function returned entirely different functions, now it does this effectively using functools.partial.
        """
        return partial(mbb_func, opthin=self.opthin, pl=self.pl)


    def _lnlike(self, params, **kwargs):
        x = self._phot[0]
        #reset rest frame wls if z is varied
        if 'z' in params.keys():
            x *= (1.0+self.z)/(1.0+params['z'])
        y = self._phot[1]
        yerr = self._phot[2]
        ymodel = self._model(x, **params, **kwargs)
        wres = np.sum(((y-ymodel)/yerr)**2)
        lnlike = -0.5*wres
        if np.isnan(lnlike):
            return -np.inf
        return lnlike
    

    def _lnprior(self, params):
        if 'T' in params.keys():
            T = params['T']
            if T < 5 or T > 120: return -np.inf
        if 'beta' in params.keys(): 
            beta = params['beta']
            if beta > 5.0 or beta < 0.1: return -np.inf
        if 'z' in params.keys(): 
            z = params['z']
            if z > 12.0 or z < 0.1: return -np.inf
        
        def ln_gauss(x,mu,sigma):
            return -(x - mu) ** 2 / (2 * sigma ** 2)

        #determine prior based on multiplying individual prior functions (addition in log space)
        if self._priors == None: return 0.0
        lnpriors = 0.0
        for p in params.keys():
            if p in self._priors.keys():
                try:
                    if type(self._priors[p]) == dict:
                        val = ln_gauss(params[p], mu=self._priors[p]['mu'], sigma=self._priors[p]['sigma'])
                    else: val = np.log(self._priors[p](params[p]))
                except KeyError as e: # if no prior available, use uniform prior
                    warnings.warn(f'Received error "{e}" due to incompatible prior for param "{p}"')
                    val = 0.0
                if not np.isnan(val) and val <= 0.0: lnpriors += val
        return lnpriors


    def _lnprob(self, params, **kwargs):
        lp = self._lnprior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._lnlike(params, **kwargs)


    def save_out_full(self,filepath):
        """write out full MBB including fit and sampler (not yet implmented)"""
        raise NotImplementedError()
    

    @classmethod
    def restore_from_file(self,filepath):
        """read in full MBB including fit and sampler (not yet implemented)"""
        raise NotImplementedError()
    


