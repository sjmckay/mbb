### Class to implement a modified blackbody SED, with the capability to fit the SED with various options
# and plot a simple version of it, as well as save the results to a file. 
# Author: Stephen McKay, Spring 2023


# Imports

import matplotlib.pyplot as plt
import numpy as np

import emcee
import corner

from astropy.table import Table, QTable
from astropy.io import fits
import astropy.units as u
import astropy.constants as con
from astropy.constants import c, k_B, h
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.30) 

from functools import partial

from multiprocessing import Pool
from multiprocessing import cpu_count

NCPU = cpu_count()

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
    
    This class can be used to encapsulate a single MBB model, or to perform an SED fit to photometry. The results can be easily plotted or updated as needed, and various parameters/statistics can be extracted.
    The models are based off of `Casey et al. (2012) <https://doi.org/10.1111/j.1365-2966.2012.21455.x>`_.

    Args:
        L (float): log10 of luminosity in solar units. If fitting data, this will set the initial guess for the fit.
        T (float): dust temperature in K. If fitting data, this will set the initial guess for the fit.
        beta (float): dust emissivity spectral index. If fitting data, this will set the initial guess for the fit.
        z (float): Redshift of this galaxy.
        alpha (float, optional): mid-IR power-law slope.
        l0 (float,optional): opacity turnover wavelength in microns.
        opthin (bool): Whether or not the model should assume optically thin dust emission.
        pl (pool): Whether or not the model should include a MIR power law (as in Casey+ 2012)
    """
    def __init__(self, L, T, beta, z, alpha=2.0,l0=200.,opthin=True, pl=False):
        self.L = L
        self.T = T 
        self.beta = beta 
        self.z = z
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
        self._chi2 = None
        self._n_bands = None
        self._n_params = None
    
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
    def chi2(self):
        return self._chi2
    
    @property
    def n_params(self):
        return self._n_params
    
    @property
    def n_bands(self):
        return self._n_bands

    def fit(self, phot, nwalkers=400, niter=2000, stepsize=1e-7,to_vary=['L','beta','T','z'],restframe=False):
        """Fit photometry

        Fit a modified blackbody to photometry.
        Updates the parameters of this MBB model to the best-fit parameters of the fit, and populates the "fit_result"
        attribute of the ModifiedBlackbody with the fit results.

        Args:
            phot (array-like): wavelengths and photometry, arranged as a 3 x N array (wavelength, flux, error). 
            Wavelengths should be given as rest-frame values.
            nwalkers (int): how many walkers should be used in the MCMC fit. 
            niter (int): how many iterations to run in the fit.
            stepsize (float): stepsize used to randomize the initial walker values. 
            to_vary (list): list of parameter names, e.g., ['L','beta','T','z','alpha','l0'] to vary in the fit. The rest will be fixed. 'L' should always be included since it reflects the normalization of the model.
            restframe (bool): whether wavelengths in `phot` are given in rest frame (default is observed frame)
        """
        phot = np.asarray(phot).reshape(3,-1) # make sure x,y,yerr are in proper shape
        if restframe: self._phot = (phot[0],phot[1],phot[2]) # emcee takes args as a list
        else: self._phot = (phot[0]/(1.0+self.z),phot[1],phot[2])
        initdict = self._fit_param_dict()
        try:
            init = [initdict[key] for key in to_vary]
        except KeyError:
            raise ValueError(f'Varied parameters must be one of {list(initdict.keys())}')
        ndim=len(init)
        #set up initial parameters
        fixed = initdict.copy()
        for key in to_vary: fixed.pop(key)
        self._to_vary = to_vary
        p0 = [np.array(init) + stepsize * np.random.randn(ndim) for i in range(nwalkers)]
        #run the MCMC fit
        result = self._run_fit(p0=p0, nwalkers=nwalkers, niter=niter, lnprob=self._lnprob, 
            ndim=ndim, to_vary = to_vary, fixed = fixed)#, data = self.phot)
        self._fit_result = result
        #get 16,50,84 percentiles of fitted parameters and update
        medtheta = self._get_theta_spread()
        updated = initdict
        for i, key in enumerate(to_vary):
            updated[key] = medtheta[1][i]
        self._update_N(**updated)
        #save chi2 and fitted parameter results in easy to access format
        yprime = self.phot[0]
        self._chi2 = np.nansum( (self.phot[1]-yprime)**2/self.phot[2]**2 )
        self._n_params = ndim
        self._n_bands = len(self.phot[0])
    
    def _get_chain_for_parameter(self, param):
        '''retrieve the chain of walker values for a given parameter that was varied in the fit'''
        if self.fit_result != None:
            if param in self._to_vary:
                chain = self.fit_result['sampler'].flatchain[:,:]
                where = np.where(np.array(self._to_vary) == param)[0]
                return np.squeeze(chain[:,where])
            else:
                raise KeyError(f'Cannot get chain for "{param}" since it was not varied in the fit.')
        return None
    
    def posterior(self,param,q=[16,50,84]):
        '''Determine the posterior values of a given fit parameter.

        Example: 

        .. code-block:: python
            
           m.posterior('beta', q = [16,50,84]) # get median and 16th--84th percentile interval
        
        Args:
            param (str): name of parameter, one of the 'to_vary' arguments passed to ModifiedBlackbody.fit()
            q (array-like of float): percentile or sequence of percentiles to compute from the posterior distribution
            
        Returns:
            float or array: the percentiles of the posterior distribution for parameter 'param'
        '''
        if self.fit_result != None:
            try:
                chain = self._get_chain_for_parameter(param)
                return np.nanpercentile(chain, q=q)
            except Exception as e:
                raise Exception(f"Unable to get posterior for parameter {param}: failed with error '{e}'")
        else: raise AttributeError(f'No fit has been run yet, so no posterior for {param} exists.')
            

    def update(self, L=None, T=None, beta=None,z=None,alpha=None,l0=None):
        """ update modified blackbody parameters (not the underlying model)."""
        if T: self.T = T 
        if beta: self.beta = beta
        if z: self.z = z
        if alpha: self.alpha = alpha
        if l0: self.l0 = l0
        if L: 
            Lcurr = np.log10(self.get_luminosity((8,1000)).value)
            while((Lcurr > (L+0.001)) | (Lcurr < (L-0.001))):
                self.N = self.N * (L/Lcurr)
                Lcurr = np.log10(self.get_luminosity((8,1000)).value)
            self.L = np.round(Lcurr,2)

    def _update_N(self, N=None, T=None, beta=None,z=None,alpha=None,l0=None):
        """ update modified blackbody parameters (not model), using N rather than luminosity (used in fitting). """
        if N: self.N = N
        if T: self.T = T 
        if z: self.z = z
        if beta: self.beta = beta
        if alpha: self.alpha = alpha
        if l0: self.l0 = l0
        self.L = np.log10(self.get_luminosity((8,1000)).value)

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

        if self.phot != None:
            #initialize fitting arrays
            if obs_frame == True:
                fit_wl = self.phot[0] * (1+self.z)
            else:
                fit_wl = self.phot[0] 
            fit_flux = 1000*self.phot[1] #mJy
            fit_err = 1000*self.phot[2]
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
        data = self.fit_result['sampler'].flatchain[::10,:]
        n = len(data)
        params = self._fit_param_dict()
        labels = {'T':r'$T$','beta':r'$\beta$','N':r'$L_{\rm IR}$','z':r'$z$','alpha':r'$\alpha$','l0':r'$\lambda_0$'}
        if 'N' in self._to_vary:
            lirs=[]
            for i in range(len(data)):
                p = params.copy()
                for j, key in enumerate(self._to_vary): p[key] = data[i][j]
                lirs.append(np.log10(self._integrate_mbb(**p,wllimits=(8,1000)).value))
            lirs = np.asarray(lirs).reshape(len(lirs),1)
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


    def get_luminosity(self, wllimits=(8,1000), cosmo=FlatLambdaCDM(H0=70.0, Om0=0.30)):
        """get integrated LIR luminosity for the current MBB state between wavelength limits
         in microns.

         Args:
            wllimits (tuple): rest-frame wavelength limits in microns (lo, hi) between which to integrate
            cosmo (astropy.cosmology): cosmology used for computing luminosity distance 

         Returns:
            float: the luminosity integrated between rest-frame wavelength limits given by ``wllimits``
         """

        return self._integrate_mbb(**self._fit_param_dict(), wllimits=wllimits,cosmo=cosmo)

    def get_peak_wavelength(self):
        '''Get the peak (rest-frame) wavelength of this ModifiedBlackbody, in microns.'''
        x = np.logspace(1,3,5000)
        y = self.eval(x,z=0)
        peak = np.nanargmax(y)
        peak_wl = x[peak] * u.micron
        return peak_wl


    def _integrate_mbb(self,N,T,beta,z=0,alpha=2,l0=200,wllimits=(8,1000), 
                       cosmo=FlatLambdaCDM(H0=70.0, Om0=0.30)):
        """
        integrate a model with given N, beta, T between wllimits in rest-frame. See docs for get_luminosity,
        which is essentially a wrapper for this function.

        """

        if len(wllimits) == 2 and wllimits[0] < wllimits[1]:
            nulow = (con.c/(wllimits[1]*u.um)).to(u.Hz)
            nuhigh = (con.c/(wllimits[0]*u.um)).to(u.Hz)
            nu = np.linspace(nulow, nuhigh, 20000)
            dnu = nu[1:] - nu[0:-1]
            DL = cosmo.luminosity_distance(z)
            lam = nu.to(u.um, equivalencies=u.spectral()).value  
            lum = np.sum(4*np.pi*DL**2 * self._eval_mbb(lam[:-1],N,T,beta, alpha=alpha,l0=l0) * dnu)/(1+z)
            return lum.to(u.Lsun)
    
    def _compute_dust_mass(self,):
        '''
        Compute dust mass for this ModifiedBlackbody.
        '''
        l0= 850.
        DL = cosmo.luminosity_distance(self.z)
        kappa_B_T = 0.15*u.m**2/u.kg * 1e26 * planckbb(l0, T=self.T) #kappa coeff: 
                                                                        # 0.0469 taken from Traina+2024/Draine+14 at 850um
                                                                        # 0.15 from Casey+12 at 850um
        Snu = self.eval(l0,z=0).value
        dustmass = Snu * DL**2 / kappa_B_T / (1.+self.z)
        return dustmass.to(u.Msun)

    def _run_fit(self, p0,nwalkers,niter,ndim,lnprob,ncores=NCPU,to_vary=['N','beta','T','z'], fixed=None):
        """
        Function to handle the actual MCMC fitting routine of this ModifiedBlackbody's internal model.

        Args:
            p0: initial parameter array (usually [N, T, beta])
            nwalkers: number of walkers to use in MCMC run
            niter: number of iterations
            ndim: dimensionality (usally len(p0))
            lnprob: function used to determine logarithmic probability
            ncores: number of CPU cores to use

        Returns:
            Dictionary with keys ``sampler``,``pos``,``prob``, and ``state``, which encode the results of the fit.
            ``sampler`` is the actual chain of parameter values from the MCMC run. 
            ``pos``, ``prob``, and ``state`` are the output of the ``run_mcmc`` function from the ``emcee.EnsembleSampler <https://emcee.readthedocs.io/en/stable/user/sampler/>``_
        """
        with Pool(ncores) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, parameter_names=to_vary, kwargs=fixed)
            print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, NBURN,progress=True)
            sampler.reset()
            print("Running fitter...")
            pos, prob, state = sampler.run_mcmc(p0, niter,progress=True)
            print("Done\n")
        return {'sampler':sampler, 'pos':pos, 'prob':prob, 'state':state}  
    

    def _fit_param_dict(self):
        """
        Convenience function to return this ModifiedBlackbody's fitting parameters (so N instead of L) as a dictionary
        """
        return {'N':self.N,'T':self.T,'beta':self.beta,'z':self.z}
    
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
        flattened_chain = self.fit_result['sampler'].flatchain
        draw = np.floor(np.random.uniform(0,len(flattened_chain),
            size=nsamples)).astype(int)
        thetas = flattened_chain[draw]
        p = self._fit_param_dict()
        for t in thetas:
            for i,key in enumerate(self._to_vary): # which parameters did we vary
                p[key] = t[i] # replace that parameter with fitted parameter
            mod = self._model(lam,**p)
            models.append(mod)
        spread = np.nanstd(models, axis=0)
        lb,med_model,ub = np.nanpercentile(models,[16,50,84],axis=0)
        return med_model, lb, ub

    def _get_theta_spread(self):
        """
        Function to get the median, 16th, and 84th percentile of the fit parameters (called theta in emcee)
        """
        thetas = self.fit_result['sampler'].flatchain
        theta_res = np.nanpercentile(thetas,[16,50,84],axis=0)
        return theta_res

    def _select_model(self):
        """
        choose which of the modifed blackbody models (include MIR power law? optically thin?) is appropriate 
        based on the ModifiedBlackbody initialization arguments pl = True/False and opthin = True/False.
        Previously this function returned entirely different functions, now it does this effectively using functools.partial.
        """
        return partial(mbb_func, opthin=self.opthin, pl=self.pl)

    def _lnlike(self, theta, **kwargs):
        x = self.phot[0]
        y = self.phot[1]
        yerr = self.phot[2]
        ymodel = self._model(x, **theta, **kwargs)
        wres = np.sum(((y-ymodel)/yerr)**2)
        lnlike = -0.5*wres
        if np.isnan(lnlike):
            return -np.inf
        return lnlike
        
    def _lnprior(self, theta):
        if 'T' in theta.keys():
            T = theta['T']
            if T < 5 or T > 120: return -np.inf
        if 'beta' in theta.keys(): 
            beta = theta['beta']
            if beta > 5.0 or beta < 0.1: return -np.inf
        if 'z' in theta.keys(): 
            z = theta['z']
            if z > 12.0 or z < 0.1: return -np.inf
        return 0.0

    def _lnprob(self, theta, **kwargs):
        lp = self._lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._lnlike(theta, **kwargs)


    def save_out_full(self,filepath):
        """write out full MBB including fit and sampler (not yet implmented)"""
        raise NotImplementedError()
    
    @classmethod
    def restore_from_file(self,filepath):
        """read in full MBB including fit and sampler (not yet implemented)"""
        raise NotImplementedError()
    


