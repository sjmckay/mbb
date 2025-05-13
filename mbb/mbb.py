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



from .mbb_funcs import mbb_fun_ot, mbb_fun_go, mbb_fun_go_pl, mbb_fun_ot_pl, planckbb

class ModifiedBlackbody:
    """Class to represent a modified blackbody (or MBB).
    
    This class can be used to encapsulate a single MBB model, or to perform an SED fit to photometry. The results can be easily plotted or updated as needed, and various parameters/statistics can be extracted.
    The models are based off of `Casey et al. (2012) <https://doi.org/10.1111/j.1365-2966.2012.21455.x>`_.

    Args:
        L (float): log10 of luminosity in solar units. If fitting data, this will set the initial guess for the fit.
        T (float): dust temperature in K. If fitting data, this will set the initial guess for the fit.
        beta (float): dust emissivity spectral index. If fitting data, this will set the initial guess for the fit.
        z (float): Redshift of this galaxy.
        opthin (bool): Whether or not the model should assume optically thin dust emission.
        pl (pool): Whether or not the model should include a MIR power law (as in Casey+ 2012)
    """
    def __init__(self, L, T, beta, z, opthin=True, pl=False):
        self.L = L
        self.T = T 
        self.beta = beta 
        self.z = z
        self.pl = pl
        self.opthin=opthin
        self.model = self._select_model()
        self.N = 11
        Lcurr = np.log10(self.get_luminosity((8,1000)).value)
        while((Lcurr > (L+0.0001)) | (Lcurr < (L-0.0001))):
            self.N = self.N + 0.1*(L-Lcurr)
            Lcurr = np.log10(self.get_luminosity((8,1000)).value)
        self.L = np.round(Lcurr,2)
        self.dust_mass = self._compute_dust_mass()

    def fit(self, phot, nwalkers=400, niter=2000, stepsize=1e-7):
        """Fit photometry

        Fit a modified blackbody to photometry.
        Updates the parameters of this MBB model to the best-fit parameters of the fit, and populates the "result"
        attribute of the ModifiedBlackbody with the fit results.

        Args:
            phot (array-like): wavelengths and photometry, arranged as a 3 x N array (wavelength, flux, error). 
            Wavelengths should be given as rest-frame values.
            nwalkers (int): how many walkers should be used in the MCMC fit. 
            niter (int): how many iterations to run in the fit.
            stepsize (float): stepsize used to randomize the initial walker values. 
        """
        phot = np.asarray(phot).reshape(3,-1) # make sure x,y,yerr are in proper shape
        self.phot = (phot[0],phot[1],phot[2]) # emcee takes args as a list
        init = [self.N,self.T,self.beta]

        if len(phot[0]) < 3:
            init = init[0:2]
        if len(phot[0]) < 2:
            init = init[0:1]
        ndim=len(init)
        p0 = [np.array(init) + stepsize * np.random.randn(ndim) for i in range(nwalkers)]
        result = self._run_fit(p0=p0, nwalkers=nwalkers, niter=niter, lnprob=self._lnprob, 
            ndim=ndim)#, data = self.phot)
        self.result = result
        medtheta = self._get_theta_spread()
        self.update(*medtheta[1])

    def update_L(self, L=None, T=None, beta=None):
        """ update modified blackbody parameters (not redshift or model), given new luminosity, temperature, and emissivity. """
        if T: self.T = T 
        if beta: self.beta = beta
        if L: 
            Lcurr = np.log10(self.get_luminosity((8,1000)).value)
            while((Lcurr > (L+0.001)) | (Lcurr < (L-0.001))):
                self.N = self.N * (L/Lcurr)
                Lcurr = np.log10(self.get_luminosity((8,1000)).value)
            self.L = np.round(Lcurr,2)
        self.dust_mass = self._compute_dust_mass()

    def update(self, N=None, T=None, beta=None):
        """ update modified blackbody parameters (not redshift or model), given new temperature,  emissivity, 
        and N value (related to luminosity---see Casey+ 2012). """
        if N: self.N = N
        if T: self.T = T 
        if beta: self.beta = beta
        self.L = np.log10(self.get_luminosity((8,1000)).value)
        self.dust_mass = self._compute_dust_mass()

    def save_state(self, filepath):
        """Save modified blackbody state
        
        write string version of ModifiedBlackbody to file that can be used to reinitialize 
        using 'ModifiedBlackbody.load_fit_from_file' 

        Args:
            filepath (str): path to where the model should be saved.
        
        """
        with open(filepath,'w+') as f:
            f.writelines('# L    T    beta    z    opthin    pl\n')
            text = f'{np.round(np.log10(self.get_luminosity((8,1000)).value),4)}'\
            + f'\t{np.round(self.T,4)}\t{np.round(self.beta,4)}\t{np.round(self.z,4)}\t{self.opthin}\t{self.pl}\t\n'
            f.writelines(text)
        return None

    @classmethod
    def load_state_from_file(cls,filepath):
        """Load state
        
        initialize ModifiedBlackbody from file containing parameters (created using the save_state() function 
        of a ModifiedBlackbody() instance.

        Args:
            filepath(string): path to where the model should be loaded from.

        Returns: 
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
            bits = lines[1].split('\t')
            L = float(bits[0])
            T = float(bits[1])
            beta = float(bits[2])
            z = float(bits[3])
            if bits[4] == 'True': opthin = True
            else: opthin = False
            if bits[5] == 'True': pl = True
            else: pl = False
        return cls(L,T,beta,z,opthin,pl)

    def save_out_full(self,filepath):
        """write out full MBB including fit and sampler"""
        raise NotImplementedError()
    
    def restore(self,filepath):
        """read in full MBB including fit and sampler"""
        raise NotImplementedError()
    
    def plot_sed(self, obs_frame=False,ax=None):
        """plot the rest-frame form of this mbb just for basic visualization. It is recommended 
        to use a separate, more detailed plotting function for figures.

        Args:
            obs_frame (bool): whether to plot against observed-frame wavelengths (default is rest frame).s
            ax (matplotlib.pyplot.Axes): axes to plot the model on. 
        """

        if ax is None: fig, ax = plt.subplots(figsize=(5,4),dpi=120) 
        else: fig = ax.get_figure()
        x = np.logspace(1,4,500)
        if hasattr(self, 'result'):
            nsamples = 200
            y, lb,ub = self._get_model_spread(x)
        else: y = self.eval(x)
        if obs_frame == True:
            x *= (1.+self.z)
            ax.set(xlabel = r'$\lambda$ observed-frame [$\mu$m]', ylabel = 'Flux [mJy]')
        else:
            ax.set(xlabel = r'$\lambda$ rest-frame [$\mu$m]', ylabel = 'Flux [mJy]')
        ax.plot(x,y*1000, ls='-',linewidth=0.7,color='k')
        if hasattr(self, 'result'): 
            ax.fill_between(x,lb*1000,ub*1000,color='steelblue',alpha=0.3)

        if hasattr(self,'phot'):
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
    
    def plot_corner(self):
        """ Plot a corner plot showing the results for beta, T, and L from the MCMC fit to the data.
        """
        data = self.result['sampler'].flatchain[::10,:]
        n = len(data)
        lirs=[]
        for i in range(len(data)):
            lirs.append(np.log10(self._integrate_mbb(*data[i],z=self.z,
                                 wllimits=(8,1000)).value))
        lirs = np.asarray(lirs).reshape(len(lirs),1)
        data = np.concatenate((data[:,1:], lirs),axis=1)
        fig = corner.corner(
        data, 
        labels=[r'$T$',r'$\beta$',r'$L_{\rm IR}$',], 
        quantiles=(0.16,0.5,0.84),
        show_titles=True
        )
        return fig

    def eval(self, wl,z=0):
        """Evaluate MBB at wavelength
        
        Return evaluation of this MBB's function if observed at the given wavelengths wl
        shifted to redshift z, in Jy. Leave z=0 to get rest-frame evaluation.
        This is a wrapper for eval_mbb but with the current mbb parameters supplied.

        Args:
            wl (float): wavelength(s) in micron
            z (float): redshift to which the model should be shifted.

        Returns:
            float: value of mbb at the wavelength ``wl``
        """
        return self._eval_mbb(wl, self.N,self.T,self.beta,z)

    def _eval_mbb(self, wl, N, T, beta, z=0):
        """Return evaluation of this MBB's function but with variable N, b, or T. See docs for eval()"""
        p = [N,T,beta]
        return self.model(p, wl/(1+z), z=z)*u.Jy

    def get_luminosity(self, wllimits=(8,1000), cosmo=FlatLambdaCDM(H0=70.0, Om0=0.30)):
        """get integrated LIR luminosity for the current MBB state between wavelength limits
         in microns.

         Args:
            wllimits (tuple): rest-frame wavelength limits in microns (lo, hi) between which to integrate
            cosmo (astropy.cosmology): cosmology used for computing luminosity distance 

         Returns:
            float: the luminosity integrated between rest-frame wavelength limits given by ``wllimits``
         """

        return self._integrate_mbb(self.N,self.T,self.beta,self.z,wllimits,cosmo)

    def _integrate_mbb(self,N,T,beta,z=0,wllimits=(8,1000), 
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
            lum = np.sum(4*np.pi*DL**2 * self._eval_mbb(lam[:-1],N,T,beta) * dnu)/(1+z)
            return lum.to(u.Lsun)
    
    def _compute_dust_mass(self):
        '''
        Compute dust mass for this ModifiedBlackbody.
        '''
        l0= 850.
        DL = cosmo.luminosity_distance(self.z)
        kappa_B_T = 0.15*u.m**2/u.kg * planckbb(l0, T=self.T)
        Snu = self.eval(l0,z=0)
        dustmass = Snu * DL**2 / kappa_B_T / (1.+self.z)
        return dustmass.to(u.Msun)

    def _run_fit(self, p0,nwalkers,niter,ndim,lnprob,ncores=NCPU):
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
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
            print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, NBURN,progress=True)
            sampler.reset()
            print("Running fitter...")
            pos, prob, state = sampler.run_mcmc(p0, niter,progress=True)
            print("Done\n")
        return {'sampler':sampler, 'pos':pos, 'prob':prob, 'state':state}  
    
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
        flattened_chain = self.result['sampler'].flatchain
        draw = np.floor(np.random.uniform(0,len(flattened_chain),
            size=nsamples)).astype(int)
        thetas = flattened_chain[draw]
        for i in thetas:
            mod = self.model(i,lam,z=self.z)
            models.append(mod)
        spread = np.std(models, axis=0)
        lb,med_model,ub = np.percentile(models,[16,50,84],axis=0)
        return med_model, lb, ub

    def _get_theta_spread(self):
        """
        Function to get the median, 16th, and 84th percentile of the fit parameters (called theta in emcee)
        """
        thetas = self.result['sampler'].flatchain
        theta_res = np.percentile(thetas,[16,50,84],axis=0)
        return theta_res

    def _select_model(self):
        """
        choose which of the modifed blackbody models (include MIR power law? optically thin?) is appropriate 
        based on the ModifiedBlackbody initialization arguments pl = True/False and opthin = True / False.
        """
        if self.opthin:
            if self.pl: return mbb_fun_ot_pl
            else: return mbb_fun_ot
        else:
            if self.pl: return mbb_fun_go_pl
            else: return mbb_fun_go

    def _lnlike(self, theta):
        x = self.phot[0]
        y = self.phot[1]
        yerr = self.phot[2]
        ymodel = self.model(theta, x, z=self.z)
        wres = np.sum(((y-ymodel)/yerr)**2)
        lnlike = -0.5*wres
        if np.isnan(lnlike):
            return -np.inf
        return lnlike
        
    def _lnprior(self,theta):
        if len(theta) > 1:
            T = theta[1]
            if T > 10 and T < 100:
                if len(theta) > 2: 
                    beta = theta[2] 
                    if beta > 5.0 or beta < 0.1:
                        return -np.inf
                return 0.0
            else: return -np.inf 
        return 0.0

    def _lnprob(self, theta):
        lp = self._lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._lnlike(theta)





