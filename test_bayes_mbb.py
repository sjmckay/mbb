
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import LogLocator
import numpy as np
import pandas as pd

import emcee
import corner

from astropy.table import Table, QTable
from astropy.io import fits
import astropy.units as u
import re
from sed_py.mpfit.mpfit import mpfit
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.30) 

from scipy.optimize import curve_fit

from multiprocessing import Pool


import astropy.constants as con
from astropy.constants import c, k_B, h
c = c.value
k_B = k_B.value
h = h.value


from time import sleep

plt.rcParams['font.family'] = 'serif' 

pathname = '/home/sjmckay3/Research/'
OUTDIR = pathname+'results/'

op_thin = True
cmb = True

Tcmb0 = 2.73

beta_lim = 4.0

speczs = [1,2,3,4,5,7,9,12,18,22,25,35,40,46,48,56,59,66,68,74]

#ids = [1]
ids = range(1,76)#[1,4,5,8,9,10,12,14,15,16,18,20,24,26,27,28,29,30,31,38,40,41,42,43,49,50,51,52,53,54,56,60,61,62,65,67,68] #list of actual id numbers of galaxies (NOTE: must be consecutive in input file)
#ids = range(4,5)

offset = 0 # number of galaxies not being fit which appear before the first one being fit, in the input file. e.g. if you
#are fitting starting with the 4th galaxy down the list, offset should  be 3. This is to allow flexibility for not always fitting every galaxy
# in the input file all at once, or to allow you to not start with the first galaxy in the file

obs = pd.read_csv(pathname+'user_files_alma/alma_cdfs_obs.dat',sep='\t')

#print(obs)
z = np.array(obs['zbest'])


NWALKERS = 180
NITER = 2000
NBURN = 300
STEPSIZE = 1e-7

CURRENT_Z = 0

LLO = 8
LHI = 1000




















def planckbb(l,T): # note although this requires wavlength in micron, it represents B_nu
    return 2*h / c**2 * (c/(l*1e-6))**3 / (np.exp(h*c/(l*1e-6*k_B*T))-1)

def gb_fun(theta,l,op_thin = True,z = 0):
    """ coupled greybody/powerlaw function based on Casey et al. 2012
        note p is parameters array, and l is wavelength (lambda) in microns.
        Returns greybody function values (at wavelength points l) in Jy.
        Added by S. McKay, June 2022
    """
    #assign variable parameter values
    Nbb = theta[0] # norm constant for greybody (gb)
    T = theta[1] # temperature of gb in K

    if len(theta) > 2: 
        beta = theta[2] # emissivity index (set to p[2] if enough data points in FIR)
    else:
        beta = 1.8

    alpha = 2.0 # powerlaw slope  (set to p[3] if enough data points in FIR)

    l_c = 0.75*(T*(alpha*7.243e-5 + 1.905e-4)+ (alpha*6.246 + 26.68)**(-2))**(-1) #critical wavelength (microns) where MIR powerlaw turns over (approx. from casey2012)
    
    l0=200. # reference wavelength in microns where optical depth = 1

    # norm constant for powerlaw (pl), approx from casey2012. factors of 1e-6 are to convert so SI units inside exponents, 
    # uses log of Nbb as Nbb parameter in order to be roughyl the same order of magnitude as T, beta.

    Npl = 10.0**Nbb* 2*h / c**2 *((1.0 -np.exp(-(l0/l_c)**beta))*(c/(l_c*1e-6))**3)/(np.exp(h*c/(l_c*1e-6*k_B*T))-1.0)/((l_c*1e-6)**alpha)  #gen opacity

    #equation for greybody+powerlaw based on casey2012. factors of 1e-6 are to convert to SI units inside exponents, 
    # uses log of Nbb as Nbb parameter in order to be roughyl the same order of magnitude as T, beta.
    
    if cmb is False:
        ### optically thin version
        if op_thin is True:
            result = 10.0**Nbb *(200e-6/c)**beta * 2*h / c**2 * (c/(l*1e-6))**(beta+3)/(np.exp(h*c/(l*1e-6*k_B*T))-1)
        else: 
        ### general opacity version
            result = 10.0**Nbb  * 2*h / c**2 * ((1.0 - np.exp(-(l0/l)**beta))*(c/(l*1.e-6))**3.0)/(np.exp(h*c/(l*1.e-6*k_B*T))-1.0) + Npl*(l*1e-6)**alpha * np.exp(-(l/l_c)**2) #uncomment second term to add in powerlaw
    else: # correct for cmb, assume l is in observed frame
        Tcmbz = Tcmb0*(1+z)
        Tz = (T**(4+beta)  +  (Tcmb0)**(4+beta) * ((1+z)**(4+beta)-1) ) **(1/(4+beta))
        ### optically thin version
        if op_thin is True:
            result = (1 - (planckbb(l,Tcmbz)/planckbb(l,Tz))) * (Tz/T)**(4+beta) * 10.0**Nbb *(200e-6/c)**beta * 2*h / c**2 * (c/(l*1e-6))**(beta+3)/(np.exp(h*c/(l*1e-6*k_B*T))-1)
        else: 
        ### general opacity version
            result = (1 - (planckbb(l,Tcmbz)/planckbb(l,Tz))) * (Tz/T)**(4+beta) * 10.0**Nbb * 2*h / c**2 * ((1.0 - np.exp(-(l0/l)**beta))*(c/(l*1.e-6))**3.0)/(np.exp(h*c/(l*1.e-6*k_B*T))-1.0) + Npl*(l*1e-6)**alpha * np.exp(-(l/l_c)**2) #uncomment second term to add in powerlaw
    

    return result
    


def lnlike(theta, x,y,yerr,op_thin = True,z=0):
    yerr[yerr==0]=-1
    ymodel = gb_fun(theta,x,op_thin=op_thin,z=z)
    wres = np.sum(((y-ymodel)/yerr)**2)
    LnLike = -0.5*wres
    if np.isnan(LnLike):
        return -np.inf
    return LnLike
    
def lnprior(theta):
    #assign variable parameter values
    Nbb = theta[0] # norm constant for greybody (gb)
    T = theta[1] # temperature of gb in K    
    
    if T > 10 and T < 90:
        if len(theta) > 2: 
            beta = theta[2] # emissivity index (set to p[2] if enough data points in FIR)
            if beta > 4.0 or beta < 0.8:
                return -np.inf
        return 0.0
    else: return -np.inf 



def get_model_spread(lam, nsamples,flattened_chain,z):
    models = []
    draw = np.floor(np.random.uniform(0,len(flattened_chain),size=nsamples)).astype(int)
    thetas = flattened_chain[draw]
    for i in thetas:
        mod = mbb_fun(i,lam,z=z)
        models.append(mod)
    spread = np.std(models,axis=0)
    lb,med_model,ub = np.percentile(models,[16,50,84],axis=0)
    return med_model,lb,ub


def lnprob_ot(theta, x,y,yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x,y,yerr, op_thin = True,z=CURRENT_Z)

def lnprob_go(theta, x,y,yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x,y,yerr, op_thin = False,z=CURRENT_Z)
    
def run_fit(p0,nwalkers,niter,ndim,lnprob,data):
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data, pool=pool)
        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, NBURN,progress=True)
        sampler.reset()
        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter,progress=True)
        print("Done\n")
        return sampler, pos, prob, state


def fit_greybody(t_phot,z, op_thin=True,add_flux_file=None,n=1):
    """
    fit a greybody function (gb_fun) to the observed fluxes using python version of mpfit. Added by S. McKay, June 2022.
    Returns an mpfit object whose best-fit parameters may be accessed with m.params, then passed to gb_fun.
    """
    global CURRENT_Z 
    CURRENT_Z = z
    
    # get bands to be fitted
    t_phot_fit = t_phot[t_phot['fit']==True]
    t_phot_fit = t_phot_fit[t_phot_fit['F_nu_obs'].value >= 0]

    obs_wl = np.array([a.value*1e-4 for a in t_phot_fit['lambda_eff']]) #microns
    fit_wl = np.array([a.value*1e-4 for a in t_phot_fit['lambda_eff']])/(1+z) #microns, rest-frame
    fit_flux = np.array([a.value for a in t_phot_fit['F_nu_obs']]) # Jy
    fit_err = np.array([a.value for a in t_phot_fit['e_F_nu_obs']]) # Jy

    # add in 5% flux scale calibration error
    err_mask = (fit_err>0)&(obs_wl > 100)
    fit_err[err_mask] = np.sqrt(fit_err[err_mask]**2 + (0.05*fit_flux[err_mask])**2)

    if op_thin is True:
        mask = (fit_wl > 50.0) # start with pure limit
#         if n in [13,15, 26,36, 47,72]: #26,48,72
#               mask = (obs_wl > 140.0)
#         elif n in [37,59]:
#              mask = (obs_wl > 400.0)
#         elif (0. < z < 5.6):
#              mask = (fit_wl > 50.0)# & ((obs_wl <=330.0) | (obs_wl >= 380.0))) #only fit rest wavelengths not in the warm dust zone
#         else:
#             mask = (obs_wl > 200.0) # for really high (sketchy) redshifts, fit spire and alma fluxes anyway
    else:
        mask = (fit_wl > 10.0)

    fit_wl = fit_wl[mask]
    fit_flux = fit_flux[mask]
    fit_err = fit_err[mask]

    if len(fit_wl) > 3:
        if op_thin is True:
            init = [11.,20., 1.8] #OP THIN inital value array for parameters Nbb,  T  (only add alpha and beta if enough data pts) NOTE: must be explicitly float.
        else:
            init = [11.,20., 1.8] #GEN OPAC inital value array for parameters 
    else:
        if op_thin is True:
            init = [11.,20.] #OP THIN inital value array for parameters Nbb,  T  
        else:
            init = [11.,20.] #GEN OPAC inital value array for parameters 

    ndim = len(init)
    data = (fit_wl,fit_flux,fit_err)
    nwalkers = NWALKERS
    niter = NITER
    stepsize = STEPSIZE
    p0 = [np.array(init) + stepsize * np.random.randn(ndim) for i in range(nwalkers)]
    
    if op_thin is True:
        sampler, pos, prob, state = run_fit(p0,nwalkers,niter,ndim,lnprob_ot,data)
    else:    
        sampler, pos, prob, state = run_fit(p0,nwalkers,niter,ndim,lnprob_go,data)
    
    return {'sampler':sampler,'pos':pos,'prob':prob,'state':state}


def int_lum_nu(theta, z=0, op_thin=True, low=8, high=1000, cosmo=FlatLambdaCDM(H0=70.0, Om0=0.30)):
    """ get total LIR for greybody"""
    llimits = np.array([high,low])*u.micron
    nulimits = (con.c/llimits).to(u.Hz)
    nu = np.linspace(nulimits[0],nulimits[1],20000)
    
    dnu = nu[1] - nu[0]
    lumdist = cosmo.luminosity_distance(z)
    lam = (con.c/nu).to(u.um).value  
    lum = np.sum(4*np.pi*lumdist**2 * gb_fun(theta, lam, op_thin=op_thin)*u.Jy * dnu)/(1+z)
    return np.log10(lum.to(u.Lsun).value)


def int_lum_nu_model(lam,model, z=0, op_thin=True, low=8, high=1000, cosmo=FlatLambdaCDM(H0=70.0, Om0=0.30)):
    """ get total LIR for greybody"""
    mask = (lam>=low)&(lam<=high)
    model = model[mask]*u.Jy
    nu = (con.c/(lam[mask]*u.micron)).to(u.Hz)
    dnu = nu[:len(nu)-1]-nu[1:]
    lumdist = cosmo.luminosity_distance(z)
    lum = np.sum(4*np.pi*lumdist**2 * model[:len(model)-1]*dnu/(1+z))
    return np.log10(lum.to(u.Lsun).value)


def get_med_theta(d_gb):
    thetas = d_gb['sampler'].flatchain
    theta_res = np.percentile(thetas,[16,50,84],axis=0)
    return theta_res

def get_mbb_params(d_gb, op_thin=True,z=0):
    
    params={}
    
    lam=np.logspace(0,3.3,3000)
    thetas = d_gb['sampler'].flatchain
    theta_res = get_med_theta(d_gb)
    med_model, modlb, modub = get_model_spread(lam, 200, thetas,op_thin=op_thin,z=z)
    
    L_med = int_lum_nu_model(lam,med_model, z=z, low=LLO, high=LHI, op_thin=op_thin) # integrate L over full wavelength range
    L_lb = int_lum_nu_model(lam,modlb, z=z, low=LLO, high=LHI, op_thin=op_thin) # integrate L over full wavelength range
    L_ub = int_lum_nu_model(lam,modub, z=z, low=LLO, high=LHI, op_thin=op_thin) # integrate L over full wavelength range
    
    params['L_IR']= L_med
    params['L_lb']= L_lb
    params['L_ub']= L_ub
    params['T_gb'] = theta_res[1,1]
    params['T_lb'] = theta_res[0,1]
    params['T_ub'] = theta_res[2,1]
    if len(theta_res[1,:]) > 2:
        params['beta'] = theta_res[1,2]
        params['beta_lb'] = theta_res[0,2]
        params['beta_ub'] = theta_res[2,2]
        
    return params

# do all the plotting of the various data and fits
def plot_sed(t_sed, t_phot, savename=None, 
    show=False,
    d_gb=None, comp=None, n=0, z=0, fig=None, ax = None,
    op_thin = True):
    
    create_ax = False

    nsamples = 300

    if ax is None:
        plt.rcParams['font.size'] = 12
        create_ax = True
        fig, ax = plt.subplots(figsize=(4,3), dpi=160, constrained_layout=True)
        ax.set(xlim=(3e1, 5e3), ylim=(1e-2, 2e2), xscale='log', yscale='log')
    else:
        plt.rcParams['font.size'] = 7.5
        ax.set(xlim=(3e1, 5e3), ylim=(1e-2, 3e2), xscale='log', yscale='log')

    if comp is not None: 
        plt.rcParams['font.size'] = 12
        
        lam = np.logspace(0,4,3000)
        med_model, lb,ub = get_model_spread(lam,nsamples,comp['sampler'].flatchain,op_thin=False,z=z)
        ax.plot(lam*(1+z), med_model*1000, c='maroon',ls='--',linewidth=0.9, label='Opacity + Power Law',zorder=1)
        ax.fill_between(lam*(1+z),lb*1000,ub*1000,color='firebrick',alpha=0.2)

    if create_ax is False: 
        xfrac = 0.62
    else:
        xfrac = 0.70
        
    if d_gb is not None:
        if comp is not None: 
            label = 'Optically thin'
        else: 
            label = 'MBB best-fit'
        
        theta_res = get_med_theta(d_gb)
        theta_max = d_gb['sampler'].flatchain[np.argmax(d_gb['sampler'].flatlnprobability)]
        theta16 = theta_res[0,:]
        theta50 = theta_res[1,:]
        theta84 = theta_res[2,:]
        
        
        lam = np.logspace(1,4,3000)
        
        bfmodel = gb_fun(theta_max, lam, op_thin=op_thin, z=z)
        
        med_model, lb,ub = get_model_spread(lam,nsamples,d_gb['sampler'].flatchain,op_thin=op_thin,z=z)
        ax.plot(lam*(1+z), bfmodel*1000, c='k',ls='-',linewidth=0.8, label=label,zorder=1)
        ax.fill_between(lam*(1+z),lb*1000,ub*1000,color='steelblue',alpha=0.2)
        
        if comp is None:
            T_gb = round(float(theta50[1]), 1)
            if len(theta50) > 2:
                beta_gb = round(theta50[2], 2) # only if we use beta as free param
                ax.annotate(r'$\beta$ = '+f'{beta_gb}',
                    (xfrac, 0.94), xycoords='axes fraction')  
            else:
                ax.annotate(r'$\beta \equiv$ 1.8, not fit',
                    (xfrac, 0.94), xycoords='axes fraction')
            ax.annotate(r'$T_d$ = '+f'{T_gb} K',
                    (xfrac, 0.87), xycoords='axes fraction')
        
    t_phot_fit = t_phot[t_phot['fit']==True]
    fit_wl = np.array([a.value*1e-4 for a in t_phot_fit['lambda_eff']]) #microns
    fit_flux =  np.array([a.value*1000 for a in t_phot_fit['F_nu_obs']]) #mJy
    fit_err =  np.array([a.value*1000 for a in t_phot_fit['e_F_nu_obs']]) #mJy
    
    
    #error floor and calibration
    err_mask = (fit_err>0) & (fit_wl > 100)
    fit_err[err_mask] = np.sqrt(fit_err[err_mask]**2 + (0.05*fit_flux[err_mask])**2)
    
    t_phot_nofit = t_phot[t_phot['fit']==False]
    nofit_wl =  np.array([a.value*1e-4 for a in t_phot_nofit['lambda_eff']])
    nofit_flux =  np.array([a.value*1000 for a in t_phot_nofit['F_nu_obs']])
    nofit_err =  np.array([a.value*1000 for a in t_phot_nofit['e_F_nu_obs']])

    scu_mask = (fit_wl > 440) & (fit_wl < 460)& (fit_err > 0)
    her_mask = ((fit_wl > 90) & (fit_wl < 400)) | ((fit_wl > 460) & (fit_wl < 600))& (fit_err > 0)
    alma_mask = (fit_wl > 600) & (fit_wl < 890)& (fit_err > 0)
    long_mask = (fit_wl > 890)& (fit_err > 0)
    no_mask = (fit_wl < 90) & (fit_err > 0)
    notfit_mask = (fit_wl/(1+z) < 50.0)


    ax.errorbar(fit_wl[no_mask], fit_flux[no_mask], yerr=fit_err[no_mask], marker='^', mec='b', c='k',
                mfc='w', ms=4, lw=0.6,elinewidth=0.5, mew=0.7,capsize = 1.5, ls='none')
    ax.errorbar(fit_wl[scu_mask], fit_flux[scu_mask], yerr=fit_err[scu_mask], marker='s', mec='limegreen', c='k',
                mfc='limegreen', ms=3,mew=0.7,elinewidth=0.5,capsize = 1.5, lw=0.6,ls='none')
    ax.errorbar(fit_wl[her_mask], fit_flux[her_mask], yerr=fit_err[her_mask], marker='*', mec='darkred', c='k',
                mfc='darkred', ms=4,mew=0.7,lw=0.6,elinewidth=0.5,capsize = 1.5, ls='none')
    ax.errorbar(fit_wl[alma_mask], fit_flux[alma_mask], yerr=fit_err[alma_mask], marker='p', mec='purple', c='k',
                mfc='purple', ms=4,mew=0.7, lw=0.6,elinewidth=0.5,capsize = 1.5,ls='none')
    ax.errorbar(fit_wl[long_mask], fit_flux[long_mask], yerr=fit_err[long_mask], marker='o', mec='orangered', c='k',
                mfc='w', ms=4,lw=0.6,mew=0.7,elinewidth=0.5,capsize = 1.5, ls='none')
    ax.errorbar(fit_wl[notfit_mask], fit_flux[notfit_mask], marker='s', mec='k', c='k', 
                ms=7,mew=0.5,mfc='none', ls='none')

    ax.set(xlabel=r'$\lambda$ [observed $\mu$m]', ylabel=r'$F_{\nu}$ [mJy]')
    ax.yaxis.set_major_locator(LogLocator(base=10,numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)*.1, numticks=20))
    ax.tick_params(axis='y', which='both',color='k')

    if comp is None:
        source_name = savename
        source_name = ' '.join(source_name.split('.pdf')[0].split('_')) # turn 'alma_1.pdf' into 'alma 1'
        ax.annotate(source_name,
                     (0.02, 0.94), xycoords='axes fraction')
            
        # add redshift annotation
        if n in speczs:
            ax.annotate(f"Specz: {np.round(z,3)}",
                    (0.02, 0.87), xycoords='axes fraction')
        elif z < 0.2:
            ax.annotate(f"z: {np.round(z,2)}",
                    (0.02, 0.87), xycoords='axes fraction')
        else:
            ax.annotate(f"Photz: {np.round(z,2)}",
                    (0.02, 0.87 ), xycoords='axes fraction')

    elif comp is not None:
        temp = savename.split('_')
        source_name = temp[0]+' '+temp[1]
        ax.annotate(source_name,
                     (0.02, 0.94), xycoords='axes fraction')
        otres = get_med_theta(d_gb)
        gores = get_med_theta(comp)
        
        beta_ot = round(otres[1,2], 2) # only if we use beta as free param
        T_ot = round(float(otres[1,1]), 1)

        ax.annotate(r'$\beta_{\rm thin}$'+f' = {beta_ot}',
            (0.02, 0.86), xycoords='axes fraction')  
        ax.annotate(r'$T_{\rm thin}$'+f' = {T_ot} K',
                    (0.02, 0.74), xycoords='axes fraction')
        beta_go = round(gores[1,2], 2) # only if we use beta as free param
        T_go = round(float(gores[1,1]), 1)        
        ax.annotate(r'$\beta_{\rm opac}$'+f' = {beta_go}',
            (0.02, 0.80), xycoords='axes fraction')  
        ax.annotate(r'$T_{\rm opac}$'+f' = {T_go} K',
                    (0.02, 0.68), xycoords='axes fraction')
        if n in speczs:
            ax.annotate(f"Specz: {z}", 
                    (0.02, 0.60), xycoords='axes fraction')
        elif z < 0.2:
            ax.annotate(f"z: {z}",
                    (0.02, 0.60), xycoords='axes fraction')
        else:
            ax.annotate(f"Photz: {z}",
                    (0.02, 0.60), xycoords='axes fraction')


    if create_ax is True:     
        if savename != None:
            print(f"Saving figure {savename} to {OUTDIR}")
            plt.savefig(OUTDIR+savename)
        if comp is None:
            plt.close(fig)
    
    elif show is True:
        plt.show()
        



def run_sed_plotter_obsfile(sed_infile, obs_infile, filt_infile, z, n=1, savename=None, op_thin = True, cig_infile=None, add_flux_file=None,fig=None,ax=None):
    print(f"\n\n\n*********** Running plot_sed.py with obs infile for galaxy {n}...  **************\n\n\n")
    t_sed = load_sed(sed_infile)
    bands = load_bands(filt_infile)
    t_flux = load_fluxes_obsfile(obs_infile, n=n)
    t_sed = add_sed_F_nu(t_sed, z)
    print()
    t_phot = get_phot_obsfile(bands, t_flux, add_flux_file=add_flux_file, n=n)
    print()

    d_gb = fit_greybody(t_phot,z,n=n, op_thin = op_thin) #added by S. McKay, June 2022

    if ax != None:
        p=plot_sed(t_sed, t_phot, savename=savename, d_gb=d_gb, n=n, z=z, fig=fig, ax = ax, op_thin = op_thin) # plot on multiplot axis

    p=plot_sed(t_sed, t_phot, savename=savename, d_gb=d_gb,n=n, z=z, op_thin = op_thin) # plot normal SED
    
    print()

    
    pars = get_mbb_params(d_gb, z=z,op_thin=op_thin)


    print(f'\n\n*********** Finished galaxy {n}. ***********\n\n')
    print(d_gb['sampler'].flatchain.shape)
    
    return pars, d_gb['sampler']
