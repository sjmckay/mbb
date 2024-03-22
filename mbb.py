### Class to implement a modified blackbody SED, with the capability to fit the SED with various options
# and plot a simple version of it, as well as save the results to a file. 
# Author: Stephen McKay, Spring 2023


# Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sed_py.mpfit.mpfit import mpfit

from astropy.table import Table, QTable
from astropy.io import fits
import astropy.units as u
import astropy.constants as con
from astropy.constants import c, k_B, h
from astropy.cosmology import FlatLambdaCDM

c = c.value
k_B = k_B.value
h = h.value

DEF_BETA = 1.8 #default beta
DEF_ALPHA = 2.0 #default alpha


# Class definition

class mbb:


	def __init__(self, name, L, T, beta, z, t_phot = None, limits=None, pl=False, opthin=True, verbose = False):
		"""return an object of class mbb which is initialized to the input parameters.
			If tphot is provided, then fits best-fit mbb and treats N, beta, and T as initial values.
			Limits can also be provided on each parameter, as an array of tuples (X,Y) where X is the lower limit and Y the upper 
			limit on each parameter. """
		
		self.name = name #string giving name of the fit
		self.pl = pl # bool determining whether this mbb includes a MIR power law
		self.opthin = opthin # bool determining whether this mbb assumes optically-thin dust emission
		self.z = z # redshift
		
		if t_phot is not None:
			if t_phot.shape[0] < 3: # if less than 3 rows
				raise ValueError(f"t_phot has shape {t_phot.shape} but needs to have shape (3, N).")
			self.t_phot = np.array(t_phot) # photometry table (3xN): first row: wavelength of observation, 
										   # second row: fluxes in Jy, third row: error in Jy
		else:
			self.t_phot = None	
		
		self.limits = limits # limits on parameters, provided as list or array of tuples of upper and lower limits. If not included,
							 # the parameters are unbounded.

		self.verbose = verbose
		
		# # if photometry is passed in, fit mbb for parameters
		# if t_phot is not None:
		# 		self.fit(initvals = np.array([N, T, beta]))	
		# 		if self.verbose == True:
		# 			self.print_status()
		# otherwise just take the ones passed in as the parameters
		# else:


		self.N = 11 # log of normalization of blackbody
		self.beta = beta # emissivity spectral index
		self.T = T # dust temperature in K
		self.beta_err = 0 # error in beta (zero if not fitting mbb)
		self.T_err = 0 # error in T (zero if not fitting mbb)
		self.fixed = [True,True,True] # array of bools denoting if the parameters [N, T, beta] are fixed
		self.mp_mbb = None # mpfit object that is only initialized if the mbb is fit to photometry
		
		Lcurr = np.log10(self.get_LIR((8,1000)).value)
		while((Lcurr > (L+0.001)) | (Lcurr < (L-0.001))):
			self.N = self.N * (L/Lcurr)
			Lcurr = np.log10(self.get_LIR((8,1000)).value)

			
	# def fit(self, initvals = np.array([-11, 35, 1.8])):
	# 	"""
	# 	Fit a modified blackbody to rest-frame photometry in Janskys, wavelengths in microns.
	# 	Returns a mbb instance with the best-fit parameters of the fit.
	# 	"""
	# 	#initialize fitting arrays
	# 	fit_wl = self.t_phot[0][:]/(1+self.z) # convert wavelengths to rest-frame before fitting
	# 	fit_flux = self.t_phot[1][:]
	# 	fit_err = self.t_phot[2][:]
		
	# 	# check for nondetections and or incorrect input
	# 	mask = (fit_wl < 0) | (fit_flux < 0) | (fit_err < 0)
	# 	fit_wl = fit_wl[~mask]
	# 	fit_flux = fit_flux[~mask]
	# 	fit_err = fit_err[~mask]

	# 	# initialize several mbb variables
	# 	limits = self.limits
	# 	opthin = self.opthin
	# 	pl = self.pl

	# 	#if we have more than 3 points, we can fit for beta as well.
	# 	if len(fit_wl) > 3:
	# 		p0 = initvals
	# 		if limits is not None:
	# 			try:
	# 				parinfo = [{'value':p0[0],'fixed':0, 'limited':[1,1], 'limits':limits[0]},	
	# 				{'value':p0[1],'fixed':0, 'limited':[1,1], 'limits':limits[1]},
	# 				{'value':p0[2],'fixed':0, 'limited':[1,1], 'limits':limits[2]}]
	# 			except:
	# 				raise ValueError(f"The limits {limits} provided did not have the correct format.")
	# 		else: 
	# 			parinfo = [{'value':p0[0],'fixed':0},{'value':p0[1],'fixed':0},{'value':p0[2],'fixed':0}]
	# 	else:
	# 		p0 = initvals[0:2]   #just N and T
	# 		if limits is not None:
	# 			try:
	# 				parinfo = [{'value':p0[0],'fixed':0, 'limited':[1,1], 'limits':limits[0]},	
	# 						{'value':p0[1],'fixed':0, 'limited':[1,1], 'limits':limits[1]}]
	# 			except:
	# 				raise ValueError(f"The limits {limits} provided did not have the correct format.")
	# 		else: 
	# 			parinfo = [{'value':p0[0],'fixed':0},{'value':p0[1],'fixed':0}]

	# 	fkw = {'x':fit_wl, 'y':fit_flux, 'err':fit_err}

	# 	# choose which deviates function to pass to mpfit
	# 	if opthin is True:
	# 		if pl is True:
	# 			to_fit = mbb_dev_ot_pl		
	# 		else:	
	# 			to_fit = mbb_dev_ot
	# 	else:
	# 		if pl is True:
	# 			to_fit = mbb_dev_go_pl
	# 		else:
	# 			to_fit = mbb_dev_go
		
	# 	if self.verbose == True:
	# 		quiet = 0
	# 	else:
	# 		quiet = 1

	# 	# fit best-fit MBB using levenberg-marquardt algorithm in mpfit
	# 	m = mpfit(to_fit,parinfo=parinfo, functkw=fkw, xtol = 1e-14, ftol=1e-12,gtol=1e-14, nocovar=0, debug=0, quiet=quiet)
		
	# 	# initialize rest of class variables based on results of fit
	# 	self.N = m.params[0]
	# 	self.T = m.params[1]
	# 	try:
	# 		self.T_err = m.perror[1]
	# 	except:
	# 		self.T_err = -1

	# 	self.fixed = [False,False,True]

	# 	if len(m.params) > 2:
	# 		self.beta = m.params[2]
	# 		try:
	# 			self.beta_err = m.perror[2]
	# 		except:
	# 			self.beta_err = -1
	# 		self.fixed[2] = False
	# 	else:
	# 		self.beta = DEF_BETA
	# 		self.beta_err = -1
	# 	self.mp_mbb = m


	def print_status(self):
		# print out current values of mbb parameters and fit stats
		if self.mp_mbb is not None:
			m = self.mp_mbb
			print('Best-fit values for MBB parameters are:')
			print('log(N):',self.N)
			print('beta:',self.beta)
			print('T:',self.T)
			print('# of iterations:', m.niter)
			print('err msg:', m.errmsg)
			print('status:', m.status)
			print()
		else:
			print('No photometry was provided. Initialized MBB parameters are:')
			print('log(N):',self.N)
			print('beta:',self.beta)
			print('T:',self.T)
			print()
		

	def save(self, filename):
		# write string version of fit to file
		with open(filename, 'a') as f: #open in append mode
			f.writelines([self.name,'\t', self.N, '\t', self.T, '\t', self.beta, '\t',self.pl, '\t', self.opthin])

	def plot(self,obs_frame=False):
		# plot this mbb just for basic visualization. It is recommended to use a separate, more detailed plotting function for figures.
		fig, ax = plt.subplots()
		x = np.logspace(0.5,3,200)
		y = 1000 * self.eval(x)
		
		if obs_frame == True:
			x *= (1 + self.z)
			ax.set(xlabel = r'$\lambda$ observed-frame [$\mu$m]', ylabel = 'Flux [mJy]')
		else:
			ax.set(xlabel = r'$\lambda$ rest-frame [$\mu$m]', ylabel = 'Flux [mJy]')

		ax.plot(x,y, ls='--',linewidth=1.0)
		
		
		if self.t_phot is not None:
			#initialize fitting arrays
			if obs_frame == True:
				fit_wl = self.t_phot[0,:]
			else:
				fit_wl = self.t_phot[0,:] / (1+self.z)

			fit_flux = 1000*self.t_phot[1,:] #mJy
			fit_err = 1000*self.t_phot[2,:]
		
			# check for nondetections and or incorrect input
			mask = (fit_wl < 0) | (fit_flux < 0) | (fit_err < 0)
			fit_wl = fit_wl[~mask]
			fit_flux = fit_flux[~mask]
			fit_err = fit_err[~mask]



			ax.errorbar(fit_wl, fit_flux, fit_err, 
						c='r', ls='', marker = 'o', ms = 5,
						elinewidth=0.5, capsize = 1.5, ecolor = 'k'
						)

		ax.set(xscale='log', yscale='log')
		ax.set(xlim = (x.min(), x.max()*1.1), ylim=(1e-1,2e2))

		ax.annotate(self.name, xy=(0.02, 0.95), xycoords = 'axes fraction')
		ax.annotate(f'z = {np.round(self.z,2)}', xy=(0.02, 0.90), xycoords = 'axes fraction')
		ax.annotate(f'beta = {np.round(self.beta,2)}', xy=(0.02, 0.85), xycoords = 'axes fraction')
		ax.annotate(f'T = {np.round(self.T,1)} K', xy=(0.02, 0.80), xycoords = 'axes fraction')

		return fig, ax
	

	def eval(self, wl,z=0):
		"""Return evaulation of this MBB's function if observed at the given wavelengths wl shifted to redshift z in Jy
		Leave z=0 to get rest-frame evaluation."""
		p = [self.N,self.T,self.beta]
		return mbb_fun(p, wl/(1+z), opthin=self.opthin, pl=self.pl,z=z)


	def get_LIR(self, wllimits, cosmo=FlatLambdaCDM(H0=70.0, Om0=0.30)):
		"""get integrated LIR luminosity between wl limits in microns"""
		if len(wllimits) == 2 and wllimits[0] < wllimits[1]:
		    nulow = (con.c/(wllimits[1]*u.um)).to(u.Hz)
		    nuhigh = (con.c/(wllimits[0]*u.um)).to(u.Hz)
		    nu = np.linspace(nulow,nuhigh,20000)
		    
		    dnu = nu[1] - nu[0]
		    DL = cosmo.luminosity_distance(self.z)
		    lam = (con.c/nu).to(u.um).value  
		    lum = np.sum(4*np.pi*DL**2 * self.eval(lam)*u.Jy * dnu)/(1+self.z)
		    return lum.to(u.Lsun)


def planckbb(l,T): # note although this requires wavlength in micron, it represents B_nu
	return 2*h / c**2 * (c/(l*1e-6))**3 / (np.exp(h*c/(l*1e-6*k_B*T))-1)

def mbb_fun(p,l, opthin = True, pl=False,z=0):
	""" coupled greybody/powerlaw function based on Casey et al. 2012
		note p is parameters array (N,T,beta), and l is wavelength (lambda) in microns.
		Returns greybody function values (at wavelength points l) in Jy.
		Uses log of normalization as Nbb parameter in order to be roughly the same order of magnitude as T, beta.
	"""
	#assign variable parameter values
	Nbb = p[0] # norm constant for greybody (gb)
	T = p[1] # temperature of gb in K

	if len(p) > 2: 
		beta = p[2] # emissivity index (set to p[2] if enough data points in FIR)
	else:
		beta = DEF_BETA
	
	# if including a MIR power law
	if pl:
		alpha = DEF_ALPHA # powerlaw slope
		l_c = 0.75*(T*(alpha*7.243e-5 + 1.905e-4)+ (alpha*6.246 + 26.68)**(-2))**(-1) #critical wavelength (microns) where MIR powerlaw turns over (approx. from casey2012)
	
	Tcmb0 = 2.73
	Tcmbz = Tcmb0*(1+z)
	Tz = (T**(4+beta)  +  (Tcmb0)**(4+beta) * ((1+z)**(4+beta)-1) ) **(1/(4+beta))
	### optically thin version
	if opthin is True:
		result = (1 - (planckbb(l,Tcmbz)/planckbb(l,Tz))) * (Tz/T)**(4+beta) * 10.0**Nbb *(200e-6/c)**beta * 2*h / c**2 * (c/(l*1e-6))**(beta+3)/(np.exp(h*c/(l*1e-6*k_B*T))-1)
	else: 
	### general opacity version
		l0=200.
		result = (1 - (planckbb(l,Tcmbz)/planckbb(l,Tz))) * (Tz/T)**(4+beta) * 10.0**Nbb * 2*h / c**2 * ((1.0 - np.exp(-(l0/l)**beta))*(c/(l*1.e-6))**3.0)/(np.exp(h*c/(l*1.e-6*k_B*T))-1.0) #+ Npl*(l*1e-6)**alpha * np.exp(-(l/l_c)**2) #uncomment second term to add in powerlaw
	
	return result

# # possible deviates functions
# def mbb_dev_ot(p,fjac=None,x=None,y=None,err=None):
# 	"""
# 	deviates function required for mpfit (optically thin)
# 	returns [status, (model- y)/err] where status is non-negative if fit should continue, and negative if STOP
# 	"""
# 	status = 0

# 	return [status, (y - mbb_fun(p=p,l=x,opthin = True))/err]

# def mbb_dev_go(p,fjac=None,x=None,y=None,err=None):
# 	"""
# 	deviates function required for mpfit (general opacity)
# 	returns [status, (model- y)/err] where status is non-negative if fit should continue, and negative if STOP
# 	"""
# 	status = 0

# 	return [status, (y - mbb_fun(p=p,l=x,opthin =False))/err]

# def mbb_dev_ot_pl(p,fjac=None,x=None,y=None,err=None):
# 	"""
# 	deviates function required for mpfit (optically thin, w power law)
# 	returns [status, (model- y)/err] where status is non-negative if fit should continue, and negative if STOP
# 	"""
# 	status = 0

# 	return [status, (y - mbb_fun(p=p,l=x,opthin = True))/err]

# def mbb_dev_go_pl(p,fjac=None,x=None,y=None,err=None,pl=True):
# 	"""
# 	deviates function required for mpfit (general opacity, w power law)
# 	returns [status, (model- y)/err] where status is non-negative if fit should continue, and negative if STOP
# 	"""
# 	status = 0

# 	return [status, (y - mbb_fun(p=p,l=x,opthin =False,pl=True))/err]