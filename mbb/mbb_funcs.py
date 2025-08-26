
import numpy as np
import astropy.units as u
import astropy.constants as con
from astropy.constants import c, k_B, h
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.30) 

c=c.value
k_B = k_B.value
h = h.value


# DEF_BETA = 1.8 #default beta
# DEF_ALPHA = 2.0 #default alpha
# DEF_T = 50
Tcmb0 = 2.75

# note in all cases, lambda (l) is in microns and rest frame. These functions return model values in Janskys 

def ot_mbb(l, Nbb, beta, T, z,l0=200):
    Tcmbz = Tcmb0*(1+z)
    Tz = (T**(4+beta)  +  (Tcmb0)**(4+beta) * ((1+z)**(4+beta)-1) ) **(1/(4+beta))
    result = (1 - (planckbb(l,Tcmbz)/planckbb(l,Tz))) * (Tz/T)**(4+beta) \
            * 10.0**Nbb *(l0*1e-6/c)**beta * 2*h / c**2 * (c/(l*1.e-6))**(beta+3)/(np.exp(h*c/(l*1e-6*k_B*T))-1)
    return result

def go_mbb(l, Nbb, beta, T, z,l0=200):
    Tcmbz = Tcmb0*(1+z)
    Tz = (T**(4+beta)  +  (Tcmb0)**(4+beta) * ((1+z)**(4+beta)-1) ) **(1/(4+beta))
    ### general opacity version
    result = (1 - (planckbb(l,Tcmbz)/planckbb(l,Tz))) * (Tz/T)**(4+beta) \
                * 10.0**Nbb * 2*h / c**2 * ((1.0 - np.exp(-(l0/l)**beta))*(c/(l*1.e-6))**3.0)/(np.exp(h*c/(l*1.e-6*k_B*T))-1.0)
    return result

def ot_pl(l, Nbb, beta, T, alpha,l0=200):
    l_c = 0.75*(T*(alpha*7.243e-5 + 1.905e-4))**(-1)
    Npl = 10.0**Nbb *(l0*1e-6/c)**beta * 2*h / c**2 * (c/(l_c*1e-6))**(beta+3)\
            /(np.exp(h*c/(l_c*1e-6*k_B*T))-1.0)/((l_c*1e-6)**alpha)
    result = Npl*(l*1e-6)**alpha * np.exp(-(l/l_c)**2)
    return result

def go_pl(l, Nbb, beta, T, alpha, l0=200):
    l_c = 0.75*(T*(alpha*7.243e-5 + 1.905e-4)+ (alpha*6.246 + 26.68)**(-2))**(-1)
    Npl = 10.0**Nbb * 2*h / c**2 * ((1.0 - np.exp(-(l0/l_c)**beta)) * (c/(l_c*1e-6))**3)\
            /(np.exp(h*c/(l_c*1e-6*k_B*T))-1.0)/((l_c*1e-6)**alpha)
    result = Npl * (l*1e-6)**alpha * np.exp(-(l/l_c)**2)
    return result


def mbb_func(l, N=12,beta=1.8,T=35,z=0,alpha=2.0, l0=200, opthin=True, pl=False):
    """ MBB function with optional powerlaw and variable opacity assumptions

    Args:
        l (float): the rest-frame wavelengths at which to evaluate the model.
        N (float): log10 normalization factor of blackbody
        beta (float): emissivity index
        T (float): effective dust temperature
        z (float): the redshift of the model
        alpha (float): power-law slope
        l0 (float): turnover wavelength at which dust is optically thin
        opthin (bool): Whether or not the model should assume optically thin dust emission.
        pl (pool): Whether or not the model should include a MIR power law (as in Casey+ 2012)
    Returns:
        float: the value(s) of the model at wavelengths ``l``
    """
    if pl:
        if opthin: return ot_mbb(l, N,beta,T,z,l0=l0) + ot_pl(l,N,beta,T,alpha,l0=l0) 
        else: return go_mbb(l, N,beta,T,z,l0=l0) + go_pl(l,N,beta,T,alpha,l0=l0) 
    else:
        if opthin: return ot_mbb(l, N,beta,T,z,l0=l0) 
        else: return go_mbb(l, N,beta,T,z,l0=l0) 


def planckbb(l,T): # note although this requires wavlength in micron, it represents B_nu
    return 2*h / c**2 * (c/(l*1e-6))**3 / (np.exp(h*c/(l*1e-6*k_B*T))-1)
