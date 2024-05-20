
import numpy as np
import astropy.units as u
import astropy.constants as con
from astropy.constants import c, k_B, h
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.30) 

c=c.value
k_B = k_B.value
h = h.value


DEF_BETA = 1.8 #default beta
DEF_ALPHA = 2.0 #default alpha
Tcmb0 = 2.75
l0=200

# note in all cases, lambda (l) is in microns and rest frame. These functions return model values in Janskys 

def ot_mbb(Nbb, beta, T, l, z):
    Tcmbz = Tcmb0*(1+z)
    Tz = (T**(4+beta)  +  (Tcmb0)**(4+beta) * ((1+z)**(4+beta)-1) ) **(1/(4+beta))
    result = (1 - (planckbb(l,Tcmbz)/planckbb(l,Tz))) * (Tz/T)**(4+beta) \
            * 10.0**Nbb *(l0*1e-6/c)**beta * 2*h / c**2 * (c/(l*1.e-6))**(beta+3)/(np.exp(h*c/(l*1e-6*k_B*T))-1)
    return result

def go_mbb(Nbb, beta, T, l, z):
    Tcmbz = Tcmb0*(1+z)
    Tz = (T**(4+beta)  +  (Tcmb0)**(4+beta) * ((1+z)**(4+beta)-1) ) **(1/(4+beta))
    ### general opacity version
    result = (1 - (planckbb(l,Tcmbz)/planckbb(l,Tz))) * (Tz/T)**(4+beta) \
                * 10.0**Nbb * 2*h / c**2 * ((1.0 - np.exp(-(l0/l)**beta))*(c/(l*1.e-6))**3.0)/(np.exp(h*c/(l*1.e-6*k_B*T))-1.0)
    return result

def ot_pl(Nbb, beta, T, alpha, l,z):
    l_c = 0.75*(T*(alpha*7.243e-5 + 1.905e-4))**(-1)
    Npl = 10.0**Nbb *(l0*1e-6/c)**beta * 2*h / c**2 * (c/(l_c*1e-6))**(beta+3)\
            /(np.exp(h*c/(l_c*1e-6*k_B*T))-1.0)/((l_c*1e-6)**alpha)
    result = Npl*(l*1e-6)**alpha * np.exp(-(l/l_c)**2)
    return result

def go_pl(Nbb, beta, T, alpha, l,z):
    l_c = 0.75*(T*(alpha*7.243e-5 + 1.905e-4)+ (alpha*6.246 + 26.68)**(-2))**(-1)
    Npl = 10.0**Nbb * 2*h / c**2 * ((1.0 - np.exp(-(l0/l_c)**beta)) * (c/(l_c*1e-6))**3)\
            /(np.exp(h*c/(l_c*1e-6*k_B*T))-1.0)/((l_c*1e-6)**alpha)
    result = Npl * (l*1e-6)**alpha * np.exp(-(l/l_c)**2)
    return result


def mbb_fun_ot_pl(theta,l,z=0):
    """ MBB function with powerlaw and optically thin assumption
    """
    Nbb = theta[0] # norm constant for greybody (gb)
    T = theta[1] # temperature of gb in K
    if len(theta) > 2: 
        beta = theta[2] # emissivity index (set to p[2] if enough data points in FIR)
    else:
        beta = DEF_BETA
    alpha = DEF_ALPHA # powerlaw slope
    return ot_mbb(Nbb,beta,T,l,z) + ot_pl(Nbb,beta,T,alpha,l,z) 
    
def mbb_fun_go_pl(theta,l,z=0):
    """ MBB function with powerlaw and general opacity assumption
    """
    Nbb = theta[0] # norm constant for greybody (gb)
    T = theta[1] # temperature of gb in K
    if len(theta) > 2: 
        beta = theta[2] # emissivity index (set to p[2] if enough data points in FIR)
    else:
        beta = DEF_BETA
    alpha = DEF_ALPHA # powerlaw slope
    return go_mbb(Nbb,beta,T,l,z) + go_pl(Nbb,beta,T,alpha,l,z) 

def mbb_fun_ot(theta,l,z=0):
    """ MBB function with no powerlaw and optically thin assumption
    """
    Nbb = theta[0] # norm constant for greybody (gb)
    T = theta[1] # temperature of gb in K

    if len(theta) > 2: 
        beta = theta[2] # emissivity index (set to p[2] if enough data points in FIR)
    else:
        beta = DEF_BETA
    return ot_mbb(Nbb,beta,T,l,z)

def mbb_fun_go(theta,l,z=0):
    """ MBB function with no powerlaw and general opacity assumptions
    """
    Nbb = theta[0] # norm constant for greybody (gb)
    T = theta[1] # temperature of gb in K
    if len(theta) > 2: 
        beta = theta[2] # emissivity index (set to p[2] if enough data points in FIR)
    else:
        beta = DEF_BETA
    return go_mbb(Nbb,beta,T,l,z)


def planckbb(l,T): # note although this requires wavlength in micron, it represents B_nu
    return 2*h / c**2 * (c/(l*1e-6))**3 / (np.exp(h*c/(l*1e-6*k_B*T))-1)
