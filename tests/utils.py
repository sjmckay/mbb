import numpy as np
from mbb import ModifiedBlackbody as MBB

SEED = 5825

def gen_phot(nbands=2, rand=True):
    if not (1 <= nbands <= 8):
        raise ValueError(f"must choose 1 to 8 bands, got {nbands}")
    bands = np.array([450,850,1100,2000,250,350,100,160])[:nbands]
    i = np.argsort(bands)
    bands = bands[i]
    flux = np.array([14, 7.3, 2.9, 0.6, 13, 15.5, 1.2, 3.8])[i]/1000
    err = np.array([2.8, 0.9, 0.5, 0.13, 2.4, 3.9, 1.1, 1.3])[i]/1000
    if rand: 
        rng = np.random.default_rng(SEED)
        flux = rng.normal(loc=flux, scale=err)
    return (bands, flux, err)


def default_mbb(pl=False, opthin=True, pl_piecewise=False, rand=False):
    defaults = dict(L=12,
                    T=35,
                    beta = 2.0,
                    z=2.5,
                    alpha=2.0,
                    l0=200.)
    if rand:
        rng = np.random.default_rng(SEED)
        for key, value in defaults.items():
            defaults[key] = value*(0.5+rng.random())
    m = MBB(**defaults,pl=pl,opthin=opthin,pl_piecewise=pl_piecewise)
    return m