import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import freeze_support

from .utils import gen_phot, default_mbb

if __name__ == "__main__":
    freeze_support()
    
    m1 = default_mbb()
    print('before fit')
    print('850um flux', m1.eval(850,z=m1.z).value*1000,'mJy')
    print(np.log10(m1.dust_mass.value))
    print(f'log10 dust mass: {np.log10(m1.dust_mass.value):.2f}')
    print(f'peak wl: {m1.get_peak_wavelength():.2f}')

    phot = gen_phot(nbands=8,rand=False)
    m1.fit(phot=phot, uplims=(phot[1]/phot[2]<3), niter=1000,params=['L','beta'],restframe=False)
    print('after fit:')
    print('850um flux', m1.eval(850,z=m1.z).value*1000,'mJy')

    print('beta post', m1.post_percentile('beta',[50]))
    print('beta', m1.beta)
    m1.plot_sed()
    plt.show(block=False)

    m1.plot_corner()
    print('existing cosmo:',m1.cosmo)

    m1 = default_mbb(pl=False, opthin=True, rand=True)
    m2 = default_mbb(pl=True, opthin=False, pl_piecewise=True, rand=True)

    phot = gen_phot(nbands=3)
    for m in [m1,m2]:
        m.fit(phot, uplims = [True, False,False], nburn=400, niter=600, params=['L','T'])
        f450 = phot[1][0]
        m.plot_sed(obs_frame=True)


    def my_prior_T(T):
        return np.exp(-(T - 15.0) ** 2 / (2 * 10 ** 2))
    
    m1.reset()
    m1.update(z=2.0)
    m1.fit(phot=([450, 850],[0.005, 0.0021],[0.0006,0.00032]),niter=10,params=['L','T'],restframe=False,priors = {'T':my_prior_T})
    print('after fit:')
    print('850um flux', m1.eval(850,z=m1.z).value*1000,'mJy')
    print('T', m1.T)
    print('L', m1.L)
    print('z', m1.z)
    print('phot wls',m1.phot[0]*(1+m1.z))

    m1.plot_corner()
    m1.plot_sed(obs_frame=True)
    plt.show()

    print('L chain', m1._get_chain_for_parameter('L',sample_by=200))
