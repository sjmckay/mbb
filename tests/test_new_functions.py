import matplotlib.pyplot as plt
import numpy as np
from mbb import ModifiedBlackbody as MBB
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    
    m1 = MBB(z=2.5,L=12.6,T=35,beta=2.0,pl=True,opthin=False)
    # print('before fit')
    # print('850um flux', m1.eval(850,z=m1.z).value*1000,'mJy')
    # print(f'log10 dust mass: {np.log10(m1.dust_mass.value):.2f}')
    # print(f'peak wl: {m1.get_peak_wavelength():.2f}')


    # m1.fit(phot=([450, 850],[0.005, 0.0021],[0.0006,0.00032]),niter=10,params=['L','beta'],restframe=False)
    # print('after fit:')
    # print('850um flux', m1.eval(850,z=m1.z).value*1000,'mJy')
    # print('beta', m1.beta)

    # try:
    #     print(m1.posterior('beta')[::2])
    #     print(m1.beta)
    # except Exception as e:
    #     print(f'Caught exception {e}')

    # m1.plot_corner()
    # plt.show()
    # print('existing cosmo:',m1.cosmo)

    # print(f'log10 dust mass: {np.log10(m1.dust_mass.value):.6f}')
    # from astropy.cosmology import Planck18 as nc
    # m1.cosmo = nc
    # print('set new cosmo!')
    # print('new cosmo:',m1.cosmo)
  
    # print(f'log10 dust mass: {np.log10(m1.dust_mass.value):.6f}')


    print('*'*50)
    print('    NEW FIT    ')
    m1.reset()

    m1.fit(phot=([450, 850],[0.005, 0.0021],[0.0006,0.00032]),niter=10,params=['L','z'],restframe=False,priors = {'z':dict(mu=3.0,sigma=0.2)})
    print('after fit:')
    print('850um flux', m1.eval(850,z=m1.z).value*1000,'mJy')
    print('z', m1.z)
    print('phot wls',m1.phot[0]*(1+m1.z))
    m1.plot_sed(obs_frame=True)


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

    print('L chain', m1._get_chain_for_parameter('L',skip=200))
