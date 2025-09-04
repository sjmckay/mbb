import matplotlib.pyplot as plt
import numpy as np
from mbb import ModifiedBlackbody as MBB
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    
    m1 = MBB(z=2.5,L=12.6,T=50,beta=2.0,pl=True,opthin=False)
    print('before fit')
    print('850um flux', m1.eval(850,z=m1.z).value*1000,'mJy')
    print(f'log10 dust mass: {np.log10(m1.dust_mass.value):.2f}')
    print(f'peak wl: {m1.get_peak_wavelength():.2f}')


    m1.fit(phot=([450, 850],[0.005, 0.0021],[0.0006,0.00032]),niter=50,to_vary=['N','beta'],restframe=False)
    print('after fit:')
    print('850um flux', m1.eval(850,z=m1.z).value*1000,'mJy')
    print(f'log10 dust mass: {np.log10(m1.dust_mass.value):.2f}')
    print(m1.posterior('N'))

    print(f'ndof, nbands, chi2: {m1.n_dof}, {m1.n_bands}, {m1.chi2}')
    try:
        print(m1.posterior('beta',q=50))
        print(m1.beta)
    except Exception as e:
        print(f'Caught exception {e}')

    m1.plot_sed(obs_frame=True)
    plt.title('m1')
    plt.show()

