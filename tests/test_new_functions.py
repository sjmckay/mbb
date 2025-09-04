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
    # m1.plot_sed(obs_frame=True)
    # plt.title('m1')

    m1.fit(phot=([850/(1+m1.z)],[0.0021],[0.00032]),niter=10,to_vary=['N'],restframe=True)
    print('after fit:')
    print('850um flux', m1.eval(850,z=m1.z).value*1000,'mJy')
    print(f'log10 dust mass: {np.log10(m1.dust_mass.value):.2f}')
    print(m1.posterior('N'))
    try:
        print(m1.posterior('beta'))
    except Exception as e:
        print(f'Caught exception {e}')

