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


    m1.fit(phot=([450, 850],[0.005, 0.0021],[0.0006,0.00032]),niter=10,params=['N','beta'],restframe=False)
    print('after fit:')
    print('850um flux', m1.eval(850,z=m1.z).value*1000,'mJy')
    print(f'log10 dust mass: {np.log10(m1.dust_mass.value):.2f}')
    print(m1.post_percentile('L'))
    print(np.median(m1.posterior('L')))

    try:
        print(m1.posterior('beta')[::2])
        print(m1.beta)
    except Exception as e:
        print(f'Caught exception {e}')

    print(m1.fit_result['sampler'], m1.fit_result['chi2'])

    print(m1._get_chain_for_parameter('L').shape, m1._get_chain_for_parameter('beta').shape)

    m1.plot_corner()
    plt.show()