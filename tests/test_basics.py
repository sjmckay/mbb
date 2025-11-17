import matplotlib.pyplot as plt
import numpy as np
from mbb import ModifiedBlackbody as MBB
from multiprocessing import freeze_support

if __name__ == "__main__":
    
    test = np.zeros(8) 
    test[[2]] = True #change to run different tests
    freeze_support()
    
    if test[0]:
        m1 = MBB(z=4,L=12,T=45,beta=2.0,pl=False,opthin=False)
        m1.fit(phot=([850/5.],[0.0031],[0.00032]),niter=10,params=['L'],restframe=True)
        print(m1.eval(850,z=4).value*1000,'mJy')
        m1.plot_sed(obs_frame=True)
        plt.title('m1')

    if test[1]:
        m2 = MBB(z=4,L=12,T=45,beta=2.0,pl=False,opthin=True)
        m2.fit(phot=([450/5., 850/5.],[0.0016, 0.0021],[0.00212, 0.00032]), uplims = [True,False],niter=1000, params=['L','T'],restframe=True)
        print(m2.eval(850,z=4).value*1000,'mJy')
        m2.plot_sed(obs_frame=True)
        plt.title('m2')

    if test[2]:
        m3 = MBB(z=2.0,L=12,T=45,beta=2.0,pl=False,opthin=False)
        # m3.plot_sed(obs_frame=True)
        m3.fit(phot=([850],[0.0021],[0.0012]), niter=1000,params=['L'], uplims=True, priors={'L':{'mu':12,'sigma':0.6}})
        print(m3.eval(850,z=2.5).value*1000,'mJy')
        m3.plot_sed(obs_frame=True)
        plt.title('m3')

    if test[3]:
        m4 = MBB(z=4,L=12,T=45,beta=2.0,pl=True,opthin=True)
        m4.plot_sed(obs_frame=True)
        m4.fit(phot=([850/5.],[0.0021],[0.00032]),niter=10,params=['L'],restframe=True)
        print(m4.eval(850,z=4).value*1000,'mJy')
        m4.plot_sed(obs_frame=True)
        plt.title('m4')

    if test[4]:
        m5 = MBB(z=2.5,L=11,T=30,beta=1.5,pl=True,opthin=False)
        print(m5.eval(850,z=4).value*1000,'mJy')
        m5.plot_sed(obs_frame=False)
        plt.title('m5')

    if test[5]:
        m6 = MBB(z=2,L=12,T=45,beta=2.0,pl=False,opthin=True)
        m6.plot_sed(obs_frame=True)
        plt.title('m6')

    if test[6]:
        m7 = MBB(z=2,L=12,T=45,beta=2.0,pl=False,opthin=True)
        m7.fit(phot=([160/3., 250/3.,  850/3., 1200/3.],[0.0025,0.0041,0.0028,0.0007],[0.00032,0.0012,0.00032,0.00032]), uplims=[True, True, False, False],niter=400,params=['N','T','beta'],restframe=True)
        print(m7.eval(850,z=2).value*1000,'mJy')
        m7.plot_sed(obs_frame=True)
        plt.title('m7')

    if test[7]:
        m8 = MBB(z=2,L=12,T=45,beta=2.0,pl=True,opthin=False)
        m8.fit(phot=([160/3., 250/3.,  850/3., 1200/3.],[0.0015,0.0035,0.0021,0.0007],[0.00032,0.0009,0.00032,0.00032]),niter=100,params=['N','T','z'],restframe=True)
        m8.plot_sed(obs_frame=True)
        plt.title('m8')
        m8.plot_corner()

    plt.show()