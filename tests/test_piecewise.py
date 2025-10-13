from mbb import ModifiedBlackbody as MBB
import matplotlib.pyplot as plt

#plot initial models
m1 = MBB(L=12, T=45, z=2.5, beta=1.8, alpha=2.0, pl=True, opthin=False, pl_piecewise=True)
m1.plot_sed()
plt.title('m1')

m2 = MBB(L=12, T=45, z=2.5, beta=1.8, alpha=2.0, pl=True, opthin=False, pl_piecewise=False)
m2.plot_sed()
plt.title('m2')

#test photometry
phot = ([100, 250, 450, 850, 1100],
    [0.002, 0.010, 0.007,0.003, 0.001],
    [0.0009,0.0036, 0.0015,0.0005,0.0004])

# plot models with data
m3 = MBB(L=12, T=45, z=2.5, beta=1.8, alpha=4.0, pl=True, opthin=True, pl_piecewise=True)
m3.fit(phot, niter=100, nwalkers=50, ncores=1, params=['L','T'])
m3.plot_sed()
plt.title('m3')

m4 = MBB(L=12, T=45, z=2.5, beta=1.8, alpha=3.0, pl=True, opthin=True, pl_piecewise=False)
m4.fit(phot, niter=100, nwalkers=50, ncores=1, params=['L','T'])
m4.plot_sed()
print(m4.eval(850))
plt.title('m4')

#single wavelength point
from mbb.mbb_funcs import mbb_func
l=150.
print(mbb_func(l, N=12,beta=1.8,T=35,z=0,
    alpha=4.0, l0=200, 
    opthin=False, pl=True))


plt.show()
