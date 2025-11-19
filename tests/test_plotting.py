import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("Agg") #don't show plots
import unittest
from .utils import gen_phot, default_mbb


class TestPlots(unittest.TestCase):

    def test_no_data(self):
        #plot initial models
        m1 = default_mbb(pl=True, opthin=False, pl_piecewise=True, rand=True)
        m1.plot_sed()
        plt.title('m1')

        m2 = default_mbb(pl=True, opthin=False, pl_piecewise=False, rand=True)
        m2.plot_sed()
        plt.title('m2')
        plt.show(block=False)

        
    def test_w_data(self):
        phot = gen_phot(nbands=8)

        # plot models with data
        m3 = default_mbb(pl=True, opthin=True, pl_piecewise=True, rand=True)
        m3.fit(phot, niter=200, nwalkers=50, ncores=1, params=['L','T'])
        m3.plot_sed()
        plt.title('m3')

        m4 = default_mbb(pl=True, opthin=True, pl_piecewise=False, rand=True)
        m4.fit(phot, niter=200, nwalkers=50, ncores=1, params=['L','T'])
        m4.plot_sed()
        plt.title('m4')
        m4.plot_corner()
        plt.show(block=False)
