from mbb import ModifiedBlackbody as MBB
from multiprocessing import freeze_support
import time
from .utils import gen_phot, default_mbb
import unittest


class TestPerformance(unittest.TestCase):

    def test_post_L(self):
        # from cProfile import Profile
        # from pstats import SortKey, Stats
        m = default_mbb(pl=True, opthin=True, pl_piecewise=False)
        phot = gen_phot(nbands=7)

        fit = m.fit(phot, nburn=500, niter=1000)
        # with Profile() as profile:
        start = time.time()
        percent = m.post_percentile('L',q=(16,50,84),sample_by=10)
        end = time.time()
        self.assertLess(end-start, 60) #less than 1 min to compute post percentiles
    
    def test_second_fit_speed_scale(self):
        m1 = default_mbb(pl=True, opthin=True, pl_piecewise=False)
        phot = gen_phot(nbands=7)
        start1 = time.time()
        fit = m1.fit(phot, nburn=500, niter=1000, ncores=4)
        end1 = time.time()

        m2 = default_mbb(pl=True, opthin=True, pl_piecewise=False)
        start2 = time.time()
        fit = m2.fit(phot, nburn=500, niter=2000,ncores=4)
        end2 = time.time()
        self.assertLess(end2-start2, 120) #shouldn't take extra long to fit second time
        self.assertLess((end2-start2)/(end1-start1), 4) #should scale as less than n**2
