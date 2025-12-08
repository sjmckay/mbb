import unittest
import os 
import astropy.units as u
from mbb import ModifiedBlackbody as MBB
import numpy as np
from .utils import gen_phot, default_mbb


class TestPickleMBB(unittest.TestCase):
    def test_pickle_basic(self):
        m1 = default_mbb(pl=True, opthin=True)
        filename = 'test_basic.pickle'
        m1.save(filename)
        m2 = MBB.from_file(filename)
        self.assertIsInstance(m2, MBB)
        self.assertEqual(m2.beta, m1.beta)
        self.assertEqual(m2.L,m1.L)
        self.assertEqual(m2.model.__repr__(), m1.model.__repr__())
        os.remove(filename)

    
    def test_pickle_fit(self):
        m1 = default_mbb(pl=False, opthin=False, pl_piecewise=True)
        phot = gen_phot(nbands=3)
        m1.fit(phot, nburn=300, niter=400, params=['L','T'])
        filename = 'test_fit.pickle'
        m1.save(filename)
        m2 = MBB.from_file(filename)
        self.assertIsInstance(m2, MBB)
        self.assertEqual(m2.beta, m1.beta)
        self.assertTrue(np.all(m2.fit_result['sampler'].get_chain(flat=True)[0:10] == m1.fit_result['sampler'].get_chain(flat=True)[0:10]))
        self.assertEqual(m2.model.__repr__(), m1.model.__repr__())
        os.remove(filename)
        
    def test_pickle_full_process(self):
        m1 = default_mbb(pl=False, opthin=False, pl_piecewise=True)
        phot = gen_phot(nbands=5)
        m1.fit(phot, nburn=300, niter=400, nwalkers=100, params=['L','T','beta'])
        filename = 'test_full.pickle'
        m1.save(filename)
        m2 = MBB.from_file(filename)
        beta_q = m2.post_percentile('beta', q=[16,50,84])
        for b in beta_q:
            self.assertGreater(b, 0)
            self.assertLess(b, 5.0)
        frac_loss = np.abs(beta_q[1]-m2.beta)/beta_q[1]
        self.assertLess(frac_loss, 0.01)
        T_chain = m1.posterior('T', sample_by=10)
        self.assertEqual(len(T_chain), 400*100//10)
        self.assertAlmostEqual(np.median(T_chain), np.squeeze(m1.post_percentile('T', q=[50], sample_by=10)), places=1)
        self.assertIsInstance(m1.get_peak_wavelength(), u.Quantity)
        self.assertIsInstance(m1.dust_mass, u.Quantity)
        os.remove(filename)

    
if __name__ == '__main__':
    unittest.main()