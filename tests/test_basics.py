import numpy as np
import unittest
import astropy.units as u

from .utils import gen_phot, default_mbb


class TestBasicMBB(unittest.TestCase):
    def test_fit(self):
        m1 = default_mbb(pl=True, opthin=True)
        m2 = default_mbb(pl=False, opthin=True)
        m3 = default_mbb(pl=True, opthin=False)
        m4 = default_mbb(pl=False, opthin=False)
        m5 = default_mbb(pl=False, opthin=False, pl_piecewise=True)
        m6 = default_mbb(pl=True, opthin=False, pl_piecewise=True)

        phot = gen_phot(nbands=2)
        for m in [m1,m2,m3,m4,m5,m6]:
            m.fit(phot, nburn=300, niter=400, params=['L','T'])
            f850 = phot[1][1]
            e850 = phot[0][1]
            loss = np.abs(f850-m.eval(850).value)/e850
            self.assertLess(loss, 3.0) #within 3sig of flux
        
    def test_variations(self):
        m1 = default_mbb(pl=False, opthin=True, rand=True)
        m2 = default_mbb(pl=True, opthin=False, pl_piecewise=True, rand=True)

        phot = gen_phot(nbands=6)
        for m in [m1,m2]:
            m.fit(phot, nburn=400, niter=600,params=['L','beta','T'])
            f850 = phot[1][1]
            e850 = phot[0][1]
            loss = np.abs(f850-m.eval(850).value)/e850
            self.assertLess(loss, 3.0) #within 3sig of flux
    
    def test_uplims(self):
        m1 = default_mbb(pl=False, opthin=True, rand=True)
        m2 = default_mbb(pl=True, opthin=False, pl_piecewise=True, rand=True)

        phot = gen_phot(nbands=3)
        for m in [m1,m2]:
            m.fit(phot, uplims = [True, False,False], nburn=400, niter=600, params=['L','T'])
            f450 = phot[1][0]
            e450 = phot[0][0]
            self.assertLessEqual(m.eval(450).value, f450+3*e450)

    def test_luminosity(self):
        m1 = default_mbb(pl=False, opthin=True, rand=False)
        self.assertAlmostEqual(np.log10(m1.get_luminosity().value), m1.L, places=1)

    def test_dust_mass(self):
        m1 = default_mbb(pl=False, opthin=True, rand=True)
        self.assertIsInstance(m1.dust_mass, u.Quantity)

    def test_peak_wavelength(self):
        m1 = default_mbb(pl=False, opthin=True, rand=True)
        self.assertIsInstance(m1.get_peak_wavelength(), u.Quantity)

    def test_priors(self):
        m1= default_mbb()
        phot=gen_phot()
        m1.fit(phot=phot,niter=100,params=['L','z'],
               restframe=False,priors = {'z':dict(mu=4.0,sigma=0.2)})
        loss = np.abs(m1.z - 4.0)
        self.assertLess(loss, 0.8)  #4-sigma

    def test_posteriors(self):
        niter = 300
        nwalkers = 100
        sample_by = 10
        m1 = default_mbb(pl=False, opthin=True, rand=True)
        phot = gen_phot(nbands=6)
        fit = m1.fit(phot, uplims = (phot[0]<200), nwalkers=nwalkers, niter=niter, nburn=400, ncores=12, params=['L','beta','T'])
        beta_q = m1.post_percentile('beta', q=[16,50,84])
        for b in beta_q:
            self.assertGreater(b, 0)
            self.assertLess(b, 5.0)
        frac_loss = np.abs(beta_q[1]-m1.beta)/beta_q[1]
        self.assertLess(frac_loss, 0.01)
        T_chain = m1.posterior('T', sample_by=sample_by)
        self.assertEqual(len(T_chain), niter*nwalkers//sample_by)
        self.assertAlmostEqual(np.median(T_chain), np.squeeze(m1.post_percentile('T', q=[50], sample_by=sample_by)), places=1)