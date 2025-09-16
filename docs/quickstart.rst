.. _quickstart:

Quickstart
========================

Once you have installed ``mbb``, import it in a python script:

.. code-block:: python

    from mbb import ModifiedBlackbody as MBB

The first step is usually to create a MBB model by filling in the necessary initial parameters: 

.. code-block:: python

    m = MBB(L=12.5, T=35, beta=1.8, z=2.65, opthin=True, pl=False)

A quick plot of this model can be made, if desired:

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, ax = m.plot_sed(obs_frame=True)
    plt.show()

.. image:: images/ex_plt_1.png
   :width: 350px

Fitting photometric data
------------------------

Most often, you want to fit a given model to photometric data points. ``mbb`` allows for Bayesian model fitting via the ``fit()`` method, which uses the ``emcee`` package to perform Markov Chain Monte Carlo (MCMC) sampling of the parameter space:

.. code-block:: python

    phot = (
        [250, 350, 450, 850, 1200], # wavelength in microns
        [0.012, 0.019, 0.0166, 0.00683, 0.0023], # flux in Jy
        [0.0044, 0.0064, 0.0036, 0.00057, 0.0003]  # error in Jy
        )
    result = m.fit(phot=phot, niter=500, params=['L', 'T', 'beta'], restframe=False)

.. code-block:: bash

    Running burn-in...
    100%|█████████████████████████████████████████| 300/300 [00:07<00:00, 38.87it/s]
    Running fitter...
    100%|█████████████████████████████████████████| 500/500 [00:12<00:00, 41.62it/s]
    Done 


You specify which parameters to fit using the ``params`` keyword argument; the options are ``L``, ``T``, ``beta``, ``alpha``, ``l0``, or ``z`` (the latter if you want to use ``mbb`` as a far-infrared photometric redshift code).

The parameters passed to initialize the ``ModifiedBlackbody`` are passed to ``emcee`` as the starting parameters of the fit.

View the resulting model after the fit, with uncertainties:

.. code-block:: python

    fig, ax = m.plot_sed(obs_frame=True)
    plt.show()


.. image:: images/ex_plt_2.png
   :width: 350px

You can also make a simple corner plot of the parameters that were varied:

.. code-block:: python

    fig = m.plot_corner()
    plt.show()

.. image:: images/ex_plt_3.png
   :width: 350px

The basic plotting routines are fairly sparse, but most plot aspects can be modified, or you can write your own functions to produce higher quality / publication-ready figures.



Modeling priors
---------------

By default, uniform priors are assumed on all the fit parameters, but you can change this by passing a dictionary, ``priors``, to ``fit``. 
Each key of ``priors`` should be the name of a parameter, and each value is either:

1. a dictionary with keywords ``mu`` and ``sigma``, to specify Gaussian priors
2. your own function, which takes the parameter as an argument and returns a number between 0.0 and 1.0.

.. code-block:: python

    result = m.fit(phot=phot, niter=500, params=['L', 'T', 'beta'], 
        restframe=False, priors = {'beta':dict(mu=1.8,sigma=0.3)})

.. code-block:: python

    Running burn-in...
    100%|█████████████████████████████████████████| 300/300 [00:07<00:00, 38.87it/s]
    Running fitter...
    100%|█████████████████████████████████████████| 500/500 [00:12<00:00, 41.62it/s]
    Done 


Accessing the fit results
-------------------------

To access the percentiles of the posterior distribition for any parameter in the fit:

.. code-block:: python

    print(m.post_percentile('beta', q=(16,50,84))) #16th, 50th, 84th percentiles

.. code-block:: python
    
    [1.56834795 1.83519843 2.10055382]

To get the reduced chi-squared value from the fit_result:

.. code-block:: python
    
    reduc_chi2 = m.fit_result['chi2'] / (m.fit_result['n_bands']-m.fit_result['n_params'])
	print(chi2)

.. code-block:: python
    
    0.8697752576488373


Currently, the measurement for ``L`` requires integration under the hood, so it can take a long time. The same applies for generating the corner plots. I'm working on speeding this process up.

The full ``emcee.EnsembleSampler`` is stored as the ``sampler`` element of the ``fit_result`` attribute. This can be used to perform any kind of analysis one would typically want with ``emcee``, such as looking at the autocorrelation time and other fit statistics, if desired.


To reset the ``fit_result`` and clear the priors, use ``reset()``. The parameters of the MBB will still be set to the best values from the previous fit, however.

.. code-block:: python
    
    m.reset()
    print(np.round(m.beta,2))


.. code-block:: python
    
    1.84


Multiprocessing
---------------

By default, ``mbb`` will try to use the number of available CPUs minus 2 to run the fit. To control this, you can either pass an integer to the ``ncores`` argument of ``fit`` (pass 1 to not use multiprocessing at all), or you can generate your own process Pool object (e.g., ``multiprocessing.pool.Pool``) and pass it as the ``pool`` argument.

Note: to avoid multiprocessing errors, the process start method is set to "fork" on Linux/macOS and to "spawn" on Windows. If you run into errors, I recommend passing in your own Pool object or forgoing multiprocessing.
