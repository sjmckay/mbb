.. _mbb_funcs:

Modeling options
=======================

There are several modeling options that can be used for the :ref:`ModifiedBlackbody <mbbclass>` fitting. 
All of these follow the form of the `Casey et al. (2012) <https://doi.org/10.1111/j.1365-2966.2012.21455.x>`_ models, 
with optional mid-infrared powerlaw (slope defaults to ``alpha=2.0``) and the choice between assuming optically thin dust or the general opacity model (turnover wavelength defaults to ``l0 = 200``, in microns).

To choose a given model, use the ``pl`` (powerlaw) and/or ``opthin`` (optically thin) flags when intializing a ModifiedBlackbody instance.

The documentation for the overall model is below:

.. automodule:: mbb.mbb_funcs
    :members: