Random Noises
=======

.. currentmodule:: moving_mode.error_sim

Various Kibble balance processes (e.g. clocks) are affected by random noises. These classes contain parameters and methods relating to these processes.

Random Noise
------
All random noise classes are a subclass of RandomNoise guaranteeing they have a ``generate_noise`` method.

.. autoclass:: RandomNoise
  :members:

Additive Gaussian White Noise (AGWN)
-------------
Additive gaussian white noise is used in most if not all random processes in the simulation.

.. autoclass:: Agwn
  :members:
  
