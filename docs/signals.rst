Signals
=======

.. currentmodule:: moving_mode.error_sim

Each simulation experiment is based on a time-series displacement signals. 
Depnding on the purpose of the simulation different signal types may be more appropriate than others. 


All signals are subclasses of the Signal class, guaranteeing each signal has a name (obsolete) and a generate_signal method
.. autoclass:: Signal
  :members:
