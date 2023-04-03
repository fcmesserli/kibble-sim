Signals
=======

.. currentmodule:: moving_mode.error_sim

Each simulation experiment is based on a time-series displacement signals. Depnding on the purpose of the simulation different signal types may be more appropriate than others. 

Signal
------
All signals are subclasses of the Signal class, guaranteeing each signal has a name (obsolete), a list for additional Signal objects, and a ``generate_signal`` method.

.. autoclass:: Signal
  :members:

Linear Signal
-------------
Linear displacement signals are traditionally used for Kibble Balances and provide a good point of reference for alternative methods.

.. autoclass:: LinearSignal
  :members:
  
Sine Signal
-------------
Sinusoidal coil motion is the proposed method for MSL's kibble balance. It requires less travel range, and has different noise rejection capabilities when compared with a linear mode.

.. autoclass:: SineSignal
  :members:
  
Vibration Noise Floor
---------------------
This class contains a set of sinusoidal displacement signals to represent mechanical noise in the system. It is generally added to the ``additional_signals`` attribute of a primary signal.

.. autoclass:: VibrationNoiseFloor
  :members:
  
Interpolated Signal
---------------------
This class is built to provide time interpolation of a signal from real measurement data allow for a continuous signal. This allows small timing differences to have an impact on the results. 

.. autoclass:: InterpolatedSignal
  :members:
