# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:40:26 2023

@author: F.Messerli
"""

# This, as it is currently set up, does not include errors from BL or from 
# vibration, and only runs each simulation once (keep in mind if adding random
# errors such as time jitter)

import numpy as np
import matplotlib.pyplot as plt
from moving_mode.error_sim import (
    Agwn,
    Bl,
    Clock,
    Coil,
    Dvm,
    LinearSignal,
    SineSignal,
    VibrationNoiseFloor,
    Interferometer,
    MovingModeExperiment,
    TimeIntervalAnalyser,
)

# All units are in SI excluding angles, which are given in degrees

# Key simulation paramters
OSC_AMP = 0.001
OSC_FREQS = [0.5, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, ]

RLC_C = 136e-9 # Total capacitance of the system
RLC_L = 594e-3 # Inductance of the coil
RLC_R = 1e10 # Internal resistance of voltmeter
RLC_RC = 75 # Resistance of the coil

DEADTIME = 2e-5 # 2e-5, For some reason without this some results are NaN for no apparent reason
SAMPLING_TIME = 0.01
EXP_TIME = 100  # Total time for each run
SAMP_TIMES = np.arange(
    0, EXP_TIME, SAMPLING_TIME
)

# General parameters
INT_REFERENCE = Clock(2.6e6, 0)
INT_WAVELENGTH = 633e-9
COIL_HEIGHT = 0.02
COIL_RADIUS = 0.120
COIL_TURNS = 929
CLK_FREQ = 1e7

B_DATA_PATH = "bl_40mm.csv"
POLYFIT_ORDER = 2 # Smaller order speeds up computation time

# Error inducing parameters
# Errors are set to their minima but reasonable values are suggested as comments
REFERENCE_JITTER = 0 # 1e-10

TIA_RESOLUTION = 0 # 2e-12
TIA_ACCURACY = 0 # 20e-12
TIA_NOISE = 0 # 440e-6

INT_SLEW_RATE = np.inf # 2e8
INT_SQUARE_NOISE = 0 # 440e-6 
XI = 1 # Relative intensity of the  reference beam
CHI = 1 # 0.5, Relative intensity of the measurement beam
PHI = 0 # 1, Offset angle of the half wave plate
THETA = 0 # 1, Offset angle of the polariser
DE1 = 0 # 2.57, Ellipticity of the horizontal polarisation component of the laser
DE2 = 0 # 2.57, Ellipticity of the vertical polarisation component of the laser

DVM_TIME_JITTER = 0 # 1e-10
DVM_LATENCY = 0 # 500e-9 
DVM_QUANTISATION = None # 7.5


def rlc_frequency_sweep(osc_freqs = OSC_FREQS):
    coil = Coil(
        COIL_HEIGHT, COIL_RADIUS, COIL_TURNS, RLC_C, RLC_L, RLC_RC
    )

    # Set up Bl and create an approximation based on bl_rel_error
    bl = Bl.from_csv(B_DATA_PATH, coil, polyfit_order=POLYFIT_ORDER)
    bl_true_poly = bl.bl_polyfit # BL polynomial withour error

    # Set up time reference with zero phase
    time_reference = Clock(CLK_FREQ, 0, Agwn(REFERENCE_JITTER))

    # Set up time_interval_analyser
    tia = TimeIntervalAnalyser(
        time_reference,
        base_resolution=TIA_RESOLUTION,
        base_accuracy=TIA_ACCURACY,
        internal_noise=TIA_NOISE,
    )

    # Set up interferometer
    interferometer = Interferometer(
        SAMPLING_TIME - DEADTIME,
        time_reference,
        interferometer_reference=INT_REFERENCE,
        tia=tia,
        square_slew_rate=INT_SLEW_RATE,
        square_noise_rms=INT_SQUARE_NOISE,
        timing_latency=0,
        wavelength=INT_WAVELENGTH,
        xi=XI,
        chi=CHI,
        phi=PHI,
        theta=THETA,
        dE1=DE1,
        dE2=DE2,
    )

    # Set up digital voltmeter
    dvm = Dvm(
        SAMPLING_TIME - DEADTIME,
        clock=Clock(1e7, 0, Agwn(DVM_TIME_JITTER)),
        timing_latency=DVM_LATENCY,
        quantisation_digits=DVM_QUANTISATION,
        internal_resistance=RLC_R,
    )
    bl_vals = []
    for osc_freq in osc_freqs:
        # Set up the experiment using established components
        exp = MovingModeExperiment(
            dvm,
            interferometer,
            None,       # Add displacement signal after
            time_reference,
            bl,
            SAMP_TIMES,
        )
        # 0 phase 0 offset signal
        exp.displacement_signal = SineSignal(osc_freq, OSC_AMP, 0, 0)

        # Whether to include RLC effects
        exp.coil_correction = True
        # Specifying the integral can help speed up computation times
        exp.voltage_integral = None
        # 
        bl_poly = bl_true_poly
        # Run the experiment once
        exp.run_experiment(
            1,
            bl_compensation=True,
            bl_poly=bl_poly,
        )
        res = exp.analyse_simple_sine_fit(2 * np.pi * osc_freq)
        bl_val = -res[0][0] * bl.at_z(0) + bl.at_z(0) # Absolute, not relative
        print(bl_val)
        bl_vals.append(bl_val)
        
    plt.plot(OSC_FREQS, bl_vals)
    plt.xlabel('Oscillation frequency (Hz)')
    plt.ylabel('BL value')
    plt.show()

        
rlc_frequency_sweep()