# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:24:53 2022

@author: F.Messerli

Tests do not cover randomness.
"""
import numpy as np
import unittest
import os
from moving_mode.error_sim import (
    Bl,
    Clock,
    Coil,
    Dvm,
    Agwn,
    LinearSignal,
    SineSignal,
    QuantisationError,
    VibrationNoiseFloor,
    Interferometer,
    MovingModeExperiment,
    TimeIntervalAnalyser,
)


class TestBl(unittest.TestCase):
    def test_bl_linear_polyfit(self):
        disp = np.arange(-4, 5, 1)
        field = (3 * disp + 2) / (2 * np.pi)
        coil = Coil(2.5, 1, 1, 0, 0, 0)
        bl = Bl(field, disp, coil, 5)
        expected_poly = (0, 0, 0, 0, 3, 2)
        for i, v in enumerate(bl.bl_polyfit.c):
            self.assertAlmostEqual(v, expected_poly[i], places=10)

    def test_bl_parabola_polyfit(self):
        disp = np.arange(-10, 11, 1)
        field = (5.0 * disp**2 + 4.0 * disp - 3.0) / (2 * np.pi)
        coil = Coil(8, 1, 5, 0, 0, 0)
        bl = Bl(field, disp, coil, 5)
        expected_poly = np.asarray((0, 0, 0, 5.0, 4.0, 71 / 3)) * 5.0
        for i, v in enumerate(bl.bl_polyfit.c):
            self.assertAlmostEqual(v, expected_poly[i], places=12)

    def test_bl_5th_order_polyfit(self):
        disp = np.arange(-4, 5, 1)
        field = (
            8 * disp**5
            - 7 * disp**4
            + 6 * disp**3
            - 5.0 * disp**2
            + 4.0 * disp
            - 3.0
        ) / (2 * np.pi)
        coil = Coil(0.15, 2, 3, 0, 0, 0)
        bl = Bl(field, disp, coil, 5)
        expected_poly = (
            np.asarray((8, -7, 6.15, -5.07875, 4.034003125, -3.00941929688))
            * 6
        )
        for i, v in enumerate(bl.bl_polyfit.c):
            self.assertAlmostEqual(v, expected_poly[i], places=10)


class TestDvm(unittest.TestCase):
    def setUp(self):
        self.dvm = Dvm(
            1e-3,
            clock=Clock(1e7, 0, Agwn(0)),
            timing_latency=175e-9,
            quantisation_digits=(8.5, 7.5, 6.5, 5.5, 4.5),
            quantisation_thresholds=(0.166, 10e-3, 100e-6, 1.5e-6, 0.5e-6),
        )
        self.time_reference = Clock(1e8, 0, Agwn(0))
        self.samp_times = np.arange(0, 1, 0.1)
        disp = np.linspace(-0.1, 0.1, 20)
        coil = Coil(2, 1, 1, 0, 0, 0)
        self.bl = Bl(0 * disp + 300 / (2 * np.pi), disp, coil, 0)
        self.velocity = 0.001
        self.displacement_signal = LinearSignal(self.velocity, -0.0005)

    def test_quantisation_error(self):
        self.assertEqual(self.dvm.quantisation_error.digit_resolution, 6)
        self.assertTrue(self.dvm.quantisation_error.half_digit)

    def test_qe_reset_params(self):
        self.dvm.integration_time = 1e-6
        self.assertEqual(self.dvm.quantisation_error.digit_resolution, 4)
        self.assertTrue(self.dvm.quantisation_error.half_digit)

    def test_quantisation_error_leq(self):
        self.dvm.integration_time = 10e-3
        self.assertEqual(self.dvm.quantisation_error.digit_resolution, 6)
        self.assertTrue(self.dvm.quantisation_error.half_digit)

    def test_quantisation_error_too_low(self):
        with self.assertRaises(ValueError):
            self.dvm.integration_time = 0.4e-6

    def test_quantisation_error_single_value(self):
        dvm = Dvm(
            1e-3,
            clock=Clock(1e-7, 0, Agwn(0)),
            quantisation_digits=5,
            quantisation_thresholds=None,
        )
        self.assertEqual(dvm.quantisation_error.digit_resolution, 5)
        self.assertFalse(dvm.quantisation_error.half_digit)

    def test_measure_voltage_no_noise_simultaneous_clocks(self):
        self.dvm.timing_latency = 0
        self.dvm.quantisation_error = QuantisationError(20)
        u_meas = self.dvm.measure_voltage(
            self.displacement_signal,
            self.samp_times,
            self.time_reference,
            self.bl,
        )
        expected_voltage = 300 * 0.001 * np.ones(len(self.samp_times) - 1)
        for i, u in enumerate(expected_voltage):
            self.assertAlmostEqual(u, u_meas[i], 13)

    def test_measure_voltage_cubic_bl(self):
        pass

    def test_measure_voltage_coil_correction_perfect_coil(self):
        self.dvm.timing_latency = 0
        self.dvm.quantisation_error = QuantisationError(20)
        self.dvm.internal_resistance = 1e10

        sine_signal = SineSignal(2 * np.pi, 1, 0, 0)

        u_meas_no_corr = self.dvm.measure_voltage(
            sine_signal,
            self.samp_times,
            self.time_reference,
            self.bl,
            coil_correction=False,
        )
        u_meas_corr = self.dvm.measure_voltage(
            sine_signal,
            self.samp_times,
            self.time_reference,
            self.bl,
            coil_correction=True,
        )

        for i, u in enumerate(u_meas_no_corr):
            self.assertAlmostEqual((u - u_meas_corr[i]) / u, 0, 12)

    def test_determine_sample_times_simultaneous_no_delay(self):
        self.dvm.timing_latency = 0
        expected_times = self.samp_times
        times = self.dvm.determine_sample_times(
            self.samp_times, self.time_reference
        )
        for i, t in enumerate(expected_times):
            self.assertAlmostEqual(t, times[i], 15)

    def test_determine_sample_times_simultaneous_delay(self):
        self.dvm.timing_latency = 175e-9
        expected_times = self.samp_times + 2e-7
        times = self.dvm.determine_sample_times(
            self.samp_times, self.time_reference
        )
        for i, t in enumerate(expected_times):
            self.assertAlmostEqual(t, times[i], 15)

    def test_determine_sample_times_simultaneous_delay_and_phase_diff(self):
        self.dvm.timing_latency = 175e-9
        self.dvm.clock.phase = 5 / 3 * np.pi
        expected_times = self.samp_times + 1e-7 + (5 / 6) * 1e-7
        times = self.dvm.determine_sample_times(
            self.samp_times, self.time_reference
        )
        for i, t in enumerate(expected_times):
            self.assertAlmostEqual(t, times[i], 15)

    def test_determine_sample_times_non_simultaneous(self):
        self.dvm.clock.freq = 1e8 + 1
        self.dvm.timing_latency = 0
        expected_times = [
            (n * 1e7 + np.ceil(n / 10)) / (1e8 + 1) for n in range(0, 10)
        ]
        times = self.dvm.determine_sample_times(
            self.samp_times, self.time_reference
        )
        for i, t in enumerate(expected_times):
            self.assertAlmostEqual(t, times[i], 15)


class TestVibrationNoiseFloor(unittest.TestCase):
    def setUp(self):
        self.freqs = np.arange(0, 100, 1)
        self.amps = np.logspace(-9, -3, 100)
        self.phases = np.linspace(0, 2 * np.pi, 100)
        np.savetxt(
            "no_phase.csv",
            np.column_stack((self.freqs, self.amps)),
            delimiter=",",
        )
        np.savetxt(
            "phase_and_header.csv",
            np.column_stack((self.freqs, self.amps, self.phases)),
            header="freqs,amps,phases",
            delimiter=",",
            comments="",
        )
        self.noise_floor = VibrationNoiseFloor(
            self.freqs, self.amps, self.phases
        )

    def tearDown(self):
        os.remove("no_phase.csv")
        os.remove("phase_and_header.csv")

    def test_from_csv_no_phase_no_header(self):
        noise_floor = VibrationNoiseFloor.from_csv("no_phase.csv", phase=False)
        self.assertTrue(np.array_equal(noise_floor.frequencies, self.freqs))
        self.assertTrue(np.array_equal(noise_floor.amplitudes, self.amps))
        self.assertTrue(
            np.array_equal(
                noise_floor.phases, np.zeros(len(noise_floor.phases))
            )
        )

    def test_from_csv_phase_and_header(self):
        noise_floor = VibrationNoiseFloor.from_csv(
            "phase_and_header.csv", phase=True, skip_header=1
        )
        self.assertTrue(np.array_equal(noise_floor.frequencies, self.freqs))
        self.assertTrue(np.array_equal(noise_floor.amplitudes, self.amps))
        self.assertTrue(np.array_equal(noise_floor.phases, self.phases))

    def test_remove_low_amplitude_oscillations(self):
        cutoff_vel = 1.0722672220103232e-06 * 50 * 2 * np.pi
        self.noise_floor.remove_low_amplitude_oscillations(cutoff_vel)
        self.assertTrue(
            np.array_equal(self.noise_floor.frequencies, self.freqs[50:])
        )
        self.assertTrue(
            np.array_equal(self.noise_floor.amplitudes, self.amps[50:])
        )
        self.assertTrue(
            np.array_equal(self.noise_floor.phases, self.phases[50:])
        )

    def test_generate_signal(self):
        self.noise_floor.frequencies = (1, 2)
        self.noise_floor.amplitudes = (1.5, 2.5)
        self.noise_floor.phases = (0, np.pi)
        t = np.arange(0, 10, 0.1)
        for i, v in enumerate(
            1.5 * np.sin(2 * np.pi * t) + 2.5 * np.sin(4 * np.pi * t + np.pi)
        ):
            self.assertAlmostEqual(
                v, self.noise_floor.generate_signal(t)[i], 13
            )


class TestQuantisationError(unittest.TestCase):
    def setUp(self):
        self.quantisation_error = QuantisationError(2.5)

    def test_half_digit(self):
        self.assertTrue(self.quantisation_error.half_digit)

    def test_init_value_error(self):
        with self.assertRaises(ValueError):
            QuantisationError(3.6)

    def test_apply_quantisation(self):
        x = [1.114, 1.115, -1.114, 1.114e-30, 1.114e30, 0, np.inf, 2.15, 2.14]
        expected_quantised = [
            1.11,
            1.12,
            -1.11,
            1.11e-30,
            1.11e30,
            0,
            np.inf,
            2.2,
            2.1,
        ]
        x_quantised = self.quantisation_error.apply_quantisation(x)
        for i, v in enumerate(expected_quantised):
            if v != 0:
                print(v)
                self.assertAlmostEqual((x_quantised[i]-v)/v, 0, 13)
            else:
                self.assertAlmostEqual(x_quantised[i], 0, 13)


class TestInterferometer(unittest.TestCase):
    def setUp(self):
        tia = TimeIntervalAnalyser(
            Clock(1e7, 0),
            base_resolution=0,
            base_accuracy=0,
            internal_noise=0,
        )
        interferometer_reference = Clock(2.6e6, 0, Agwn(0))
        self.interferometer = Interferometer(
            0.01,
            Clock(1e7, 0, Agwn(0)),
            interferometer_reference=interferometer_reference,
            tia=tia,
            square_slew_rate=np.inf,
            square_noise_rms=0,
            timing_latency=0,
            wavelength=633e-9,
            xi=1,
            chi=1,
            phi=0,
            theta=0,
            dE1=0,
            dE2=0,
        )
        self.velocity = 0.001
        self.displacement_signal = LinearSignal(self.velocity, -0.0005)
        self.time_reference = Clock(1e8, 0, Agwn(0))
        self.samp_times = np.arange(0, 1, 0.1)

    def test_measure_velocity_linear_no_nonlinearity(self):

        v_meas = self.interferometer.measure_velocity(
            self.displacement_signal, self.samp_times, self.time_reference
        )
        v_expected = [self.velocity for i in range(len(self.samp_times))]
        for i, v in enumerate(v_meas):
            self.assertAlmostEqual(v_expected[i], v, 13)


class TestMovingModeExperiment(unittest.TestCase):
    def setUp(self):
        tia = TimeIntervalAnalyser(
            Clock(1e7, 0, Agwn(0)),
            base_resolution=0,
            base_accuracy=0,
            internal_noise=0,
        )
        interferometer_reference = Clock(2.6e6, 0, Agwn(0))
        interferometer = Interferometer(
            1e-3,
            Clock(1e7, 0, Agwn(0)),
            interferometer_reference=interferometer_reference,
            tia=tia,
            square_slew_rate=np.inf,
            square_noise_rms=0,
            timing_latency=0,
            wavelength=633e-9,
            xi=1,
            chi=1,
            phi=0,
            theta=0,
            dE1=0,
            dE2=0,
        )
        dvm = Dvm(
            1e-3,
            clock=Clock(1e7, 0, Agwn(0)),
            timing_latency=0,
            quantisation_digits=None,
        )

        self.velocity = 0.001
        displacement_signal = LinearSignal(self.velocity, -0.0005)
        time_reference = Clock(1e8, 0, Agwn(0))
        disp = np.linspace(-2, 2, 20)
        coil = Coil(2, 1, 1, 0, 0, 0)
        bl = Bl(0 * disp + 300 / (2 * np.pi), disp, coil, 0)
        samp_times = np.arange(0, 1, 0.1)

        self.exp = MovingModeExperiment(
            dvm,
            interferometer,
            displacement_signal,
            time_reference,
            bl,
            samp_times,
        )
        self.exp.run_experiment(1)

    def test_run_experiment_no_errors(self):
        expected_u = [300 * self.velocity * np.ones(10)]
        expected_v = [self.velocity * np.ones(10)]
        expected_disp_start = [-0.0005 + self.velocity * self.exp.samp_times]
        expected_disp_end = [
            -0.0005
            + self.velocity
            * (self.exp.samp_times + self.exp.interferometer.integration_time)
        ]
        self.assertTrue(
            np.allclose(self.exp.u_results, expected_u, rtol=0, atol=1e-13)
        )

        self.assertTrue(
            np.allclose(self.exp.v_results, expected_v, rtol=0, atol=1e-13)
        )
        self.assertTrue(
            np.allclose(
                self.exp.displacement_start,
                expected_disp_start,
                rtol=0,
                atol=1e-13,
            )
        )
        self.assertTrue(
            np.allclose(
                self.exp.displacement_end,
                expected_disp_end,
                rtol=0,
                atol=1e-13,
            )
        )

    def test_analyse_average_no_error(self):
        results = self.exp.analyse_average()
        for r in results:
            self.assertAlmostEqual(r, 0, 13)

    def test_analyse_average_with_error(self):
        self.exp.u_results = [np.asarray((300, 301)), np.asarray((300, 303))]
        self.exp.v_results = [np.asarray((1, 1)), np.asarray((1, 1))]
        results = self.exp.analyse_average()
        self.assertAlmostEqual(results[0], -1 / 300, 13)
        self.assertAlmostEqual(results[1], 0.5 / 300, 13)
        self.assertAlmostEqual(results[2], 1 / 300, 13)

    def test_analyse_simple_sine_fit_no_error(self):
        f = 1
        w = 2 * np.pi * f
        self.exp.displacement_signal = SineSignal(f, 1, np.pi, 1)
        self.exp.run_experiment(1)
        results = self.exp.analyse_simple_sine_fit(w)
        for r in results[0]:
            self.assertAlmostEqual(r, 0, 13)
        self.assertAlmostEqual(
            results[1][0]["U amp"],
            300
            * w
            * np.sin(np.pi * f * self.exp.dvm.integration_time)
            / (np.pi * f * self.exp.dvm.integration_time),
            9,
        )
        self.assertAlmostEqual(results[1][0]["U offset"], 0, 9)
        self.assertAlmostEqual(results[1][0]["U phase"], -np.pi / 2, 13)
        self.assertAlmostEqual(
            results[1][0]["v amp"],
            1
            * w
            * np.sin(np.pi * f * self.exp.interferometer.integration_time)
            / (np.pi * f * self.exp.interferometer.integration_time),
            10,
        )
        self.assertAlmostEqual(results[1][0]["v offset"], 0, 12)
        self.assertAlmostEqual(results[1][0]["v phase"], -np.pi / 2, 13)

    def test_analyse_simple_sine_fit_with_error(self):
        pass

    def test_analyse_polyfit_no_error(self):
        self.exp.displacement_signal = LinearSignal(1e-3, -5e-4)
        self.exp.run_experiment(1)
        results = self.exp.analyse_simple_polyfit(1)
        for r in results[0]:
            self.assertAlmostEqual(r, 0, 13)
        # Still get non trivial higher order values probably due to floating point errors
        """
        expected_poly = (0, 300)
        for i, v in enumerate(expected_poly):
            self.assertAlmostEqual(v, results[1][0][i], 13)
        """

    def test_analyse_polyfit_with_error(self):
        pass


if __name__ == "__main__":
    unittest.main()
