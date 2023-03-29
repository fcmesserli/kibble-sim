"""Tools to help simulate the moving mode of MSL's kibble balance.

Errors not considered (not exhaustive list): thermal, alignment, gain errors of
the DVM, other interferometer beam errors (e.g. diffraction), PJVS errors,
clock stability, current effect, laser frequency stability, DVM noise 
(incl 3458A auto-zero), float precision, mains power noise, transient coil 
impedence, sine wave frequency measurement etc. Feel free to add them in!!!!

Equipment not considered: multiple DVMs, multiple interferometers, PJVS, 
frequency counter for oscillation frequency measurement.

I have attempted to document my assumptions via the README and through code
comments however I will have undoubtably made some implicit assumptions that
I have not recognised. 

This code is currently functional as-is, however it contains a long list of 
imporovement TODOs - mostly to increase clarity and sanitise user inputs.

Created on Mon Oct 10 13:27:31 2022
@author: F.Messerli

Typical usage example:
  Initialise params
  e = MovingModeExperiment(objects, params)
  e.run_experiment()
  e.analyse_simple_sine_fit()
"""


# Used to create abstract methods: methods that need to be overridden by a
# subclass
from abc import ABCMeta, abstractmethod

# Type hints e.g. error_name: str (error_name is a string type object)
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt
from scipy import interpolate


class Signal(object):
    """Parent for all signals; stores and applies time series functions.

    Attributes:
        signal_name: Name of the signal.
        additional_signals: List of other Signals to add when generating the
            signal.
    """

    __metaclass__ = ABCMeta

    def __init__(self, signal_name: str) -> None:
        """Initialise Signal."""
        self.signal_name = signal_name
        self.additional_signals = []

    @abstractmethod  # This function must be overridden by subclass
    def generate_signal(self, t: npt.ArrayLike) -> None:
        """Generate signal as a funciton of time.

        Args:
            t: Either array of times or single time for which to obtain signal
                value.
        """
        pass

    def add_signal(self, new_signal: "Signal") -> None:
        """Add a signal object whose generated signal will be added to this one."""
        self.additional_signals.append(new_signal)


class LinearSignal(Signal):
    """Linear signal with function y=v*t+y0.

    Could be used for signals like linear voltage drift or displacement in
    moving mode. A linear displacement signal used in most Kibble balances.

    Attributes:
        velocity: Velocity of the movement.
        offset: Offset constant for the linear signal i.e. could be used as the
            initial position for a linear displacement mode.
        signal_name: Name of the signal.
    """

    def __init__(
        self, velocity: float, offset: float, signal_name: str = "linear"
    ) -> None:
        """Init LinearSignal."""
        super().__init__(signal_name)
        self.velocity = velocity
        self.offset = offset

    def generate_signal(self, t: npt.ArrayLike) -> npt.ArrayLike:
        """Generate linear signal as a funciton of time.

        Args:
            t: Either array of times or single time for which to obtain signal.

        Returns:
            Value of the linear signal + any additional signals at time(s) t as
            array or float.
        """
        output_signal = self.velocity * t + self.offset
        for signal in self.additional_signals:
            output_signal += signal.generate_signal(t)
        return output_signal


class SineSignal(Signal):
    """Sinusoidal signal with function y = A sine(wt+phase).

    Could be used for signals like voltage noise or sinusoidal displacement.
    A sinusoidal displacement signal was proposed for the MSL kibble balance
    and used in the PTB balances. Simulations so far have demonstrated good
    vibration rejection but high sensitivity to synchronisation between the
    voltage and velocity signal.

    Attributes:
        frequency: Frequency of the oscillation.
        amplitude: Amplitude of the oscillation.
        phase: Inital phase of the oscillation.
        offset: Initial offset/ midpoint of the oscillation.
        signal_name: Name of the signal, default 'linear'.
    """

    def __init__(
        self,
        frequency: float,
        amplitude: float,
        phase: float,
        offset: float,
        signal_name: str = "sinusoidal",
    ) -> None:
        """Initialise SineSignal with sine parameters."""
        super().__init__(signal_name)
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.offset = offset

    def generate_signal(self, t: npt.ArrayLike) -> npt.ArrayLike:
        """Generate sinusoidal signal as a funciton of time.

        Args:
            t: Either array of times or single time for which to obtain signal.

        Returns:
            Value of the sinusoidal signal + any additional signals at time(s)
            t as array or float.
        """
        output_signal = (
            self.amplitude
            * np.sin(2 * np.pi * self.frequency * t + self.phase)
            + self.offset
        )
        for signal in self.additional_signals:
            output_signal += signal.generate_signal(t)
        return output_signal


class VibrationNoiseFloor(Signal):
    """Characterises the displacement noise floor in the frequency space.

    Attributes:
        frequencies: Frequencies of the noise sinusoids
        amplitudes: Amplitudes of the noise sinusoids.
        phases: Phases of the noise sinusoids.
    """

    def __init__(
        self,
        frequencies: npt.ArrayLike,
        amplitudes: npt.ArrayLike,
        phases: npt.ArrayLike,
        signal_name: str = "noise_floor",
    ) -> None:
        """Initialise VibrationNoiseFloor with arrays of sine parameters."""
        super().__init__(signal_name)
        self.frequencies = np.asarray(frequencies)
        self.amplitudes = np.asarray(amplitudes)
        self.phases = np.asarray(phases)

    @classmethod
    # Use like: noise = VibrationNoiseFloor.from_csv(fname)
    def from_csv(
        cls,
        fname: str,
        delimiter: str = ",",
        skip_header: int = 0,
        phase: bool = True,
    ) -> "VibrationNoiseFloor":
        """Import frequencies, amplitudes, and maybe phases from csv.

        Import noise floor with frequencies as the first column, amplitudes
        as the second column, and phase as the third column if phase==True.

        Args:
            fname: Name of csv file to import.
            delimiter: Delimiter used in csv file.
            skip_header: Number of header rows to skip.
            phase: Whether the file contains phases of the signals.
        """
        # TODO(finneganc): precondition on phases and width of csv
        csv_read = np.genfromtxt(
            fname, delimiter=delimiter, skip_header=skip_header
        )
        frequencies = csv_read[:, 0]
        amplitudes = csv_read[:, 1]
        if phase:
            phases = csv_read[:, 2]
        else:
            phases = np.zeros(len(frequencies))
        return cls(frequencies, amplitudes, phases)

    def randomise_phase(self) -> None:
        """Set phase values to random values between 0 and 2pi."""
        self.phases = np.random.uniform(0, 2 * np.pi, len(self.frequencies))

    def remove_low_amplitude_oscillations(
        self, cutoff_velocity: float
    ) -> None:
        """Remove noise signals with low velocity amplitude.

        Including lots of noise signals/frequency components can be slow. As Bl
        error seems to track with the velocity amplitude of the noise, removing
        the low amplitude signals could speed up the process without compromising
        on accuracy. This has not been extensively tested.

        Args:
            cutoff_velocity: All signals with velocity amplitude below this value
                will be removed.
        """
        new_amps = []
        new_freqs = []
        new_phases = []
        for freq, amp, phase in zip(
            self.frequencies, self.amplitudes, self.phases
        ):
            if amp * freq * 2 * np.pi >= cutoff_velocity:
                new_freqs.append(freq)
                new_amps.append(amp)
                new_phases.append(phase)

        self.frequencies = np.asarray(new_freqs)
        self.amplitudes = np.asarray(new_amps)
        self.phases = np.asarray(new_phases)

    def generate_signal(self, t: npt.ArrayLike) -> npt.ArrayLike:
        """Generate noise floor signal as a funciton of time.

        Args:
            t: Either array of times or single time for which to obtain signal.

        Returns:
            Value of the sinusoidal signal + any additional signals at time(s)
            t as array or float.
        Raises:
            ValueError: If t is not scalar or one dimensional array.
        """
        t = np.asarray(t)
        if len(t.shape) == 1:
            noise = np.zeros(len(t))
        elif len(t.shape) == 1:
            noise = 0
        else:
            raise ValueError("Must be a scalar or one dimensional array.")

        for freq, amp, phase in zip(
            self.frequencies, self.amplitudes, self.phases
        ):
            noise += amp * np.sin(2 * np.pi * freq * t + phase)

        output_signal = noise
        for signal in self.additional_signals:
            output_signal += signal.generate_signal(t)
        return output_signal


class InterpolatedSignal(Signal):
    """Create signal from discrete data with polynomial interpolation.

    Uses cubic spline interpolation with 'not a knot' boundary condition.

    Attributes:
        signal_interp: Scipy CubicSpline interpolation object of provided data.
        signal_name: Name of the signal.
    """

    def __init__(
        self,
        times: npt.ArrayLike,
        signal_values: npt.ArrayLike,
        signal_name: str = "interpolated_signal",
    ) -> None:
        """Initialise InterpolatedSignal.

        Args:
            times: Array of sampled times corresponding to signal values.
            signal_values: Value of the signal at corresponding times.
        """
        super().__init__(signal_name)
        self.signal_interp = interpolate.CubicSpline(
            times, signal_values, extrapolate=False
        )

    @classmethod
    # Use like: noise = VibrationNoiseFloor.from_csv(fname)
    def from_csv(
        cls,
        fname: str,
        time_range: Optional[Tuple[float, float]] = None,
        delimiter: str = ",",
        skip_header: int = 0,
    ) -> "VibrationNoiseFloor":
        """Import signal timeseries from csv.

        Import signal with times as the first column, and signal values
        as the second column.

        Args:
            fname: Name of csv file to import.
            time_range: Time range in csv to use for signal in a tuple (start,
                                                                        end).
            delimiter: Delimiter used in csv file.
            skip_header: Number of header rows to skip.
        """
        csv_read = np.genfromtxt(
            fname, delimiter=delimiter, skip_header=skip_header
        )
        times = csv_read[:, 0]
        signal_values = csv_read[:, 1]

        if time_range != None:
            start_idx = np.argmin(np.abs(times - time_range[0]))
            end_idx = np.argmin(np.abs(times - time_range[1])) + 1

            # Truncate arrays
            signal_values = signal_values[start_idx:end_idx]
            times = times[start_idx:end_idx]
            # Let the first time be zero
            times = times - times[0]

        return cls(times, signal_values)

    def generate_signal(self, t: npt.ArrayLike) -> npt.ArrayLike:
        """Generate noise floor signal as a funciton of time.

        Args:
            t: Either array of times or single time for which to obtain signal.

        Returns:
            Value of the sinusoidal signal + any additional signals at time(s)
            t as array or float.
        Raises:
            ValueError: If t contains a value outside the range provided when
                constructing the interpolator.
        """
        output_signal = self.signal_interp(t)
        if np.isnan(output_signal).any():
            raise ValueError(
                f"Either {max(t)} or {min(t)} or both are outside of the time range provided to the interpolator."
            )
        for signal in self.additional_signals:
            output_signal += signal.generate_signal(t)
        return output_signal


class RandomNoise(object):
    """Randomly generatde noise to be added to time-series data.

    Can be applied to either the measurement signal or other parameters like
    the timesteps or phase.

    Attributes:
        error_name: name of the error.
    """

    __metaclass__ = ABCMeta

    @abstractmethod  # This function must be overridden by subclass
    def generate_noise(self, num_samples) -> None:
        """Create an array to be added to a measurement signal.

        Args:
            num_samples: number of samples in the measurement signal
        """
        pass


class Agwn(RandomNoise):
    """Additive gaussian white noise with zero mean.

    Can be applied to either the measurement signal or other parameters like
    the timesteps or phase.

    Attributes:
        error_name: Name of the error.
        sigma: Standard deviation of the gaussian white noise.
    """

    def __init__(self, standard_deviation: float) -> None:
        """Init Agwn with name and standard deviation."""
        self.sigma = standard_deviation

    def generate_noise(self, num_samples: int) -> np.ndarray:
        """
        Create an array to be added to a measurement signal.

        Args:
            num_samples: Number of samples in the measurement signal.

        Returns:
            Numpy array of the added noise.
        """
        return np.random.normal(0, self.sigma, num_samples)


class Clock(object):
    """Time reference object.

    Clock stability is not considered here and is currently assumed to be
    perfect.

    Attributes:
        freq: Frequency of the time reference.
        phase: Phase of the clock relative to the time reference. So if the
            reference is at phase pi/2 when this clock ticks, the phase is
            pi/2.
        time_jitter: Random noise in timing.
    """

    def __init__(
        self, freq: float, phase: float, time_jitter: RandomNoise = Agwn(0)
    ) -> None:
        """Init Clock with frequency, phase, and jitter."""
        self.freq = freq
        self.phase = phase
        self.time_jitter = time_jitter


class QuantisationError(object):
    """Quantisation error associated with analogue to digital converters.

    Resolution is typically measured in digits, which can include a half digit.
    The extra .5 means there can be an extra most significant digit with value
    1 or 0.

    Args:
        digit_resultion: Integer digit cutoff for quantisation error.
        half_digit: Boolean. If resolution includes a half digit then the
            number may have an additional most significant digit with value 1.
    """

    def __init__(self, digit_resolution_float: float) -> None:
        """Init QuantisationError with resolution.

        Args:
            digit_resolution_float: Resolution in digits, must be multiple of
                0.5.
        """
        self.digit_resolution = int(digit_resolution_float)
        if (
            float(2 * digit_resolution_float).is_integer()
            and not float(digit_resolution_float).is_integer()
        ):
            self.half_digit = True
        elif float(digit_resolution_float).is_integer():
            self.half_digit = False
        else:
            raise ValueError("digit_resolution must be multiple of 0.5")

    def apply_quantisation(self, x: npt.ArrayLike) -> np.ndarray:
        """Apply quantisation error to an array.

        Currently a placeholder implementation.
        The 3458A likely has a different implementation with its SINT and DINT
        data formats. Hopefully this is a resonable approximation, however if
        it stores a scale factor the error could be much worse for a highly
        variable signal.

        Args:
            x: one dimensional array to be quantised
        """
        # TODO(finneganc): Check how the quantisation works based on data format
        # I worry it was something like it store a scale factor for the whole
        # run to avoid using up bits as exponents? Need to find out.
        # TODO(finneganc): precond, make sure digit_resolution is an integer
        x = np.asarray(x)

        # Do not accept 0 dimensional arrays
        if len(x.shape) != 1:
            raise ValueError("x must be a 1D array")

        # abs(x) where x=0 becomes 10^(resolution-1)
        x_positive = np.where(
            np.isfinite(x) & (x != 0),
            np.abs(x),
            10 ** (self.digit_resolution - 1),
        )
        if self.half_digit:
            extra_digit = np.asarray(
                [
                    1 if (str(x[i])[0] == "1" or str(x[i])[0:2] == "-1") else 0
                    for i in range(len(x))
                ]
            )
        else:
            extra_digit = np.zeros(len(x))
        mags = 10 ** (
            self.digit_resolution
            + extra_digit
            - 1
            - np.floor(np.log10(x_positive))
        )
        return np.round(x * mags) / mags


class Coil(object):
    """Primary coil for the Kibble Balance.

    Attributes:
        height: Height (in z direction) of the coil.
        radius: Radius of the coil.
        turns: Number of turns in the coil.
        capacitance: Inter-winding capacitance and capacitance of the coil and
            wires to the DVM to surrounding surfaces.
        inductance: Self inductance of the coil.
        resistance: Resistance of the wire in the coil and wires to the DVM.
    """

    def __init__(
        self,
        height: float,
        radius: float,
        turns: float,
        capacitance: float,
        inductance: float,
        resistance: float,
    ):
        """Init coil."""
        self.height = height
        self.radius = radius
        self.turns = turns
        self.c = capacitance
        self.l = inductance
        self.r = resistance

    def get_voltage_sin_scale_factor(self, w, r_i):
        """Find scale factor for sin component of the voltage due to LRC nature of the coil.

        Assumes the voltage is of the form V=V0 * sin(wt+phase)

        Args:
            w: Angular frequency of the displacement component.
            r_i: Internal resistance of the DVM.

        """
        scale_factor = (
            r_i
            * (r_i + self.r - self.c * self.l * r_i * w**2)
            / (
                r_i**2
                + 2 * r_i * self.r
                + self.r**2
                + self.l**2 * w**2
                - 2 * self.c * self.l * r_i**2 * w**2
                + self.c**2 * self.r**2 * r_i**2 * w**2
                + self.c**2 * self.l**2 * r_i**2 * w**4
            )
        )
        return scale_factor

    def get_voltage_cos_scale_factor(self, w, r_i):
        """Find scale factor for additional cos component of the voltage due to LRC nature of the coil.

        Assumes the voltage is of the form V=V0 * sin(wt+phase)

        Args:
            w: Angular frequency of the displacement component.
            r_i: Internal resistance of the DVM.

        """
        scale_factor = (
            r_i
            * (self.l + self.c * r_i * self.r)
            * w
            / (
                r_i**2
                + 2 * r_i * self.r
                + self.r**2
                + self.l**2 * w**2
                - 2 * self.c * self.l * r_i**2 * w**2
                + self.c**2 * self.r**2 * r_i**2 * w**2
                + self.c**2 * self.l**2 * r_i**2 * w**4
            )
        )
        return scale_factor

    # Add coil to Bl rather than parameters
    # Modify measure_voltage with an optional tag for 'add_coil_effects' or
    # something - that then takes coil object, and calculates the voltage from that


class Bl(object):
    """Bl or gamma: magnetic field strength x coil wire length.

    The above relation is only true if everything is properly aligned.

    Attributes:
        bl_polyfit: np.poly1d polynomial fit for Bl
    """

    def __init__(
        self,
        b_field: npt.ArrayLike,
        b_displacement: npt.ArrayLike,
        coil: Coil,
        polyfit_order: int = 8,
    ) -> None:
        """Init bl.

        Args:
            b_field: Magnetic field strength array with each point corresponding
                to the location in _b_displacement.
            b_displacement: Displacement array corresponding to _b_field.
            coil: Coil object containing coil dimensions.
            polyfit_order: Order of polynomial fit for Bl.
        """
        # Attributes are private because changing the variables wouldn't auto-
        # matically change the bl_polyfit. They could be made properties.
        self._b_field = b_field
        self._b_displacement = b_displacement
        self.coil = coil
        self._polyfit_order = polyfit_order
        # Lengh of coil wire in magnetic field
        self._l = 2 * np.pi * self.coil.radius * self.coil.turns
        self.bl_polyfit = self._create_bl_polyfit()

    @classmethod
    def from_csv(
        cls,
        fname: str,
        coil: Coil,
        polyfit_order: int = 8,
        delimiter: str = ",",
        skip_header: int = 0,
    ):
        """Import B(z) field from csv.

        Import B with position as the first column, and field values as the
        second column.

        Args:
            fname: Name of csv file to import.
            delimiter: Delimiter used in csv file.
            coil: Coil object containing coil dimensions.
            polyfit_order: Order of polynomial fit for Bl.
            skip_header: Number of header rows to skip.
        """
        csv_read = np.genfromtxt(
            fname, delimiter=delimiter, skip_header=skip_header
        )
        b_displacement = csv_read[:, 0]
        b_field = csv_read[:, 1]

        return cls(
            b_field,
            b_displacement,
            coil,
            polyfit_order,
        )

    def _create_bl_polyfit(self) -> np.poly1d:
        """Create a polynomial fit for the Bl field-position data.

        Both creates a fit for the B field then considers the effect of the
        coil width to get an average Bl experienced by the coil at any position.
        Note that the polyfit only seems accurate to about ~1e-12 in scenarios
        tested.
        """
        # Fit the Bl data
        bl_fit = np.polyfit(
            self._b_displacement, self._b_field * self._l, self._polyfit_order
        )
        # Initialise polynomial for definite integration
        definite_integral_poly = np.poly1d([0])
        # Integrate the initial fit
        indefinite_integral_poly = np.flip(np.polyint(bl_fit))
        for i, coefficient in enumerate(indefinite_integral_poly):
            if i != 0:  # Don't care about the 0th order
                # Add polynomials defined by their roots to compute definite
                # integration across the coil
                definite_integral_poly += np.poly1d(
                    coefficient
                    * np.poly1d(np.zeros(i) - self.coil.height / 2, True).c
                )
                definite_integral_poly -= np.poly1d(
                    coefficient
                    * np.poly1d(np.zeros(i) + self.coil.height / 2, True).c
                )
        # Divide definite integral by integration width to obtain average bl
        bl_interp_poly = np.polydiv(
            definite_integral_poly, np.poly1d([self.coil.height])
        )[0]
        # TODO(finneganc): handle 0 coil width
        return bl_interp_poly

    def _create_spline_interpolation(self) -> interpolate.CubicSpline:
        # TODO(finnegac): Cubic spline is probably more appropriate here? It would
        # mean that polynomial fitting cannot necessarily be perfect (which is
        # true IRL).
        raise NotImplementedError(
            "Spline interpolation not yet implemented because the integration problem is a bit of a pain. I think it can be done with Simpsons rule."
        )

    def at_z(self, z: npt.ArrayLike) -> np.ndarray:
        """Evaluate the polyomial fit at a specific position.

        Args:
            z: Vertical position of the coil relative to weighing position.
        Raises:
            ValueError: if z is outside of the range of B data provided.
        """
        z = np.asarray(z)
        if np.min(z) < np.min(self._b_displacement) or np.max(z) > np.max(
            self._b_displacement
        ):
            raise ValueError(
                f"z of {z} is outside the range of B field data provided."
            )
        return np.polyval(self.bl_polyfit, z)


# TODO: Change DVM to DigitalVoltmeter or TimeIntervalAnalyser to TIA
# TODO: Implement integral and differential noise
# TODO: Implement processing delay
class Dvm(object):
    """Digital voltmeter class containing error contributions and parameters.

    Used to contain all relevant information for a DVM with default parameters
    referring to the 3458A specifications. Factors not considered here include
    the 100ns time jumps that can occur even when only using the internal
    clock, the 0.01% frequency variation, and linearity. If linearity becomes
    a problem a quantum sampling voltmeter could be used i.e. output a sine
    signal from the PJVS.

    Attributes:
        clock: Clock object containing relavent information for the internal
            clock of the DVM.
        timing_latency: Timing latency when using an external trigger. The
            3458A can differ by up to 125ns model to model. Worth testing as it
            was much larger than specified for Lapuh (2018).
        integration_time:
    """

    def __init__(
        self,
        integration_time: float,
        clock: Clock = Clock(
            1e7, np.random.uniform(0, 2 * np.pi), Agwn(1e-10)
        ),
        timing_latency: float = 175e-9,
        quantisation_digits: Optional[npt.ArrayLike] = (
            8.5,
            7.5,
            6.5,
            5.5,
            4.5,
        ),
        quantisation_thresholds: Optional[npt.ArrayLike] = (
            0.166,
            10e-3,
            100e-6,
            1.5e-6,
            0.5e-6,
        ),
        internal_resistance: float = 1e10,
    ) -> None:
        """Init Dvm."""
        # TODO(finneganc): make integration_time & quantisations properties
        self.clock = clock
        self.timing_latency = timing_latency
        self._integration_time = integration_time
        self.internal_resistance = internal_resistance
        # TODO(finneganc): Lots of preconditioning - relationship, ordering, type etc.
        self._quantisation_digits = quantisation_digits
        self._quantisation_thresholds = quantisation_thresholds

        if type(self._quantisation_digits) != type(None):
            self.quantisation_error = self._get_quantisation_error()

    # Allow changing of _integration_time but recompute quantisation error if
    # changed
    @property
    def integration_time(self) -> float:
        """Get integration time."""
        return self._integration_time

    @integration_time.setter
    def integration_time(self, integration_time) -> None:
        self._integration_time = integration_time
        self.quantisation_error = self._get_quantisation_error()

    def _get_quantisation_error(self) -> float:
        """Get quantisation error object depending on Dvm parameters."""
        # TODO(finneganc): There must be a better way of doing this
        # Initialise the quantisation error based on the integration time
        # If quantisation error is integration_time independant
        if len(np.asarray(self._quantisation_digits).shape) == 0:
            return QuantisationError(self._quantisation_digits)
        else:
            # find the correct digit resolution based on integration time.
            idx = 0
            for threshold in self._quantisation_thresholds:
                if self.integration_time > threshold:
                    break
                else:
                    idx += 1
            else:
                raise ValueError(
                    f"Dvm integration time of {self.integration_time} is lower than minimum threshold of {self._quantisation_thresholds[-1]}."
                )
            return QuantisationError(self._quantisation_digits[idx])

    def determine_sample_times(
        self, samp_times: npt.ArrayLike, time_reference: Clock
    ) -> np.ndarray:
        """Determine the actual start times of the voltage measurement.

        This implementation assumes that the timing delay in measurement occurs
        before the internal clock sees the trigger signal. This implies that a
        nominal delay of say 175ns with no timing jitter will result in a delay
        greater than 175ns.

        Args:
            samp_times: The times of the trigger signals linked to the time
                reference.
            time_reference: Time reference clock for the balance.

        Returns:
            Numpy array of actual start times of the voltage measurement.
        """
        # Lapuh (2018) p 119 has more to say on optimising the relative
        # frequencies of the internal and trigger clocks for minimum time jitter

        # np.ceil will ceil an int cast as a floating point number due to floating point errors
        rough_ceil = lambda x, threshold: np.where(
            (x != 0) & (abs((np.floor(x) - x) / x) < threshold),
            np.floor(x),
            np.ceil(x),
        )
        # TODO(finneganc): get rid of the divide by zero warning above

        samp_times_clock_ticks = rough_ceil(
            samp_times * time_reference.freq, 1e-15
        )

        if (
            samp_times_clock_ticks / time_reference.freq - samp_times > 1e-15
        ).any():
            raise ValueError(
                f"sampling times must be governed by clock reference i.e on the counts of frequency {time_reference.freq}"
            )

        internal_jitter = self.clock.time_jitter.generate_noise(
            len(samp_times)
        )

        num_internal_ticks = self.clock.freq * (
            samp_times
            + time_reference.time_jitter.generate_noise(len(samp_times))
            + internal_jitter
            + self.timing_latency
            - self.clock.phase / (2 * np.pi * self.clock.freq)
        )

        num_internal_ticks = rough_ceil(num_internal_ticks, 1e-15)

        # TODO(finneganc): Check whether this aligns with Lapuh et. al. 2015.

        voltage_time = (
            num_internal_ticks / self.clock.freq
            + internal_jitter
            + self.clock.phase / (2 * np.pi * self.clock.freq)
        )
        return voltage_time

    def measure_voltage(
        self,
        displacement_signal: Signal,
        samp_times: npt.ArrayLike,
        time_reference: Clock,
        bl: Bl,
        coil_correction: bool = False,
    ) -> np.ndarray:
        """Measure the average voltage over the integration time.

        Args:
            displacement signal: signal object of the displacement in moving
                mode. Typically a combination of linear or sinusoidal, with
                noise.
            samp_times: the times at which samples are desired.
            time_reference: the time reference creating the trigger signals.
            bl: Bl object containing position dependance of the B field.

        Returns:
            voltage measurement at (roughly) the specified sampled times
        """
        if min(np.diff(samp_times)) < self.integration_time:
            # This is because the DVM needs time to integrate (and maybe process)
            raise ValueError(
                "Sampling time cannot be less than integration time"
            )
        voltage_time = self.determine_sample_times(samp_times, time_reference)
        displacement_start = displacement_signal.generate_signal(voltage_time)
        displacement_end = displacement_signal.generate_signal(
            voltage_time + self.integration_time
        )

        # TODO(finneganc): protect against an out of range z
        poly_int = np.polyint(bl.bl_polyfit)
        average_voltage = (
            np.polyval(poly_int, displacement_end)
            - np.polyval(poly_int, displacement_start)
        ) / self.integration_time

        # TODO(finneganc): Consider implementing the transient coil effect for
        # linear or at least determine its time constant to ensure it isn't a
        # problem
        if coil_correction:
            if type(displacement_signal) != SineSignal:
                # At this stage the symbolic integration only works with sinusoids
                raise ValueError(
                    f"displacement signal must be of type SineSignal not {type(displacement_signal)}"
                )
            # Prepare an array where each entry represents one signal saved as
            # [w, A, phase, correction factor for sine, correction factor for cosine]
            signal_array = []
            w = displacement_signal.frequency * 2 * np.pi
            signal_array.append(
                [
                    w,
                    displacement_signal.amplitude,
                    displacement_signal.phase,
                    bl.coil.get_voltage_sin_scale_factor(
                        w, self.internal_resistance
                    ),
                    bl.coil.get_voltage_cos_scale_factor(
                        w, self.internal_resistance
                    ),
                ]
            )
            # If the noise floor is present, add each component to the array
            for signal in displacement_signal.additional_signals:
                if type(displacement_signal) != VibrationNoiseFloor:
                    # At this stage the symbolic integration only works with sinusoids
                    raise ValueError(
                        f"additional signal must be of type VibrationNoiseFloor not {type(displacement_signal)}"
                    )
                for f, a, phase in zip(
                    signal.frequencies, signal.amplitudes, signal.phases
                ):
                    w = 2 * np.pi * f
                    signal_array.append(
                        [
                            w,
                            a,
                            phase,
                            bl.coil.get_voltage_sin_scale_factor(
                                w, self.internal_resistance
                            ),
                            bl.coil.get_voltage_cos_scale_factor(
                                w, self.internal_resistance
                            ),
                        ]
                    )

            import sympy

            t = sympy.Symbol("t")
            position = 0
            # Set up symbolic position signal
            for signal in signal_array:
                position += signal[1] * sympy.sin(signal[0] * t + signal[2])
            bl_at_t = 0
            # Use polynomial form of B field to find Bl(z(t))
            for i, c in enumerate(bl.bl_polyfit.c):
                bl_at_t += bl.bl_polyfit.c[i] * position ** (
                    len(bl.bl_polyfit.c) - i - 1
                )
            # Determine the integrand according to the circuit model
            to_integrate = 0
            for signal in signal_array:
                to_integrate += (
                    bl_at_t
                    * signal[0]
                    * signal[1]
                    * (
                        signal[3]
                        * sympy.sin(signal[0] * t + signal[2] + np.pi / 2)
                        + signal[4]
                        * sympy.cos(signal[0] * t + signal[2] + np.pi / 2)
                    )
                )
            # Integrate
            voltage_integral = sympy.integrate(to_integrate, t)
            # Determine measured voltage
            average_voltage = (
                np.asarray(
                    [
                        voltage_integral.evalf(
                            subs={t: time + self.integration_time}
                        )
                        - voltage_integral.evalf(subs={t: time})
                        for time in voltage_time
                    ]
                )
                / self.integration_time
            )
            # Sympy uses it's own float types. Convert to numpy
            average_voltage = np.asarray(average_voltage).astype(np.float64)

        if type(self._quantisation_digits) != type(None):
            average_voltage = self.quantisation_error.apply_quantisation(
                average_voltage
            )
        return average_voltage


class TimeIntervalAnalyser(object):
    """Measures time margins between data signals.

    Instead of fully simulating the TIA's operation the errors are approximated
    from the datasheet of the Carmel Instruments NK732. This is done for
    simplicity at the expense of specificity and generality.

    Attributes:
        clock: Internal time reference.
        base_resolution: Max resolution of the TIA (rms/std dev).
        base_accuracy: Maximum accuracy of the TIA, origins unclear.
        internal_noise: Internal noise in V^2
    """

    def __init__(
        self,
        clock: Clock,
        base_resolution: float = 2e-12,
        base_accuracy: float = 20e-12,
        internal_noise: float = 440e-6,
    ):
        """Init the TIA."""
        self.clock = clock
        self.base_resolution = base_resolution
        self.base_accuracy = base_accuracy
        self.internal_noise = internal_noise

    def measure_interval(self, interval_times, slew_rate, external_noise):
        """Add TIA errors to the interval times.

        Args:
            interval_times: interval times measured by the TIA (the differences
                between the start and end points).
            slew_rate: Input signal slew rate at zero crossing in V/s.
            external_noise: RMS noise in the TIA input signals.
        """
        n = len(interval_times)
        base_res = np.random.normal(0, self.base_resolution, n)

        start_trigger_error = np.random.normal(
            0,
            np.sqrt(self.internal_noise**2 + external_noise**2)
            / slew_rate,
            n,
        )
        stop_trigger_error = np.random.normal(
            0,
            np.sqrt(self.internal_noise**2 + external_noise**2)
            / slew_rate,
            n,
        )
        # TODO(finneganc): figure out exactly how to implement timebase error
        timebase_error = self.clock.time_jitter.generate_noise(
            n
        ) + self.clock.time_jitter.generate_noise(n)

        trigger_level_time_error = np.random.normal(0, 5e-3 / slew_rate, n)

        # Assume base accuracy is constant over a run but varies experiment to
        # experiment (and is uniform). This error is likely a result of the
        # digital interpolation scheme used to exceed the 10MHz (100ns) limit
        # so this seems like a reasonable approximation to the error. Might be
        # single shot though, which would be better. Sticking with worst case
        # for now.
        base_accuracy = np.random.uniform(
            -self.base_accuracy / 2, self.base_accuracy / 2
        )

        return (
            interval_times
            + base_res
            + start_trigger_error
            + stop_trigger_error
            + timebase_error
            + trigger_level_time_error
            + base_accuracy
        )


class Interferometer(object):
    """Heterodyne Michelson interferometer for accurate position measurement.

    In the MSL kibble balance three of these will be used to determine the z
    position of the center of the coil.

    Attributes:
        integration_time: Time over which one displacement measurement is taken.
        clock: Internal clock, currently does nothing.
        interferometer_reference: A clock representing the reference beat
            signal of the interferometer.
        tia: Time interval analyser associated with the interferometer.
        square_slew_rate: Slew rate (gradient) of the square waves that are
            passed to the tia.
        square_noise_rms: RMS noise of the square waves that are passed to the
            tia.
        timing_latency: Time between receiving the trigger signal and being
            prepared to make the measurement.
        wavelength: Wavelength of the laser.
        xi: Intensity of the measurement arm as it reaches the polariser, arb
            units.
        chi: Intensity of the reference arm as it reaches the polariser.
        alpha: Offset angle of the polarised light meant for the reference arm
            (rad).
        beta: Offset angle of the polarised light meant for the measurement arm
            (rad).
        theta: Angle of the polariser relative to pi/4 radians.
        dE1: Ellipticity of the light meant to go into the measurement arm (rad).
        dE2: Ellipticity of the light meant to go into the reference arm (rad.
    """

    def __init__(
        self,
        integration_time: float,
        clock: Clock,  # TODO(finneganc): implement clock - currently assumes it's the time reference
        interferometer_reference: Clock,  # Clock(2.6e6, 0, Agwn('time-jitter', 0))
        tia: TimeIntervalAnalyser,
        square_slew_rate: float = 2e8,
        square_noise_rms: float = 0,
        timing_latency: float = 0,
        wavelength: float = 633e-9,
        xi: float = 1.0,
        chi: float = 1.8,
        phi: float = 0.5,
        theta: float = 0.5,
        dE1: float = 0.05,
        dE2: float = 0.05,
    ) -> None:
        """Init Interferometer with key parameters including NLE params.

        Args:
            phi: Rotation of the half wave plate relative to its ideal value (
                matching the incoming light and the polarising beam splitter)
                in degrees.
            theta: Angle of the polariser relative to 45 degrees
            dE1: Ellipticity of the light meant to go into the measurement arm
                in degrees.
            dE2: Ellipticity of the light meant to go into the reference arm
                in degrees.
        """
        # TODO(finneganc): find out about interferometer latency
        self.integration_time = integration_time
        self.clock = clock
        self.interferometer_reference = interferometer_reference
        self.tia = tia
        self.square_slew_rate = square_slew_rate
        self.square_noise_rms = square_noise_rms
        self.timing_latency = timing_latency
        self.wavelength = wavelength
        self.xi = xi
        self.chi = chi
        self.alpha = phi * np.pi * 2 / 180
        self.beta = -phi * np.pi * 2 / 180
        self.theta = theta * np.pi / 180
        self.dE1 = dE1 * np.pi / 180
        self.dE2 = dE2 * np.pi / 180

        # TODO(finneganc): Change this to a warning and instead just alter the integration time to be one cycle less
        if (
            (self.integration_time * self.interferometer_reference.freq) % 1
        ) / max(
            (self.integration_time, self.interferometer_reference.freq)
        ) > 1e-15:
            raise ValueError(
                f"integration time of {self.integration_time} must include an integer number of reference cycles (frequency {self.interferometer_reference.freq})"
            )

    def get_nonlinearity_from_signal(
        self, carrier_signal: npt.ArrayLike
    ) -> np.ndarray:
        """Create nonlinearity error to add to measurement signal.

        Model from (Cosijins et. al, 2002). This has been model has been tested
        at MSL and provides similar results to experiments though with
        deviations on the same order of magnitude as the result. The effect of
        this error can be minimised by sampling over one fringe.

        Args:
            carrier_signal: Signal for which the nle applies.

        Returns:
            Numpy array of the added non-linearity error.
        """
        conversion = 4 * np.pi
        del_phi = conversion * carrier_signal / self.wavelength

        A = (
            -(
                self.xi**2 * (np.sin(self.beta)) ** 2
                + self.chi**2 * (np.cos(self.beta)) ** 2
            )
            * np.cos(0.5 * self.dE1)
            * np.sin(0.5 * self.dE2)
            - (
                self.xi**2 * (np.cos(self.alpha)) ** 2
                + self.chi**2 * (np.sin(self.alpha)) ** 2
            )
            * np.sin(0.5 * self.dE1)
            * np.cos(0.5 * self.dE2)
        ) * np.cos(del_phi) + (
            self.xi**2 * np.cos(self.alpha) * np.sin(self.beta)
            + self.chi**2 * np.sin(self.alpha) * np.cos(self.beta)
        ) * np.cos(
            0.5 * self.dE1 + 0.5 * self.dE2
        ) * np.sin(
            del_phi
        )

        B = (
            (
                self.xi**2 * (np.sin(self.beta)) ** 2
                - self.chi**2 * (np.cos(self.beta)) ** 2
            )
            * np.cos(0.5 * self.dE1)
            * np.sin(0.5 * self.dE2)
            + (
                self.xi**2 * (np.cos(self.alpha)) ** 2
                - self.chi**2 * (np.sin(self.alpha)) ** 2
            )
            * np.sin(0.5 * self.dE1)
            * np.cos(0.5 * self.dE2)
        ) * np.cos(del_phi) + (
            -self.xi**2 * np.cos(self.alpha) * np.sin(self.beta)
            + self.chi**2 * np.sin(self.alpha) * np.cos(self.beta)
        ) * np.cos(
            0.5 * self.dE1 + 0.5 * self.dE2
        ) * np.sin(
            del_phi
        )

        C = (
            self.xi
            * self.chi
            * (
                np.cos(self.beta)
                * np.sin(self.beta)
                * np.cos(0.5 * self.dE1)
                * np.sin(0.5 * self.dE2)
                * (1 - np.cos(2 * del_phi))
                + np.sin(self.alpha)
                * np.sin(self.beta)
                * np.cos(0.5 * self.dE1)
                * np.cos(0.5 * self.dE2)
                * np.sin(2 * del_phi)
                - np.cos(self.alpha)
                * np.cos(self.beta)
                * np.sin(0.5 * self.dE1)
                * np.sin(0.5 * self.dE2)
                * np.sin(2 * del_phi)
                - np.sin(self.alpha)
                * np.cos(self.alpha)
                * np.sin(0.5 * self.dE1)
                * np.cos(0.5 * self.dE2)
                * (1 + np.cos(2 * del_phi))
            )
        )

        D = (
            (
                self.xi**2 * (np.sin(self.beta)) ** 2
                + self.chi**2 * (np.cos(self.beta)) ** 2
            )
            * np.cos(0.5 * self.dE1)
            * np.sin(0.5 * self.dE2)
            + (
                self.xi**2 * (np.cos(self.alpha)) ** 2
                + self.chi**2 * (np.sin(self.alpha)) ** 2
            )
            * np.sin(0.5 * self.dE1)
            * np.cos(0.5 * self.dE2)
        ) * np.sin(del_phi) + (
            self.xi**2 * np.cos(self.alpha) * np.sin(self.beta)
            + self.chi**2 * np.sin(self.alpha) * np.cos(self.beta)
        ) * np.cos(
            0.5 * self.dE1 + 0.5 * self.dE2
        ) * np.cos(
            del_phi
        )

        E = (
            (
                -self.xi**2 * (np.sin(self.beta)) ** 2
                + self.chi**2 * (np.cos(self.beta)) ** 2
            )
            * np.cos(0.5 * self.dE1)
            * np.sin(0.5 * self.dE2)
            + (
                -self.xi**2 * (np.cos(self.alpha)) ** 2
                + self.chi**2 * (np.sin(self.alpha)) ** 2
            )
            * np.sin(0.5 * self.dE1)
            * np.cos(0.5 * self.dE2)
        ) * np.sin(del_phi) + (
            -self.xi**2 * np.cos(self.alpha) * np.sin(self.beta)
            + self.chi**2 * np.sin(self.alpha) * np.cos(self.beta)
        ) * np.cos(
            0.5 * self.dE1 + 0.5 * self.dE2
        ) * np.cos(
            del_phi
        )

        F = (
            self.xi
            * self.chi
            * (
                np.cos(self.beta)
                * np.sin(self.beta)
                * np.cos(0.5 * self.dE1)
                * np.sin(0.5 * self.dE2)
                * np.sin(2 * del_phi)
                + np.cos(self.alpha)
                * np.cos(self.beta)
                * (
                    np.cos(0.5 * self.dE1) * np.cos(0.5 * self.dE2)
                    - np.sin(0.5 * self.dE1)
                    * np.sin(0.5 * self.dE2)
                    * np.cos(2 * del_phi)
                )
                + np.sin(self.alpha)
                * np.sin(self.beta)
                * (
                    -np.sin(0.5 * self.dE1) * np.sin(0.5 * self.dE2)
                    + np.cos(0.5 * self.dE1)
                    * np.cos(0.5 * self.dE2)
                    * np.cos(2 * del_phi)
                )
                + np.sin(self.alpha)
                * np.cos(self.alpha)
                * np.sin(0.5 * self.dE1)
                * np.cos(0.5 * self.dE2)
                * np.sin(2 * del_phi)
            )
        )

        # Combining terms
        nonlin_model = -np.arctan(
            (A + B * np.sin(2 * self.theta) + C * np.cos(2 * self.theta))
            / (D + E * np.sin(2 * self.theta) + F * np.cos(2 * self.theta))
        )
        nonlin_model = nonlin_model * self.wavelength / conversion
        self.nonlin_model = nonlin_model
        return nonlin_model

    def determine_sample_times(
        self, samp_times: npt.ArrayLike, time_reference: Clock
    ) -> np.ndarray:
        """Determine the actual start and stop times of the velocity measurement.

        Args:
            samp_times: The times of the trigger signals linked to the time
                reference.
            time_reference: Time reference clock for the balance.

        Returns:
            Numpy array of actual start times of the velocity measurement.

        Raises:
            ValueError: If the sampling time does not align with time
            reference.
        """
        # Lapoh (2018) p 119 has more to say on optimising the relative
        # frequencies of the internal and trigger clocks for minimum time jitter

        # np.ceil will ceil an int cast as a floating point number due to floating point errors
        rough_ceil = lambda x, threshold: np.where(
            (x != 0) & (abs((np.floor(x) - x) / x) < threshold),
            np.floor(x),
            np.ceil(x),
        )
        # TODO(finneganc): get rid of the divide by zero warning above

        samp_times_clock_ticks = rough_ceil(
            samp_times * time_reference.freq, 1e-15
        )

        if (
            samp_times_clock_ticks / time_reference.freq - samp_times > 1e-15
        ).any():
            raise ValueError(
                f"sampling times must be governed by clock reference i.e on the counts of frequency {time_reference.freq}"
            )

        internal_jitter = (
            self.interferometer_reference.time_jitter.generate_noise(
                len(samp_times)
            )
        )

        # TODO(finneganc): synchronise jitter with voltage
        # TODO(finneganc): add jitters to end times too
        num_internal_ticks = self.interferometer_reference.freq * (
            samp_times
            + time_reference.time_jitter.generate_noise(len(samp_times))
            + self.timing_latency
            + internal_jitter
            - self.interferometer_reference.phase
            / (2 * np.pi * self.interferometer_reference.freq)
        )

        num_internal_ticks = rough_ceil(num_internal_ticks, 1e-15)

        samp_times = (
            num_internal_ticks / self.interferometer_reference.freq
            + internal_jitter
            + self.interferometer_reference.phase
            / (2 * np.pi * self.interferometer_reference.freq)
        )
        return samp_times

    def measure_position(
        self, displacement_signal, samp_times, time_reference
    ):
        """Measure the displacement across the integration time.

        Args:
            displacement_signal: The true displacement of the coil.
            samp_times: Sample trigger signals linked to time reference.
            time_reference: Clock acting as time reference for the balance.


        Returns:
            List of displacements at the start of each measurement and list of
            displacements at the eand of each measurement, in a tuple.

        Raises:
            ValueError: If sampling time less than integration time.
        """
        if min(np.diff(samp_times)) < self.integration_time:
            # This is because the DVM needs time to integrate (and maybe process)
            raise ValueError(
                "Sampling time cannot be less than integration time"
            )

        velocity_time = self.determine_sample_times(samp_times, time_reference)
        displacement_start = displacement_signal.generate_signal(velocity_time)
        displacement_end = displacement_signal.generate_signal(
            velocity_time + self.integration_time
        )

        displacement_start = (
            displacement_start
            + self.get_nonlinearity_from_signal(displacement_start)
        )

        # Change in time seen by TIA due to phase shift due to doppler. Note
        # This could correspond to a phase shift of more than 2 pi so is not
        # actually the phase difference between the nearest reference edge
        # and next measurement edge.
        start_dt = (
            2
            * displacement_start
            / (self.wavelength * self.interferometer_reference.freq)
        )
        start_dt = self.tia.measure_interval(
            start_dt, self.square_slew_rate, self.square_noise_rms
        )
        displacement_start = (
            self.wavelength * self.interferometer_reference.freq * start_dt / 2
        )

        displacement_end = (
            displacement_end
            + self.get_nonlinearity_from_signal(displacement_end)
        )

        end_dt = (
            2
            * displacement_end
            / (self.wavelength * self.interferometer_reference.freq)
        )
        end_dt = self.tia.measure_interval(
            end_dt, self.square_slew_rate, self.square_noise_rms
        )
        displacement_end = (
            self.wavelength * self.interferometer_reference.freq * end_dt / 2
        )

        return displacement_start, displacement_end

    def measure_velocity(
        self,
        displacement_signal,
        samp_times,
        time_reference,
        return_position=False,
    ):
        """Measure the average velocity across the integration time.

        Args:
            displacement_signal: The true displacement of the coil.
            samp_times: Sample trigger signals linked to time reference.
            time_reference: Clock acting as time reference for the balance.
            return_position: If true, also return the start and end position
                for each sample.

        Returns:
            Either just the average velocity for each sample time or that and
            the displacement at the start of each measurement and the
            displacement at the eand of each measurement, in a tuple.
        """
        displacement_start, displacement_end = self.measure_position(
            displacement_signal, samp_times, time_reference
        )
        average_velocity = (
            displacement_end - displacement_start
        ) / self.integration_time
        if return_position:
            return average_velocity, displacement_start, displacement_end
        else:
            return average_velocity


class MovingModeExperiment(object):
    """One or more continuous moving mode measurements.

    Setup, run, and analyse.

    Attributes:
        dvm: Digital voltmeter used in the experiment.
        interferometer: Laser interferometer used in experiment.
        displacement_signal: The true displacement of the coil.
        time_referece: The time reference clock.
        bl: The true Bl of the magnet/coil setup.
        samp_times: When to send the trigger signal to the DVM and
            interferometer.
        weighing_pos: The z position to be used in the weighing mode.
        u_results: List of voltage measurements for each run.
        v_results: List of velocity measurements for each run.
        displacement_start: The displacement recorded at the start of each
            velocity measurement.
        displacement_end: The displacement recorded at the end of each
            velocity measurement.
        bl_weighing_pos: The value of Bl at the weighing position. This is the
            target for the experiment and analysis.
        coil_correction: Whether alter measured voltage due to LRC nature of
            the coil.
    """

    def __init__(
        self,
        dvm: Dvm,
        interferometer: Interferometer,
        displacement_signal: Signal,
        time_reference: Clock,
        bl: Bl,
        samp_times: npt.ArrayLike,
        weighing_pos: float = 0,
        coil_correction: bool = False,
    ) -> None:
        """Set up experiment."""
        self.dvm = dvm
        self.interferometer = interferometer
        self.displacement_signal = displacement_signal
        self.time_reference = time_reference
        self.bl = bl
        self.samp_times = samp_times
        self.u_results = None
        self.v_results = None
        self.displacement_start = None
        self.displacement_end = None
        self.weighing_pos = weighing_pos
        # TODO(finneganc): Rename to bl_at_weighing_pos
        self.bl_weighing_pos = self.bl.at_z(self.weighing_pos)
        self.coil_correction = coil_correction

    # TODO(finneganc): Consider moving this to a utils class/module
    @classmethod
    def sine_fit(cls, w: float, t: np.ndarray, y: np.ndarray) -> tuple:
        """Compute a three-parameter sine fit for time series data.

        A three parameter sine fit was chosen over the Fast Fourier Transform (FFT)
        as the FFT struggles with the picket fence effect. Under the condition of
        coherent sampling at a frequency sampled by the FFT the result should be
        identical. A four parameter sine fit may be necessary if frequency is not
        known beforehand.

        Args:
            w: Angular frequency of the sinusoid.
            t: Array of times corresponding to y values.
            y: Time-series data to fit.

        Returns:
            Output of sine fitting with tuple (amplitude, phase, offset).
        """
        D1 = np.sin(w * t)
        D2 = np.cos(w * t)
        D3 = np.ones(len(t))
        D = np.column_stack((D1, D2, D3))

        params, _, _, _ = np.linalg.lstsq(D, y, rcond=None)
        amplitude = np.sqrt(params[0] ** 2 + params[1] ** 2)

        phase = np.arctan2(params[1], params[0])
        offset = params[2]

        return amplitude, phase, offset

    def run_experiment(
        self,
        num_runs: int = 1,
        bl_compensation: bool = False,
        bl_poly: Optional[np.poly1d] = None,
    ) -> None:
        """Return and overwrite results.

        Args:
            num_runs: The number times to repeat the experiment.
            bl_compensation: Whether to try remove the effect of a curved Bl(z)
                with an estimate of Bl(z)/Bl(meas_pos).
            bl_poly: Polynomial approximating Bl(z)/Bl(meas_pos) for use in Bl
                compensation.
        """
        self.u_results = []
        self.v_results = []
        self.displacement_start = []
        self.displacement_end = []
        for run in range(num_runs):
            measured_voltage = self.dvm.measure_voltage(
                self.displacement_signal,
                self.samp_times,
                self.time_reference,
                self.bl,
                coil_correction=self.coil_correction,
            )

            (
                measured_velocity,
                disp_start,
                disp_end,
            ) = self.interferometer.measure_velocity(
                self.displacement_signal,
                self.samp_times,
                self.time_reference,
                return_position=True,
            )

            # To compensate for bl voltage measurement from a value derived
            # from a measurement of Bl(z)/Bl(0) (only the value relative to
            # Bl(0) matters).
            if bl_compensation:
                # Normalise to weighing_pos value
                bl_poly /= np.polyval(bl_poly, self.weighing_pos)
                # Integrate as it is the voltage integral that is measured
                poly_comp = np.polyint(bl_poly)
                # Meas = Bl(0)*(polycomp(disp_end)-polycomp(disp_start))
                # Want = Bl(0)*(disp_end-disp_start) (what you get if Bl(z)=Bl(0))
                # So multiply Meas by Want/Meas to get Want!
                to_multiply = (disp_end - disp_start) / (
                    np.polyval(poly_comp, disp_end)
                    - np.polyval(poly_comp, disp_start)
                )
                measured_voltage *= to_multiply

            self.u_results.append(measured_voltage)
            self.v_results.append(measured_velocity)
            self.displacement_start.append(disp_start)
            self.displacement_end.append(disp_end)

    def analyse_average(self) -> Tuple[float, ...]:
        """Average all Bl(z) measurements and compare to true Bl(weighing_pos).

        All metrics are normalised to the true value of Bl at the weighing
        position.

        Returns:
            Tuple of metrics including: rel_bias - the bias of the measured
            Bl, rel_stddev - the stddev of the measured Bl across multiple
            runs, and rel_avg_residuals - the average across all runs of the
            MAE (mean absolute error) of the fit.
        """
        bls_meas = []
        residuals = []
        for u_result, v_result in zip(self.u_results, self.v_results):
            result = u_result / v_result
            bls_meas.append(np.average(result))
            residuals.append(abs(np.average(result) - result))
        rel_bias = (
            self.bl_weighing_pos - np.average(bls_meas)
        ) / self.bl_weighing_pos
        rel_stddev = np.std(bls_meas) / self.bl_weighing_pos
        rel_avg_residuals = np.average(residuals) / self.bl_weighing_pos
        return (rel_bias, rel_stddev, rel_avg_residuals)

    # TODO(finneganc): Make 4 parameter sine fit
    def analyse_simple_sine_fit(
        self,
        w: tuple,
    ) -> Tuple[Tuple[float, ...], float]:
        """Sine fit U and v with Bl as the ratio of the amplitudes.

        All metrics are normalised to the true value of Bl at the weighing
        position.

        Args:
            w: The angular frequency of the sine to be fitted.

        Returns:
            Tuple of metrics including: rel_bias - the bias of the measured
            Bl, rel_stddev - the stddev of the measured Bl across multiple
            runs, and rel_avg_residuals - the average across all runs of the
            MAE (mean average error) of the fit propogated and normalised to
            Bl. Params is the last item: a list of dictionaries each containing
            the sine fit parameters for the run corresponding to the position
            in the list.
        """
        params = []
        bls_meas = []
        residuals = []
        for u_result, v_result in zip(self.u_results, self.v_results):
            # Have the sample time be the middle of the integrated sample
            u_t = self.samp_times + self.dvm.integration_time / 2
            u_amp, u_phase, u_offset = MovingModeExperiment.sine_fit(
                w, u_t, u_result
            )
            u_residuals = np.average(
                abs(u_result - u_amp * np.sin(w * u_t + u_phase) - u_offset)
            )

            # Have the sample time be the middle of the integrated sample
            v_t = self.samp_times + self.interferometer.integration_time / 2
            v_amp, v_phase, v_offset = MovingModeExperiment.sine_fit(
                w, v_t, v_result
            )
            v_residuals = np.average(
                abs(v_result - v_amp * np.sin(w * v_t + v_phase) - v_offset)
            )

            # Propagate residuals with the partial derivative method
            residuals.append(
                u_residuals / v_amp + u_amp * v_residuals / (v_amp**2)
            )
            bls_meas.append(u_amp / v_amp)
            params.append(
                {
                    "U amp": u_amp,
                    "U offset": u_offset,
                    "U phase": u_phase,
                    "v amp": v_amp,
                    "v offset": v_offset,
                    "v phase": v_phase,
                }
            )
        rel_bias = (
            self.bl_weighing_pos - np.average(bls_meas)
        ) / self.bl_weighing_pos
        rel_stddev = np.std(bls_meas) / self.bl_weighing_pos
        rel_avg_residuals = np.average(residuals) / self.bl_weighing_pos

        return (rel_bias, rel_stddev, rel_avg_residuals), params

    # TODO(finneganc): reconcile sine_fit and polyfit naming
    def analyse_simple_polyfit(
        self, deg: tuple
    ) -> Tuple[Tuple[float, ...], float]:
        """Polynomial fit Bl(z) to find Bl(0), and analyse.

        All metrics are normalised to the true value of Bl at the weighing
        position.

        Args:
            deg: The polynomail degree to be fitted to the measured data.

        Returns:
            Tuple of metrics including: rel_bias - the bias of the measured
            Bl, rel_stddev - the stddev of the measured Bl across multiple
            runs, and rel_avg_residuals - the average across all runs of the
            MAE (mean average error) of the fit propogated and normalised to
            Bl. Params is the last item: a list of arrays with each array
            containing the coefficients for the run corresponding to its
            position in the list. Highest order coefficients are first.
        """
        bls_meas = []
        residuals = []
        params = []
        for u_result, v_result, disp_start, disp_end in zip(
            self.u_results,
            self.v_results,
            self.displacement_start,
            self.displacement_end,
        ):
            bls_meas_of_z = u_result / v_result

            # Measurement position at the middle integration range
            # Alternatively could do middle of the integration time. For linear
            # this would be the same.
            # There is some kind of correction to be done here for Bl(z)!=Bl(0)
            meas_pos = disp_start + (disp_end - disp_start) / 2

            coeffs = np.polyfit(meas_pos, bls_meas_of_z, deg=deg)
            poly = np.poly1d(coeffs)
            bls_meas.append(np.polyval(poly, self.weighing_pos))
            residuals.append(
                np.average(abs(bls_meas_of_z - np.polyval(poly, meas_pos)))
            )
            params.append(coeffs)

        rel_bias = (
            self.bl_weighing_pos - np.average(bls_meas)
        ) / self.bl_weighing_pos
        rel_stddev = np.std(bls_meas) / self.bl_weighing_pos
        rel_avg_residuals = np.average(residuals) / self.bl_weighing_pos

        return (rel_bias, rel_stddev, rel_avg_residuals), params


# TODO LIST: ##=priority
# Put in proper file structure?
## Finish interferometer tests
# Check types, comments, and docstrings
## Finish tests
# Add preconditions, check types
#    - write function to convert arraylike to ndarray or float or something
# Run better linter
# Run typing
# Add more features
# - e.g. 4 param sine fit, frequency counter for sine wave etc.
## Write README (with citations)
# Put analyses in their own object/objects
# Finish TODOs
# Add a re-randomise everything option to MovingModeExperiment if necessary
