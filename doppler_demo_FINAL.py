"""
Doppler Effect Demonstration - Raspberry Pi + IR Sensor + LED + Speaker
-------------------------------------------------------------------------
CONCEPT BEING SIMULATED:
The Doppler effect describes how the observed frequency of a wave changes
when the source and observer move relative to each other (e.g. an
ambulance siren sounding higher-pitched as it approaches, lower as it
moves away).

IMPORTANT LIMITATION TO NOTE IN YOUR REPORT:
A basic digital IR obstacle sensor only outputs HIGH or LOW - it detects
"object within range" vs "no object", not the actual distance. So this
script demonstrates the CONCEPT of frequency change using two discrete
states (NEAR / FAR), rather than a smooth, continuously variable
frequency you'd get from a continuous sensor (like the camera-based
version of this project). This is a fair simplification for a school
demo, but is worth stating explicitly so your explanation is technically
accurate.

AUDIO OUTPUT:
Tones play through the Pi's own 3.5mm audio jack (no physical buzzer
needed). Plug earphones or a small speaker into the Pi's audio port
before running this script.

WIRING (matches the CONFIRMED WORKING setup after hardware testing):
  IR module V    -> Pi Pin 1  (3.3V)
  IR module GND  -> Pi Pin 9  (GND)
  IR module OUT  -> Pi Pin 13 (GPIO27)   <-- NOT GPIO17/Pin 11; that pin
                                              tested faulty on this board
  LED anode      -> 1k ohm resistor -> Pi Pin 12 (GPIO18)
  LED cathode    -> Pi Pin 14 (GND)
  Speaker/earphones -> Pi's 3.5mm audio jack
  (LED is direct-wired, no breadboard - a breadboard row tested unreliable)

SETUP (run once on a freshly flashed Raspberry Pi OS):
  1. Update and install audio tools:
       sudo apt update
       sudo apt install alsa-utils
  2. Check numpy (often already installed on Raspberry Pi OS):
       python3 -c "import numpy; print(numpy.__version__)"
     If missing: sudo apt install python3-numpy
  3. Force audio output to the 3.5mm jack (not HDMI):
       sudo raspi-config -> System Options -> Audio -> Headphones
  4. Test your speaker works at all:
       speaker-test -t sine -f 440 -l 1
  5. GPIO note for current Raspberry Pi OS (Bookworm and later): the
     GPIO backend changed from the old RPi.GPIO library to lgpio.
     gpiozero (used below) handles this automatically in almost all
     cases. If you get a GPIO-related error on first run, install the
     backend explicitly:
       sudo apt install python3-lgpio

Requires: gpiozero, numpy
"""

from gpiozero import DigitalInputDevice, LED
import numpy as np
import wave
import os
import time

# ---------------------------------------------------------------------
# GPIO PIN SETUP (BCM numbering, matches gpiozero defaults)
# ---------------------------------------------------------------------
IR_PIN = 27   # confirmed working pin after testing (GPIO17 was faulty)
LED_PIN = 18

ir_sensor = DigitalInputDevice(IR_PIN)
led = LED(LED_PIN)

# ---------------------------------------------------------------------
# CONFIGURATION - tune these to make the demo more/less dramatic
# ---------------------------------------------------------------------
NEAR_BLINK_INTERVAL = 0.1   # seconds - fast blink when object is near
FAR_BLINK_INTERVAL = 0.6    # seconds - slow blink when object is far
NEAR_FREQ_HZ = 900          # higher pitch = "approaching"
FAR_FREQ_HZ = 300           # lower pitch = "moving away"
TONE_DURATION = 0.15        # seconds - length of each beep

# Most IR modules pull the output LOW when an object IS detected
# (active-low). If your LED/sound behave backwards, flip this flag.
ACTIVE_LOW = True

TONE_DIR = "/tmp"
NEAR_TONE_FILE = os.path.join(TONE_DIR, "near_tone.wav")
FAR_TONE_FILE = os.path.join(TONE_DIR, "far_tone.wav")


def generate_tone_file(filename, frequency, duration, volume=0.5, sample_rate=44100):
    """Generate a simple sine-wave WAV file at the given frequency."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = volume * np.sin(2 * np.pi * frequency * t)
    audio = (waveform * 32767).astype(np.int16)

    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())


def play_tone_file(filename):
    """Play a WAV file through the Pi's audio output using aplay."""
    os.system("aplay -q {}".format(filename))


def object_is_near():
    """Return True if the IR sensor currently detects an object."""
    if ACTIVE_LOW:
        return ir_sensor.value == 0
    return ir_sensor.value == 1


def run_demo():
    print("Generating tone files...")
    generate_tone_file(NEAR_TONE_FILE, NEAR_FREQ_HZ, TONE_DURATION)
    generate_tone_file(FAR_TONE_FILE, FAR_FREQ_HZ, TONE_DURATION)

    print("Doppler Effect Demo running. Press CTRL+C to stop.")
    try:
        while True:
            loop_start = time.time()
            near = object_is_near()

            if near:
                interval = NEAR_BLINK_INTERVAL
                tone_file = NEAR_TONE_FILE
            else:
                interval = FAR_BLINK_INTERVAL
                tone_file = FAR_TONE_FILE

            led.on()
            play_tone_file(tone_file)

            # Compensate for time already spent this iteration (audio
            # playback) so the blink period matches 'interval' more
            # closely instead of silently drifting longer.
            elapsed = time.time() - loop_start
            remaining = max(0, interval - elapsed)
            time.sleep(remaining)

            led.off()
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    finally:
        led.off()


if __name__ == "__main__":
    run_demo()
