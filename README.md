# camera_doppler_effect


# Doppler Effect Demonstration — Raspberry Pi

A hands-on physics demonstration built on a Raspberry Pi that simulates the **Doppler effect** — the way a sound's pitch changes as its source moves closer to or farther from an observer (like an ambulance siren rising in pitch as it approaches, then dropping as it passes).

This project includes **two independent working implementations**, built to compare a simple approach against a more advanced one:

| | IR Sensor Version | Camera Version |
|---|---|---|
| Detection method | Digital IR proximity sensor | USB webcam + OpenCV (motion + skin-color detection) |
| Distance sensing | Binary (near / far) | Continuous, calibrated per session |
| Complexity | Simple circuit, simple code | Computer vision, real-time processing |
| Reliability | Very stable, fixed threshold | Depends on lighting/calibration |
| Extra features | — | Live distance meter, session graphs, CSV export, live physics overlay |

---

## Table of Contents

- [How It Works](#how-it-works)
- [The Physics](#the-physics)
- [Hardware Used](#hardware-used)
- [Wiring](#wiring)
- [Setup](#setup)
- [Running the Demo](#running-the-demo)
- [Camera Version Features](#camera-version-features)
- [Repository Structure](#repository-structure)
- [Known Limitations](#known-limitations)
- [Author](#author)

---

## How It Works

Both versions follow the same core idea: **detect how close a hand is, then map that distance to an LED blink/pulse rate and an audio pitch** — faster blinking and higher pitch when the hand is close (simulating an "approaching" source), slower blinking and lower pitch when far (simulating a "receding" source).

**IR Sensor Version** — a digital IR proximity module detects whether an object is within a fixed range (set by an onboard sensitivity trimmer). This gives a simple two-state (near/far) simulation.

**Camera Version** — a USB webcam continuously estimates hand distance using OpenCV: it detects the hand via motion against a learned background, samples that person's actual skin color to build a personalized detection range for the session, then tracks the hand's blob size in each frame (a bigger blob = a closer hand) to produce a smooth, continuously varying reading.

---

## The Physics

The real Doppler effect formula:

```
f_observed = f_source × v_sound / (v_sound ∓ v_source)
```

- `f_source` — original frequency of the sound (Hz)
- `f_observed` — frequency heard by the observer (Hz)
- `v_sound` — speed of sound in air (~343 m/s)
- `v_source` — speed of the source relative to the observer (minus when approaching, plus when receding)

The camera version actually **computes this formula live**, estimating a pseudo-velocity from how fast the detected hand distance changes between frames, and calculates the real theoretical frequency shift that motion would cause.

**Important honesty check, printed by the program itself:** hand-speed movement (a few cm/s) produces a real shift of only a tiny fraction of a Hz — genuinely imperceptible to the human ear. The LED and audio tone in this demo are **deliberately exaggerated** across a much wider range specifically so the underlying concept — frequency changing with relative motion — is actually visible and audible. This is stated explicitly in the program's output, not glossed over.

---

## Hardware Used

**Shared:**
- Raspberry Pi 3 Model B
- LED (any color) + 1kΩ resistor
- Earphones/speaker (via 3.5mm audio jack)

**IR Sensor Version only:**
- Digital IR obstacle/proximity sensor module (3-pin: GND, V, OUT)

**Camera Version only:**
- Any USB webcam

---

## Wiring

### IR Sensor Version
| Component | Raspberry Pi Pin |
|---|---|
| IR module — V | Pin 1 (3.3V) |
| IR module — GND | Pin 9 (GND) |
| IR module — OUT | Pin 13 (GPIO27) |
| LED anode → 1kΩ resistor | Pin 12 (GPIO18) |
| LED cathode | Pin 14 (GND) |

### Camera Version
| Component | Raspberry Pi Pin |
|---|---|
| LED anode → 1kΩ resistor | Pin 12 (GPIO18) |
| LED cathode | Pin 14 (GND) |
| USB webcam | Any USB port |

Full labeled wiring diagrams are included in this repository (see [Repository Structure](#repository-structure)).

---

## Setup

### Common (both versions)
```bash
sudo apt update
sudo apt install alsa-utils
sudo raspi-config   # System Options -> Audio -> Headphones
```

### IR Sensor Version
```bash
python3 -c "import numpy" || sudo apt install python3-numpy
```

### Camera Version
```bash
sudo apt install python3-opencv python3-matplotlib
python3 -c "import numpy" || sudo apt install python3-numpy
```

---

## Running the Demo

**IR Sensor Version:**
```bash
python3 doppler_demo.py
```

**Camera Version:**
```bash
python3 camera_doppler_demo.py
```
On launch, the camera version runs a short guided calibration:
1. **Move your hand completely out of frame** when prompted — it learns the background
2. **Hold your hand close to the camera** when prompted — it learns your hand's size and color

This means it recalibrates for whoever is using it, rather than being tuned to one specific person.

Stop either program anytime with `Ctrl+C`.

---

## Camera Version Features

- **Motion + color-based detection** that adapts to different people's skin tones each session, rather than a fixed assumption
- **Auto-retry calibration** — automatically retries once if the first attempt doesn't produce reliable results
- **Live distance meter** — a text bar in the terminal showing relative hand distance at a glance
- **Color-coded terminal output** — red (close), yellow (mid-range), blue (far/not detected)
- **Live physics overlay** — shows the real, calculated Doppler shift alongside the demo's exaggerated tone, every reading
- **Session graph (PNG)** — auto-generated on exit, plotting hand distance and blink interval over time, with the exact recording date/time on the chart
- **CSV data export** — every session's raw readings saved for further analysis in Excel/Sheets
- **End-of-session summary** — duration, detection rate, calibration thresholds, and range statistics printed on exit
- **Full physics explanation** — printed at the end of every run, explaining the Doppler formula and reporting the largest real shift measured that session

---

## Repository Structure

```
├── doppler_demo.py                    # IR sensor version
├── camera_doppler_demo.py             # Camera (OpenCV) version
├── wiring_1_IR_sensor_version.svg     # IR version wiring diagram
├── wiring_2_camera_version.svg        # Camera version wiring diagram
├── checklist_1_IR_sensor.md           # Pre-demo checklist (IR version)
├── checklist_2_camera.md              # Pre-demo checklist (camera version)
├── troubleshooting_1_IR_sensor.md     # Troubleshooting guide (IR version)
├── troubleshooting_2_camera.md        # Troubleshooting guide (camera version)
└── README.md                          # This file
```

---

## Known Limitations

- **IR sensor version**: fixed-threshold detection (near/far only), not truly continuous — an intentional, stated simplification of a basic digital sensor's capability
- **Camera version**: detection quality depends on lighting conditions and requires a one-time calibration per session/person; performance is modest on the Raspberry Pi 3's hardware, particularly for real-time video processing

---

## Author

Built by Ram, Class 9, (with some help by @Claude, of course) as part of a practical project for The INNOVATION CLUB, JMPS, JAMMU & KASHMIR (https://www.jodhamalschool.org/) — including both hardware assembly, wiring debugging, and software development (IR sensor logic and OpenCV-based computer vision).
