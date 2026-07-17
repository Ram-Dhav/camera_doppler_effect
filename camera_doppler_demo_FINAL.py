"""
Doppler Effect Demonstration - Raspberry Pi + USB Webcam + LED + Speaker
------------------------------------------------------------------------------
Detects hand distance continuously via webcam and maps it to LED pulse
rate and audio pitch, simulating the Doppler effect. Works across
different people's skin tones via per-session calibration (motion-based
hand detection, not a fixed color guess).

FEATURES:
  - Motion-based hand detection + per-session skin-color calibration
    (recalibrates for whoever is using it)
  - Auto-retry calibration if the first attempt fails validation
  - Clear on-screen warning before background capture, so the person
    knows to move their hand fully out of frame
  - OpenCV 3.x/4.x compatibility
  - ROI as a percentage of actual camera resolution (works regardless
    of what resolution the webcam actually provides), camera warm-up
    frames before readings are trusted
  - Exponential smoothing on distance readings (reduces frame-to-frame
    jitter)
  - 16-step resolution, live color-coded distance meter in the terminal
  - Live Doppler physics overlay every reading: shows the REAL frequency
    shift your hand's actual speed would cause, next to the demo's
    deliberately exaggerated tone
  - Smooth PWM LED fade instead of a hard on/off blink
  - On exit (Ctrl+C): prints a session summary, exports all readings to
    CSV, saves a labeled PNG graph (with recording date/time on it), and
    prints a full explanation of the Doppler formula used

WIRING:
  LED anode -> 1k ohm resistor -> GPIO18 (Pi Pin 12)
  LED cathode -> GND (Pi Pin 14)
  USB webcam -> any USB port
  Speaker/earphones -> Pi's 3.5mm audio jack

SETUP:
  sudo apt update
  sudo apt install python3-opencv python3-matplotlib alsa-utils
  python3 -c "import numpy" || sudo apt install python3-numpy
  sudo raspi-config -> System Options -> Audio -> Headphones

Requires: gpiozero, opencv (cv2), numpy, matplotlib
"""

from gpiozero import PWMLED
import cv2
import numpy as np
import wave
import os
import time
import datetime
import csv
import json

import matplotlib
matplotlib.use("Agg")  # headless - no display attached over SSH
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# GPIO PIN SETUP
# ---------------------------------------------------------------------
LED_PIN = 18
led = PWMLED(LED_PIN)

# ---------------------------------------------------------------------
# CAMERA CONFIGURATION
# ---------------------------------------------------------------------
CAMERA_INDEX = 0
REQUESTED_WIDTH = 320
REQUESTED_HEIGHT = 240
ROI_MARGIN_FRACTION = 0.10
NUM_FRAMES_TO_WARM_UP = 10

# ---------------------------------------------------------------------
# SKIN COLOR RANGE - determined per session by calibration (see below).
# ---------------------------------------------------------------------
FALLBACK_YCRCB_LOWER = np.array([0, 133, 77], dtype=np.uint8)
FALLBACK_YCRCB_UPPER = np.array([255, 173, 127], dtype=np.uint8)
current_ycrcb_lower = FALLBACK_YCRCB_LOWER.copy()
current_ycrcb_upper = FALLBACK_YCRCB_UPPER.copy()

# ---------------------------------------------------------------------
# CALIBRATION CONFIGURATION
# ---------------------------------------------------------------------
CALIBRATION_COUNTDOWN_SEC = 3
CALIBRATION_SAMPLE_FRAMES = 20
MOTION_DIFF_THRESHOLD = 25
MIN_MOTION_BLOB_AREA = 400
COLOR_RANGE_MARGIN_CR = 8
COLOR_RANGE_MARGIN_CB = 8
BACKGROUND_SAFETY_MARGIN = 1.4
MAX_CALIBRATION_ATTEMPTS = 2
MIN_ACCEPTABLE_BRIGHTNESS = 60  # 0-255 scale; below this, warn about dim lighting

FALLBACK_MIN_HAND_AREA = 800
FALLBACK_MAX_HAND_AREA = 15000

# ---------------------------------------------------------------------
# BLINK / TONE / SMOOTHING CONFIGURATION
# ---------------------------------------------------------------------
NEAR_BLINK_INTERVAL = 0.08
FAR_BLINK_INTERVAL = 0.6
NEAR_FREQ_HZ = 1000
FAR_FREQ_HZ = 300
TONE_DURATION = 0.12
NUM_TONE_STEPS = 16
SMOOTHING_ALPHA = 0.35

# ---------------------------------------------------------------------
# GRAPH / LOGGING CONFIGURATION
# ---------------------------------------------------------------------
GRAPH_OUTPUT_DIR = os.path.expanduser("~")  # works regardless of your actual username
DISTANCE_BAR_WIDTH = 24
CALIBRATION_CACHE_FILE = os.path.join(GRAPH_OUTPUT_DIR, ".doppler_calibration.json")

# ---------------------------------------------------------------------
# PHYSICS OVERLAY CONFIGURATION
# ---------------------------------------------------------------------
SPEED_OF_SOUND_MPS = 343.0
PSEUDO_DISTANCE_FAR_CM = 50.0
PSEUDO_DISTANCE_NEAR_CM = 5.0

TONE_DIR = "/tmp"
tone_files = []

# Set once in run_demo() based on the camera's ACTUAL resolution vs what
# was requested. Used to scale pixel-area constants (tuned for 320x240)
# so they still make sense if the webcam gives a different resolution.
resolution_scale_factor = 1.0

# ANSI color codes for terminal output (no CPU cost, just text formatting)
COLOR_RED = "\033[91m"      # hand close
COLOR_YELLOW = "\033[93m"   # hand mid-range
COLOR_BLUE = "\033[94m"     # hand far / not detected
COLOR_RESET = "\033[0m"


def get_color_for_step(step, detected):
    if not detected:
        return COLOR_BLUE
    fraction = step / float(NUM_TONE_STEPS - 1)
    if fraction >= 0.66:
        return COLOR_RED
    elif fraction >= 0.33:
        return COLOR_YELLOW
    return COLOR_BLUE


def print_startup_banner():
    print("")
    print("+" + "-" * 50 + "+")
    print("|" + "DOPPLER EFFECT DEMONSTRATION".center(50) + "|")
    print("|" + "Camera-based hand distance sensing".center(50) + "|")
    print("+" + "-" * 50 + "+")
    print("  {}".format(datetime.datetime.now().strftime("%A, %d %B %Y - %H:%M")))
    print("  Move your hand closer/farther from the camera to")
    print("  change the LED pulse rate and audio pitch.")
    print("")


def generate_tone_file(filename, frequency, duration, volume=0.5, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = volume * np.sin(2 * np.pi * frequency * t)
    audio = (waveform * 32767).astype(np.int16)
    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())


def generate_all_tones():
    print("Generating {} tone steps...".format(NUM_TONE_STEPS))
    for i in range(NUM_TONE_STEPS):
        fraction = i / float(NUM_TONE_STEPS - 1)
        freq = FAR_FREQ_HZ + fraction * (NEAR_FREQ_HZ - FAR_FREQ_HZ)
        filename = os.path.join(TONE_DIR, "tone_{}.wav".format(i))
        generate_tone_file(filename, freq, TONE_DURATION)
        tone_files.append(filename)


def play_tone_file(filename):
    """Play a WAV file through the Pi's audio output. Blocking on purpose:
    keeps exactly one reading paired with one beep, and avoids spawning
    overlapping background processes that were competing with the camera
    for CPU time on the Pi 3."""
    os.system("aplay -q {}".format(filename))


def find_contours_compat(mask):
    result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(result) == 3:
        _, contours, _ = result
    else:
        contours, _ = result
    return contours


def get_roi_coords(frame_width, frame_height):
    x1 = int(frame_width * ROI_MARGIN_FRACTION)
    y1 = int(frame_height * ROI_MARGIN_FRACTION)
    x2 = int(frame_width * (1 - ROI_MARGIN_FRACTION))
    y2 = int(frame_height * (1 - ROI_MARGIN_FRACTION))
    return x1, y1, x2, y2


def get_skin_mask(roi_frame):
    ycrcb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, current_ycrcb_lower, current_ycrcb_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask


def get_hand_area_from_roi(roi_frame):
    mask = get_skin_mask(roi_frame)
    contours = find_contours_compat(mask)
    if not contours:
        return 0
    largest = max(contours, key=cv2.contourArea)
    return cv2.contourArea(largest)


def get_hand_area(frame, roi_coords):
    x1, y1, x2, y2 = roi_coords
    roi = frame[y1:y2, x1:x2]
    return get_hand_area_from_roi(roi)


def get_largest_motion_contour(roi_frame, background_gray_roi):
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, background_gray_roi)
    _, thresh = cv2.threshold(diff, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours = find_contours_compat(thresh)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_MOTION_BLOB_AREA * resolution_scale_factor:
        return None, None
    mask = np.zeros(thresh.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return largest, mask


def countdown(seconds):
    for i in range(seconds, 0, -1):
        print("  {}...".format(i))
        time.sleep(1)


def check_lighting(frame):
    """Warn upfront if the scene looks too dim for reliable skin-color detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if brightness < MIN_ACCEPTABLE_BRIGHTNESS:
        print("")
        print("+" + "-" * 56 + "+")
        print("|" + "LIGHTING WARNING".center(56) + "|")
        print("+" + "-" * 56 + "+")
        print("  Current brightness: {:.0f}/255 (looks dim)".format(brightness))
        print("  Detection works best with more even, brighter lighting.")
        print("  Consider turning on a light or facing a window.")
        print("+" + "-" * 56 + "+")
        print("")
    return brightness


def save_calibration_cache(min_area, max_area):
    data = {
        "ycrcb_lower": current_ycrcb_lower.tolist(),
        "ycrcb_upper": current_ycrcb_upper.tolist(),
        "min_area": min_area,
        "max_area": max_area,
        "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(CALIBRATION_CACHE_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print("Could not save calibration cache ({})".format(e))


def load_calibration_cache():
    if not os.path.exists(CALIBRATION_CACHE_FILE):
        return None
    try:
        with open(CALIBRATION_CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def offer_cached_calibration():
    """
    If a previous calibration was saved, ask whether to reuse it instead
    of recalibrating from scratch. Returns (min_area, max_area) if the
    person chose to reuse it, or None if they declined / none exists.
    """
    global current_ycrcb_lower, current_ycrcb_upper

    cached = load_calibration_cache()
    if cached is None:
        return None

    print("")
    print("Found a saved calibration from: {}".format(cached.get("saved_at", "unknown time")))
    try:
        choice = input("Use this saved calibration instead of recalibrating? (y/n): ").strip().lower()
    except EOFError:
        choice = "n"

    if choice != "y":
        return None

    current_ycrcb_lower = np.array(cached["ycrcb_lower"], dtype=np.uint8)
    current_ycrcb_upper = np.array(cached["ycrcb_upper"], dtype=np.uint8)
    print("Loaded saved calibration.")
    print("")
    return cached["min_area"], cached["max_area"]


def print_calibration_warning():
    print("")
    print("+" + "-" * 56 + "+")
    print("|" + "!  IMPORTANT  !".center(56) + "|")
    print("|" + "Move your hand COMPLETELY out of camera view NOW".center(56) + "|")
    print("|" + "(background capture starts after the countdown)".center(56) + "|")
    print("+" + "-" * 56 + "+")
    print("")


def attempt_calibration(cap, roi_coords):
    """One calibration attempt. Returns (min_area, max_area, success_bool)."""
    global current_ycrcb_lower, current_ycrcb_upper
    x1, y1, x2, y2 = roi_coords

    print_calibration_warning()
    print("Step 1: Move your hand OUT of the camera's view completely.")
    countdown(CALIBRATION_COUNTDOWN_SEC)
    print("Capturing background...")

    background_frames = []
    for _ in range(CALIBRATION_SAMPLE_FRAMES):
        ret, frame = cap.read()
        if ret:
            background_frames.append(frame[y1:y2, x1:x2])
        time.sleep(0.03)

    if not background_frames:
        return FALLBACK_MIN_HAND_AREA, FALLBACK_MAX_HAND_AREA, False

    background_gray_stack = np.stack(
        [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in background_frames]
    )
    background_gray_avg = np.mean(background_gray_stack, axis=0).astype(np.uint8)

    print("")
    print("Step 2: Hold your hand as CLOSE to the camera as you will during the demo.")
    print("(Keep it inside the center of the frame)")
    countdown(CALIBRATION_COUNTDOWN_SEC)
    print("Measuring your hand...")

    near_areas = []
    sampled_cr = []
    sampled_cb = []

    for _ in range(CALIBRATION_SAMPLE_FRAMES):
        ret, frame = cap.read()
        if not ret:
            continue
        roi_frame = frame[y1:y2, x1:x2]
        contour, mask = get_largest_motion_contour(roi_frame, background_gray_avg)
        if contour is None:
            time.sleep(0.03)
            continue
        near_areas.append(cv2.contourArea(contour))
        ycrcb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2YCrCb)
        hand_pixels = ycrcb[mask == 255]
        if len(hand_pixels) > 0:
            sampled_cr.extend(hand_pixels[:, 1].tolist())
            sampled_cb.extend(hand_pixels[:, 2].tolist())
        time.sleep(0.03)

    if not near_areas or not sampled_cr:
        return FALLBACK_MIN_HAND_AREA, FALLBACK_MAX_HAND_AREA, False

    cr_low = max(0, int(np.percentile(sampled_cr, 5)) - COLOR_RANGE_MARGIN_CR)
    cr_high = min(255, int(np.percentile(sampled_cr, 95)) + COLOR_RANGE_MARGIN_CR)
    cb_low = max(0, int(np.percentile(sampled_cb, 5)) - COLOR_RANGE_MARGIN_CB)
    cb_high = min(255, int(np.percentile(sampled_cb, 95)) + COLOR_RANGE_MARGIN_CB)

    current_ycrcb_lower = np.array([0, cr_low, cb_low], dtype=np.uint8)
    current_ycrcb_upper = np.array([255, cr_high, cb_high], dtype=np.uint8)

    print("Personalized skin-color range set: Cr[{},{}] Cb[{},{}]".format(
        cr_low, cr_high, cb_low, cb_high))

    near_level = float(np.percentile(near_areas, 90))
    background_color_areas = [get_hand_area_from_roi(f) for f in background_frames]
    background_level = max(background_color_areas) if background_color_areas else 0

    min_area = background_level * BACKGROUND_SAFETY_MARGIN
    max_area = near_level

    if max_area <= min_area * 1.2:
        return FALLBACK_MIN_HAND_AREA, FALLBACK_MAX_HAND_AREA, False

    return min_area, max_area, True


def run_calibration(cap, roi_coords):
    print("")
    print("=== CALIBRATION ===")

    for attempt in range(1, MAX_CALIBRATION_ATTEMPTS + 1):
        if attempt > 1:
            print("")
            print("Calibration attempt {} didn't succeed - retrying automatically...".format(attempt - 1))
            print("(Tip: better/more even lighting helps a lot)")
            print("")

        min_area, max_area, success = attempt_calibration(cap, roi_coords)

        if success:
            print("")
            print("=== CALIBRATION COMPLETE (attempt {}) ===".format(attempt))
            print("MIN_HAND_AREA = {:.0f}   MAX_HAND_AREA = {:.0f}".format(min_area, max_area))
            print("")
            save_calibration_cache(min_area, max_area)
            return min_area, max_area

    print("")
    print("WARNING: Calibration failed after {} attempts.".format(MAX_CALIBRATION_ATTEMPTS))
    print("Using generic fallback thresholds - detection may be less accurate.")
    print("")
    return min_area, max_area


def map_area_to_step(area, min_area, max_area):
    if area < min_area:
        return 0
    clamped = min(area, max_area)
    span = float(max_area - min_area)
    fraction = (clamped - min_area) / span if span > 0 else 0
    step = int(round(fraction * (NUM_TONE_STEPS - 1)))
    return max(0, min(step, NUM_TONE_STEPS - 1))


def step_to_blink_interval(step):
    fraction = step / float(NUM_TONE_STEPS - 1)
    return FAR_BLINK_INTERVAL - fraction * (FAR_BLINK_INTERVAL - NEAR_BLINK_INTERVAL)


def make_distance_bar(fraction, width=DISTANCE_BAR_WIDTH):
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    bar = "|" * filled + "-" * (width - filled)
    return "[{}] {:3.0f}%".format(bar, fraction * 100)


def fraction_to_pseudo_distance_cm(fraction):
    fraction = max(0.0, min(1.0, fraction))
    return PSEUDO_DISTANCE_FAR_CM - fraction * (PSEUDO_DISTANCE_FAR_CM - PSEUDO_DISTANCE_NEAR_CM)


def compute_real_doppler_freq(base_freq, velocity_mps):
    if velocity_mps > 0:
        return base_freq * SPEED_OF_SOUND_MPS / (SPEED_OF_SOUND_MPS - velocity_mps)
    elif velocity_mps < 0:
        return base_freq * SPEED_OF_SOUND_MPS / (SPEED_OF_SOUND_MPS + abs(velocity_mps))
    return base_freq


def build_session_summary_lines(timestamps, areas, intervals, distance_fractions, min_area, max_area):
    if len(timestamps) < 2:
        return ["Not enough data collected for a summary."]

    duration = timestamps[-1] - timestamps[0]
    detected_fractions = [d for d in distance_fractions if d > 0]

    lines = []
    lines.append("+" + "-" * 50 + "+")
    lines.append("|" + "SESSION SUMMARY".center(50) + "|")
    lines.append("+" + "-" * 50 + "+")
    lines.append("  Duration            : {:.1f} seconds".format(duration))
    lines.append("  Total readings       : {}".format(len(timestamps)))
    lines.append("  Readings per second  : {:.1f}".format(len(timestamps) / duration if duration > 0 else 0))
    lines.append("  Hand detected in     : {:.0f}% of readings".format(
        100.0 * len(detected_fractions) / len(distance_fractions)))
    lines.append("  Closest reading      : {:.0f}%".format(max(distance_fractions) * 100))
    lines.append("  Calibrated area range: {:.0f} - {:.0f} px".format(min_area, max_area))
    lines.append("  Blink interval range : {:.3f}s (fastest) - {:.3f}s (slowest)".format(
        min(intervals), max(intervals)))
    lines.append("+" + "-" * 50 + "+")
    return lines


def print_session_summary(timestamps, areas, intervals, distance_fractions, min_area, max_area):
    lines = build_session_summary_lines(timestamps, areas, intervals, distance_fractions, min_area, max_area)
    print("")
    for line in lines:
        print(line)
    print("")
    return lines


def build_physics_explanation_lines(session_real_shifts):
    max_shift = max((abs(s) for s in session_real_shifts), default=0)

    lines = []
    lines.append("+" + "-" * 62 + "+")
    lines.append("|" + "THE PHYSICS BEHIND THIS DEMO".center(62) + "|")
    lines.append("+" + "-" * 62 + "+")
    lines.append("")
    lines.append("  The Doppler effect formula used in this program:")
    lines.append("")
    lines.append("      f_observed = f_source * v_sound / (v_sound -+ v_source)")
    lines.append("")
    lines.append("  Where:")
    lines.append("    f_source   = frequency of the original sound (Hz)")
    lines.append("    f_observed = frequency heard by the observer (Hz)")
    lines.append("    v_sound    = speed of sound in air (~343 m/s)")
    lines.append("    v_source   = speed of the source relative to the observer")
    lines.append("                 (m/s). Use MINUS when approaching, PLUS when")
    lines.append("                 receding.")
    lines.append("")
    lines.append("  This program estimates v_source from how fast the detected")
    lines.append("  hand distance changes between camera frames, then plugs it")
    lines.append("  into the formula above to compute a REAL frequency shift.")
    lines.append("")
    lines.append("  Largest real shift measured this session: {:.5f} Hz".format(max_shift))
    lines.append("  (This is why it's inaudible on its own - hand-speed motion")
    lines.append("   produces a shift of a tiny fraction of a Hz. The LED and")
    lines.append("   audio tone in this demo are DELIBERATELY EXAGGERATED across")
    lines.append("   a much wider range so the underlying concept - frequency")
    lines.append("   changing with relative motion - is actually perceivable.)")
    lines.append("+" + "-" * 62 + "+")
    return lines


def print_physics_explanation(session_real_shifts):
    lines = build_physics_explanation_lines(session_real_shifts)
    for line in lines:
        print(line)
    print("")
    return lines


def save_session_report(summary_lines, physics_lines):
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(GRAPH_OUTPUT_DIR, "doppler_session_{}.txt".format(timestamp_str))
    try:
        with open(filename, "w") as f:
            f.write("\n".join(summary_lines))
            f.write("\n\n")
            f.write("\n".join(physics_lines))
            f.write("\n")
        print("Session report (text) saved to: {}".format(filename))
    except Exception as e:
        print("Could not save session report ({})".format(e))


def export_session_csv(timestamps, areas, distance_fractions, intervals, demo_freqs, real_shifts):
    if len(timestamps) < 2:
        print("Not enough data collected to export a CSV.")
        return

    t0 = timestamps[0]
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(GRAPH_OUTPUT_DIR, "doppler_session_{}.csv".format(timestamp_str))

    try:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_sec", "hand_area_px", "distance_fraction",
                              "blink_interval_sec", "demo_tone_hz", "real_doppler_shift_hz"])
            for i in range(len(timestamps)):
                writer.writerow([
                    "{:.3f}".format(timestamps[i] - t0),
                    "{:.1f}".format(areas[i]),
                    "{:.3f}".format(distance_fractions[i]),
                    "{:.3f}".format(intervals[i]),
                    "{:.1f}".format(demo_freqs[i]),
                    "{:.6f}".format(real_shifts[i]),
                ])
        print("Session data (CSV) saved to: {}".format(filename))
    except Exception as e:
        print("Could not save CSV ({})".format(e))


def generate_session_graph(timestamps, areas, intervals):
    if len(timestamps) < 2:
        print("Not enough data collected to generate a graph.")
        return

    t0 = timestamps[0]
    rel_times = [t - t0 for t in timestamps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    run_datetime_str = datetime.datetime.fromtimestamp(t0).strftime("%d %B %Y, %H:%M:%S")
    fig.suptitle("Doppler Effect Demo - Session Recording", fontsize=14, fontweight="bold")
    ax1.set_title("Recorded: {}".format(run_datetime_str), fontsize=10, color="#555555")

    ax1.plot(rel_times, areas, color="#2e7d32")
    ax1.set_ylabel("Detected hand area (pixels)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(rel_times, intervals, color="#1565c0")
    ax2.set_ylabel("Blink interval (seconds)")
    ax2.set_xlabel("Time (seconds since start)")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(GRAPH_OUTPUT_DIR, "doppler_session_{}.png".format(timestamp_str))
    try:
        plt.savefig(filename, dpi=120)
        print("Session graph saved to: {}".format(filename))
    except Exception as e:
        print("Could not save graph ({}). Trying /tmp instead...".format(e))
        fallback = os.path.join("/tmp", "doppler_session_{}.png".format(timestamp_str))
        plt.savefig(fallback, dpi=120)
        print("Session graph saved to: {}".format(fallback))


def run_demo():
    print_startup_banner()
    generate_all_tones()

    print("Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open camera. Check 'ls /dev/video*' and CAMERA_INDEX.")
        return

    print("Warming up camera ({} frames)...".format(NUM_FRAMES_TO_WARM_UP))
    for _ in range(NUM_FRAMES_TO_WARM_UP):
        cap.read()
        time.sleep(0.05)

    ret, sample_frame = cap.read()
    if not ret:
        print("ERROR: Could not read a frame from the camera.")
        cap.release()
        return
    actual_height, actual_width = sample_frame.shape[:2]
    print("Actual camera resolution: {}x{}".format(actual_width, actual_height))

    roi_coords = get_roi_coords(actual_width, actual_height)
    print("Detection zone (ROI): {}".format(roi_coords))

    check_lighting(sample_frame)

    cached_result = offer_cached_calibration()
    if cached_result is not None:
        min_area, max_area = cached_result
    else:
        min_area, max_area = run_calibration(cap, roi_coords)

    print("Doppler Effect Demo running. Press CTRL+C to stop.")
    print("")

    smoothed_area = 0.0
    session_timestamps = []
    session_areas = []
    session_intervals = []
    session_demo_freqs = []
    session_real_shifts = []
    session_distance_fractions = []

    prev_distance_cm = None
    prev_time = None
    frame_count = 0

    try:
        while True:
            loop_start = time.time()
            frame_count += 1

            ret, frame = cap.read()
            if not ret:
                print("Warning: failed to read frame, retrying...")
                time.sleep(0.2)
                continue

            raw_area = get_hand_area(frame, roi_coords)
            smoothed_area = (SMOOTHING_ALPHA * raw_area) + ((1 - SMOOTHING_ALPHA) * smoothed_area)

            step = map_area_to_step(smoothed_area, min_area, max_area)
            interval = step_to_blink_interval(step)
            detected = smoothed_area >= min_area

            distance_fraction = 0.0
            if max_area > min_area:
                distance_fraction = (smoothed_area - min_area) / float(max_area - min_area)
            distance_fraction = max(0.0, min(1.0, distance_fraction))

            # --- Physics overlay (values computed every frame, printed occasionally) ---
            # Velocity tracking resets whenever the hand isn't detected, so
            # re-appearing after being out of frame doesn't register as one
            # huge instantaneous jump in speed.
            velocity_mps = 0.0
            if detected:
                distance_cm = fraction_to_pseudo_distance_cm(distance_fraction)
                if prev_distance_cm is not None and prev_time is not None:
                    dt = loop_start - prev_time
                    if dt > 0:
                        velocity_mps = (prev_distance_cm - distance_cm) / 100.0 / dt
                prev_distance_cm = distance_cm
                prev_time = loop_start
            else:
                prev_distance_cm = None
                prev_time = None

            real_freq = compute_real_doppler_freq(FAR_FREQ_HZ, velocity_mps)
            real_shift_hz = real_freq - FAR_FREQ_HZ
            demo_freq = FAR_FREQ_HZ + (step / float(NUM_TONE_STEPS - 1)) * (NEAR_FREQ_HZ - FAR_FREQ_HZ)

            # One reading per beep - printed every frame, no throttling/averaging
            bar = make_distance_bar(distance_fraction)
            color = get_color_for_step(step, detected)
            print("{}{}  detected={}  step={:2d}/{}  |  demo tone={:.0f} Hz  real shift={:+.5f} Hz{}".format(
                color, bar, detected, step, NUM_TONE_STEPS - 1, demo_freq, real_shift_hz, COLOR_RESET))

            session_timestamps.append(loop_start)
            session_areas.append(smoothed_area)
            session_intervals.append(interval)
            session_demo_freqs.append(demo_freq)
            session_real_shifts.append(real_shift_hz)
            session_distance_fractions.append(distance_fraction)

            led.pulse(fade_in_time=interval * 0.4, fade_out_time=interval * 0.4,
                      n=1, background=True)
            play_tone_file(tone_files[step])  # blocking - keeps timing accurate

            elapsed = time.time() - loop_start
            remaining = max(0, (interval * 2) - elapsed)
            time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    finally:
        led.off()
        cap.release()
        summary_lines = print_session_summary(session_timestamps, session_areas, session_intervals,
                                               session_distance_fractions, min_area, max_area)
        export_session_csv(session_timestamps, session_areas, session_distance_fractions,
                            session_intervals, session_demo_freqs, session_real_shifts)
        generate_session_graph(session_timestamps, session_areas, session_intervals)
        physics_lines = print_physics_explanation(session_real_shifts)
        if summary_lines and physics_lines:
            save_session_report(summary_lines, physics_lines)


if __name__ == "__main__":
    run_demo()
