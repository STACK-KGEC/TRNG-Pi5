# 🌊 Raspberry Pi True Random Number Generator using Light and Water

A **True Random Number Generator (TRNG)** built using a **Raspberry Pi 5** and a **BH1750 light intensity sensor**, capturing the chaotic behavior of **bubbling water**. This project collects **unpredictable, real-world data** from the environment to generate high-entropy randomness, usable for simulations, cryptographic experiments, or chaotic modeling.

---

## 📸 Project Idea

Most random number generators (like NumPy's or Python's `random`) are **pseudo-random** — deterministic with a seed. This project instead uses **natural, physical processes** to collect real randomness:

> 🔦 A light sensor (BH1750) is placed above a vessel of **bubbling water** (e.g., boiling or air-pumped), and the light intensity is read multiple times over time. The randomness comes from the unpredictable movement and refraction caused by bubbles, splashes, and flickering.

---

## 🧪 Why Is This Truly Random?

This is a **hardware entropy source**:

- **Bubbling water** creates random, chaotic changes in how light reflects and refracts.
- The BH1750 captures rapid **analog light fluctuations** caused by these changes.
- The setup is sensitive to:
  - Bubble size and burst timing
  - Surface ripples
  - Ambient lighting shifts
  - Steam or vibration
  
Since all of these are governed by **thermodynamic noise**, they are **unpredictable and non-deterministic**, satisfying the criteria for **true randomness**.

---

## 🧠 Hardware & Software Stack

### 🔌 Hardware
- Raspberry Pi 5
- BH1750 Light Intensity Sensor (connected via I2C)
- Water container + bubbling source (e.g., pump or heat)
- Optional: paper shade or LED to enhance lighting contrast

### 🖥️ Software
- Python 3
- `smbus2`, `socket`, `numpy`, `pickle`, `struct`
- PC/Client to request and receive random values over LAN

---

## 🔄 How It Works

### 📍 Sensor Side (Raspberry Pi)

The Pi runs a Python server that:
1. Initializes the BH1750 sensor over I2C
2. Waits for incoming TCP requests from a client
3. Collects `N` light readings based on the request
4. Sends the readings (as a NumPy array) using `pickle` over the network

### 📱 Client Side (Laptop/PC)

The client connects via socket to the Pi and requests `N` values.

It:
1. Sends an integer (number of samples needed)
2. Receives and unpacks the light readings
3. **Normalizes** the readings to `[0, 1]` to serve as usable random floats

---

## 🧠 Algorithm (Client)

```python
# 1. Normalize readings
normed = (readings - readings.min()) / (readings.max() - readings.min() + 1e-9)

# 2. Use directly for rand(), or convert using scipy for randn()
from scipy.stats import norm
rand = normed
randn = norm.ppf(np.clip(normed, 1e-6, 1 - 1e-6))
```

- `rand()` → Uniform random numbers in [0, 1)
- `randn()` → Gaussian (normal) random numbers using inverse CDF
- `randint(a, b)` → Integer randoms via scaling

---

## 🧪 Sample Output

| Function | Example Output                  |
|----------|----------------------------------|
| `rand(5)`   | `[0.45, 0.78, 0.23, 0.65, 0.91]` |
| `randint(0, 10, size=4)` | `[2, 9, 3, 7]`     |
| `randn(3)`  | `[-0.34, 0.83, -1.75]`         |

These values are entirely based on **live ambient randomness**, not any software PRNG.

---

## 🧪 Testing Randomness

We verified the randomness using:
- **Histograms** (to check uniform or Gaussian distribution)
- **Autocorrelation** (should be low for true random)
- **Boxplots & Entropy** (distribution spread and unpredictability)

Example plots:

will be shown in future after more experimentations

---

## 📦 LightRandom Class (Client-Side)

The following Python class fetches and returns true random numbers:

```python
class LightRandom:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def _request_data(self, length):
        # Connect to Pi and fetch `length` readings
        # Normalize and return as array
        ...

    def rand(self, *shape):
        ...

    def randint(self, low, high=None, size=None):
        ...

    def randn(self, *shape):
        ...

    def shuffle(self, arr):
        ...

    def choice(self, a, size=None):
        ...
```

> 📌 Every function triggers a **real-time data request** to the Pi and returns fresh randomness.

---

## 📂 Project Structure

```
.
├── collected_data        # Generated True Random Numbers
├── first_steup           # Setting Up The Raspberry Pi 5
├── lib                   # Main Python Script For Requesting And Receiving Random Numbers With Post Processing
├── test_notebooks        # To Test And Visualise The Generated True Random Numbers With Post Analysis And Further Implementation
├── Pi_scripts            # The Functional Python Script On RAspberry Pi 5 (Server) To Collect Data Over Bubbling Water
└── README.md             # This file
```

---

## 🚀 Future Improvements

- Add whitening functions (logistic maps, XOR-fold, entropy amplification)
- Save readings to a database for analysis
- Package into a Python module (`pip install lightrandom`)
- Stream entropy to multiple clients

---

## 🙌 Acknowledgments

- Inspired by nature's chaos, light, and physics!
- Thanks to the creators of the BH1750 sensor and Raspberry Pi ecosystem.
- Special credit to all open-source projects that power scientific creativity 🌍

---

## © Copyright

This work is the intellectual property of the *Nirjhar Debnath* and intended for research publication. Redistribution or reproduction in any form without permission is prohibited.

---

## 🤖 Made with Real Randomness

This README was written while coding under an LED light — which also powered some of the randomness you're reading from ✨
