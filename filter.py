import streamlit as st
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("Step 1 & 2: Clean and Noisy ECG (20s, 200Hz)")

# Parameters
duration = 20  # seconds
fs = 200       # Hz

# Generate clean ECG
ecg_clean = nk.ecg_simulate(duration=duration, sampling_rate=fs, method="ecgsyn")
time = np.linspace(0, duration, len(ecg_clean))

# Add white Gaussian noise
noise_std = 0.2
noise = np.random.normal(loc=0, scale=noise_std, size=len(ecg_clean))
ecg_noisy = ecg_clean + noise

# ----------------------------------------------------
# Row layout: Clean and Noisy in side-by-side figures
# ----------------------------------------------------
st.subheader("Clean vs Noisy ECG (Side-by-Side View)")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(7, 3))
    ax1.plot(time, ecg_clean, color="blue")
    ax1.set_title("Clean ECG")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.plot(time, ecg_noisy, color="red", alpha=0.7)
    ax2.set_title("Noisy ECG")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True)
    st.pyplot(fig2)

# ----------------------------------------------------
# Combined view: overlay both signals in one plot
# ----------------------------------------------------
st.subheader("Overlay: Clean and Noisy ECG Together")

fig3, ax3 = plt.subplots(figsize=(14, 4))
ax3.plot(time, ecg_noisy, label="Noisy ECG", color="red", alpha=0.6)
ax3.plot(time, ecg_clean, label="Clean ECG", color="blue", linestyle="--")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Amplitude")
ax3.set_title("Overlay of Clean and Noisy ECG Signals")
ax3.grid(True)
ax3.legend()
st.pyplot(fig3)
