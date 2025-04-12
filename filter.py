import streamlit as st
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("Step 1 & 2: Clean and Noisy ECG (Interactive Version)")

# Interactive parameters
noise_std = st.slider("Noise Standard Deviation", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
duration = st.slider("Duration (seconds)", min_value=5, max_value=30, value=20, step=1)
fs = 200  # Hz

# Cached ECG simulation
@st.cache_data
def generate_clean_ecg(duration, fs):
    return nk.ecg_simulate(duration=duration, sampling_rate=fs, method="ecgsyn")

ecg_clean = generate_clean_ecg(duration, fs)
time = np.linspace(0, duration, len(ecg_clean))

# Add white Gaussian noise
noise = np.random.normal(loc=0, scale=noise_std, size=len(ecg_clean))
ecg_noisy = ecg_clean + noise

# Display stats
st.write("### Signal Statistics")
st.write(f"Clean ECG Mean: `{np.mean(ecg_clean):.4f}`")
st.write(f"Noisy ECG Standard Deviation: `{np.std(ecg_noisy):.4f}`")

# Clean and noisy side-by-side
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

# Overlay
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

# Optional zoom-in toggle
if st.checkbox("Show Zoomed-in Segment (First 2 Seconds)"):
    idx_zoom = int(2 * fs)
    fig4, ax4 = plt.subplots(figsize=(12, 3))
    ax4.plot(time[:idx_zoom], ecg_noisy[:idx_zoom], label="Noisy ECG", color="red", alpha=0.6)
    ax4.plot(time[:idx_zoom], ecg_clean[:idx_zoom], label="Clean ECG", color="blue", linestyle="--")
    ax4.set_title("Zoomed View: First 2 Seconds")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Amplitude")
    ax4.grid(True)
    ax4.legend()
    st.pyplot(fig4)
