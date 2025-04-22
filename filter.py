import streamlit as st
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
import matplotlib.ticker as ticker # For frequency plot formatting

# --- Wiener Filter Implementation ---

@st.cache_data # Cache the results of the filter calculation
def apply_wiener_filter(noisy_signal, clean_signal, order, symmetric=False):
    """
    Applies an 'ideal' Wiener filter using known clean signal.

    Args:
        noisy_signal (np.ndarray): The input signal corrupted by noise (x[n]).
        clean_signal (np.ndarray): The desired clean signal (s[n]).
        order (int): The order parameter 'p' of the Wiener filter.
                     For causal, filter length = p.
                     For symmetric, filter length = 2*p + 1.
        symmetric (bool): If True, compute a non-causal symmetric filter.
                          If False (default), compute a causal filter.

    Returns:
        np.ndarray: The filtered signal, or None if filtering fails.
    """
    if order < 1:
        st.warning("Filter order (p) must be at least 1.")
        return None # Return None to indicate failure clearly

    N = len(noisy_signal)
    filter_length = order if not symmetric else 2 * order + 1

    if N <= filter_length:
         st.warning(f"Signal length ({N}) must be greater than filter length ({filter_length}). Adjust order or signal duration.")
         return None

    # --- Calculate Correlations (Biased estimates) ---
    # Autocorrelation of the noisy signal (for Rxx)
    phi_xx_full = np.correlate(noisy_signal, noisy_signal, mode='full')
    center_idx = N - 1 # Index corresponding to lag 0

    # Cross-correlation between noisy signal (x) and clean signal (s) (for Rxs)
    phi_xs_full = np.correlate(noisy_signal, clean_signal, mode='full')

    try:
        if not symmetric:
            # --- Causal Wiener Filter ---
            # Rxx: Uses lags 0 to p-1 for the first column of the p x p Toeplitz matrix
            autocorr_for_toeplitz = phi_xx_full[center_idx : center_idx + order] / N
            Rxx = toeplitz(autocorr_for_toeplitz)

            # Rxs: Uses lags 0 to p-1 for the p x 1 vector
            Rxs = phi_xs_full[center_idx : center_idx + order] / N

            filter_size = order

        else:
            # --- Symmetric (Non-Causal) Wiener Filter ---
            # Filter length L = 2p + 1
            L = filter_length
            # Rxx: Uses lags 0 to 2p (L-1) for the first column of the L x L Toeplitz matrix
            # Need lags 0, 1, ..., L-1
            autocorr_for_toeplitz = phi_xx_full[center_idx : center_idx + L] / N
            if len(autocorr_for_toeplitz) != L:
                 raise ValueError(f"Incorrect autocorrelation length for symmetric Rxx. Expected {L}, got {len(autocorr_for_toeplitz)}")
            Rxx = toeplitz(autocorr_for_toeplitz)

            # Rxs: Uses lags -p to +p for the L x 1 vector
            # Indices: center_idx - p to center_idx + p
            start_idx = center_idx - order
            end_idx = center_idx + order + 1
            if start_idx < 0 or end_idx > len(phi_xs_full):
                raise ValueError(f"Cannot extract required cross-correlation lags for symmetric filter. Check signal length and order.")
            Rxs = phi_xs_full[start_idx : end_idx] / N
            if len(Rxs) != L:
                 raise ValueError(f"Incorrect cross-correlation length for symmetric Rxs. Expected {L}, got {len(Rxs)}")

            filter_size = L

        # --- Solve for Filter Coefficients ---
        # Add regularization (small value to diagonal) to prevent singularity
        epsilon = 1e-7
        Rxx_reg = Rxx + epsilon * np.identity(filter_size)

        # Solve Rxx * h = Rxs
        h_wiener = np.linalg.solve(Rxx_reg, Rxs)

    except np.linalg.LinAlgError:
         st.error(f"Singular matrix encountered for Wiener filter (order={order}, symmetric={symmetric}). Cannot solve Wiener-Hopf equations. Try a different order or noise level.")
         return None # Indicate failure
    except ValueError as e:
         st.error(f"Error constructing Wiener filter components: {e}")
         return None

    # --- Apply the Filter ---
    # Convolve the *noisy* signal with the filter coefficients
    # 'same' mode keeps the output length equal to the input length and handles centering
    filtered_signal = np.convolve(noisy_signal, h_wiener, mode='same')

    # Store filter coefficients for potential display
    # st.session_state['wiener_coeffs'] = h_wiener

    return filtered_signal


# --- HRV Analysis Function (NO CHANGES HERE) ---
def perform_hrv_analysis(signal, fs, signal_name="Signal", show_intermediate=False):
    """Performs R-peak detection and HRV analysis, optionally showing intermediate steps."""
    st.write(f"--- HRV Analysis for {signal_name} ---")
    results = {"hrv_summary": None, "rpeaks": None, "rr_intervals": None, "error": None}
    fig_peaks = None
    fig_rr = None
    fig_poincare = None

    try:
        # 1. Find R-peaks
        peaks_info, _ = nk.ecg_peaks(signal, sampling_rate=fs, method='neurokit', correct_artifacts=True)
        rpeaks_indices = peaks_info['ECG_R_Peaks'] # Get the indices

        if rpeaks_indices is None or len(rpeaks_indices) < 3: # Need at least 3 peaks for basic HRV
            st.warning(f"Not enough R-peaks found in {signal_name} ({len(rpeaks_indices) if rpeaks_indices is not None else 0}) for full HRV analysis.")
            results["error"] = "Insufficient R-peaks"
            return results, fig_peaks, fig_rr, fig_poincare # Return early with None figures

        results["rpeaks"] = rpeaks_indices

        # Calculate RR intervals (in ms)
        rr_intervals = np.diff(rpeaks_indices) / fs * 1000
        results["rr_intervals"] = rr_intervals

        if show_intermediate:
             # Show R-peak indices
             st.write(f"R-peak indices ({len(rpeaks_indices)} found):")
             st.dataframe(rpeaks_indices) # Use dataframe for better scrolling

             # Plot R-peaks on signal
             fig_peaks, ax_peaks = plt.subplots(figsize=(12, 3))
             time_axis = np.arange(len(signal)) / fs
             ax_peaks.plot(time_axis, signal, label=f'{signal_name} ECG', alpha=0.8, linewidth=1.0)
             valid_rpeaks = rpeaks_indices[rpeaks_indices < len(signal)]
             ax_peaks.scatter(time_axis[valid_rpeaks], signal[valid_rpeaks], color='red', label='Detected R-peaks', zorder=5, s=30) # Increased size
             ax_peaks.set_title(f'R-Peak Detection on {signal_name}')
             ax_peaks.set_xlabel('Time (s)')
             ax_peaks.set_ylabel('Amplitude')
             ax_peaks.legend()
             ax_peaks.grid(True)

             # Show RR intervals (Tachogram)
             if len(rr_intervals) > 0:
                 fig_rr, ax_rr = plt.subplots(figsize=(10, 3))
                 ax_rr.plot(rr_intervals, marker='o', linestyle='-', markersize=4)
                 ax_rr.set_title(f'RR Intervals (Tachogram) for {signal_name}')
                 ax_rr.set_xlabel('Beat Number')
                 ax_rr.set_ylabel('RR Interval (ms)')
                 ax_rr.grid(True)
             else:
                 st.write("No RR intervals to plot.")

        # 2. Calculate HRV indices
        try:
            hrv_summary = nk.hrv(rpeaks_indices, sampling_rate=fs, show=False) # show=False prevents nk plotting
            results["hrv_summary"] = hrv_summary
            st.write("HRV Summary Metrics:")
            st.dataframe(hrv_summary.round(3)) # Round for display
        except Exception as e_hrv:
            st.warning(f"Could not compute HRV metrics for {signal_name}: {e_hrv}")
            results["error"] = f"HRV computation failed: {e_hrv}"


        # Optional: Show Poincaré Plot (non-linear)
        if show_intermediate and len(rpeaks_indices) > 5:
            try:
                if len(rr_intervals) > 1:
                    rr_n = rr_intervals[:-1]
                    rr_n_plus_1 = rr_intervals[1:]

                    fig_poincare, ax_poincare = plt.subplots(figsize=(5, 5))
                    ax_poincare.scatter(rr_n, rr_n_plus_1, marker='o', alpha=0.7, s=15)
                    ax_poincare.set_xlabel('RR_n (ms)')
                    ax_poincare.set_ylabel('RR_n+1 (ms)')
                    ax_poincare.set_title(f'Poincaré Plot for {signal_name}')
                    ax_poincare.grid(True)
                    ax_poincare.set_aspect('equal', adjustable='box')
                else:
                     st.write("Not enough RR intervals for Poincaré plot.")

            except Exception as e_poincare:
                st.warning(f"Could not generate Poincaré plot for {signal_name}: {e_poincare}")
                results["error"] = results.get("error", "") + f" | Poincaré failed: {e_poincare}"

    except Exception as e:
        st.error(f"Critical Error during HRV analysis for {signal_name}: {e}")
        results["error"] = f"Critical analysis error: {e}"

    return results, fig_peaks, fig_rr, fig_poincare

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("ECG Denoising Demo: Wiener Filter")

# --- Simulation Parameters ---
st.sidebar.header("Simulation Parameters")
duration = st.sidebar.slider("Signal Duration (s)", min_value=10, max_value=30, value=20, step=5)
fs = 200
st.sidebar.write(f"Sampling Frequency: {fs} Hz")
noise_std = st.sidebar.slider("Noise Standard Deviation", min_value=0.0, max_value=0.4, value=0.2, step=0.05)

# --- Filter Parameters (Wiener Only) ---
st.sidebar.header("Wiener Filter Parameters") # Changed Header
wiener_order = st.sidebar.slider("Wiener Filter Order (p)", min_value=1, max_value=10, value=4, step=1)
wiener_symmetric = st.sidebar.checkbox("Use Symmetric (Non-Causal) Filter", value=False)
st.sidebar.info("Ideal Wiener filter uses the *clean* signal for optimal coefficient calculation.")
# Add Wiener explanation expander
with st.sidebar.expander("Wiener Filter Explanation (Conceptual)"):
    st.markdown("""
    *   **Goal:** Estimate clean signal `s[n]` from noisy `x[n]`.
    *   **Method:** Minimize Mean Squared Error `E[(s[n] - ŝ[n])²]`.
    *   **Solution:** Wiener-Hopf Equation `Rxx * h = Rxs`.
    *   **Ideal:** Uses known clean `s[n]` to find `Rxs`.
    *   **Causal:** Uses only past/present `x[n]`. Filter length `p`.
    *   **Symmetric:** Uses past, present, future `x[n]`. Filter length `2p+1`. Requires signal buffering in real-time.
    """)

# --- Generate Signals ---
# Cached ECG simulation
@st.cache_data
def generate_clean_ecg(duration, fs):
    # Using a fixed random state for reproducibility within the cached function
    try:
      # Specify heart rate variability if desired
      return nk.ecg_simulate(duration=duration, sampling_rate=fs, method="ecgsyn", heart_rate=70, heart_rate_std=5, random_state=42)
    except Exception as e:
      st.error(f"Error generating clean ECG: {e}. Try different parameters.")
      # Fallback to a simple sine wave if generation fails
      st.warning("Falling back to a simple sine wave.")
      t = np.linspace(0, duration, int(fs * duration), endpoint=False)
      return 0.5 * np.sin(2 * np.pi * 1 * t) # 1 Hz sine wave

ecg_clean = generate_clean_ecg(duration, fs)
time = np.linspace(0, duration, len(ecg_clean), endpoint=False)

# Add white Gaussian noise (regenerate noise if std changes)
# Use a fixed seed related to the noise_std for some consistency
np.random.seed(int(noise_std * 10000)) # Seed based on noise level
noise = np.random.normal(loc=0, scale=noise_std, size=len(ecg_clean))
ecg_noisy = ecg_clean + noise

# --- Apply Wiener Filter ---
st.header("Wiener Filter Results") # Changed Header
ecg_filtered = apply_wiener_filter(ecg_noisy, ecg_clean, wiener_order, wiener_symmetric)
filter_params_str = f"p={wiener_order}, symmetric={wiener_symmetric}"


# --- Display Stats ---
st.subheader("Signal Statistics & Filter Performance")
col_stats1, col_stats2, col_stats3 = st.columns(3)

with col_stats1:
    st.metric("Clean ECG Std Dev", f"{np.std(ecg_clean):.4f}")
with col_stats2:
    st.metric("Noisy ECG Std Dev", f"{np.std(ecg_noisy):.4f}")
with col_stats3:
    if ecg_filtered is not None:
        st.metric("Filtered ECG Std Dev", f"{np.std(ecg_filtered):.4f}")
    else:
        st.metric("Filtered ECG Std Dev", "N/A")

# Calculate MSE
col_mse1, col_mse2 = st.columns(2)
with col_mse1:
    mse_noisy = np.mean((ecg_noisy - ecg_clean)**2)
    st.write(f"**MSE (Noisy vs Clean): {mse_noisy:.6f}**")
with col_mse2:
    if ecg_filtered is not None:
        mse_filtered = np.mean((ecg_filtered - ecg_clean)**2)
        st.write(f"**MSE (Filtered vs Clean): {mse_filtered:.6f}**")
        if mse_noisy > 1e-12 and mse_filtered > 1e-12: # Avoid division by zero or near-zero
            improvement = mse_noisy / mse_filtered
            st.write(f"**MSE Improvement Factor: {improvement:.2f}x**")
        else:
            st.write("MSE Improvement Factor: N/A")
    else:
        st.write("**MSE (Filtered vs Clean): N/A (Filtering Failed)**")


# --- Plotting ---
st.subheader("Time Domain Signal Plots") # Renamed for clarity

# Plot 1: Overlay
fig_compare, ax_compare = plt.subplots(figsize=(14, 4))
ax_compare.plot(time, ecg_noisy, label=f"Noisy (MSE={mse_noisy:.4f})", color="red", alpha=0.6, linewidth=1.0)
if ecg_filtered is not None:
    ax_compare.plot(time, ecg_filtered, label=f"Wiener Filtered ({filter_params_str}, MSE={mse_filtered:.4f})", color="green", alpha=0.9, linewidth=1.2)
ax_compare.plot(time, ecg_clean, label="Clean ECG", color="blue", linestyle="--", alpha=0.8, linewidth=1.0)
ax_compare.set_xlabel("Time (s)")
ax_compare.set_ylabel("Amplitude")
ax_compare.set_title("Overlay: Clean, Noisy, and Wiener Filtered ECG")
ax_compare.grid(True)
ax_compare.legend()
ax_compare.set_xlim(0, duration)
st.pyplot(fig_compare)

# Optional zoom-in toggle
zoom_plot = st.checkbox("Show Zoomed-in Signal Plot", value=True)
if zoom_plot:
    max_zoom = min(5.0, duration) # Limit max zoom
    zoom_duration = st.slider("Zoom Duration (s)", min_value=0.5, max_value=max_zoom, value=min(3.0, max_zoom), step=0.25, key="signal_zoom_slider")
    idx_zoom = int(zoom_duration * fs)

    fig_zoom, ax_zoom = plt.subplots(figsize=(14, 4))
    ax_zoom.plot(time[:idx_zoom], ecg_noisy[:idx_zoom], label="Noisy ECG", color="red", alpha=0.6)
    if ecg_filtered is not None:
        ax_zoom.plot(time[:idx_zoom], ecg_filtered[:idx_zoom], label=f"Wiener Filtered ({filter_params_str})", color="green", alpha=0.9, linewidth=1.5)
    ax_zoom.plot(time[:idx_zoom], ecg_clean[:idx_zoom], label="Clean ECG", color="blue", linestyle="--", alpha=0.8)
    ax_zoom.set_title(f"Zoomed Signal View: First {zoom_duration:.2f} Seconds")
    ax_zoom.set_xlabel("Time (s)")
    ax_zoom.set_ylabel("Amplitude")
    ax_zoom.grid(True)
    ax_zoom.legend()
    st.pyplot(fig_zoom)

# --- NEW: Frequency Domain Plots ---
st.subheader("Frequency Domain Analysis")

if ecg_filtered is not None: # Only plot if filtering was successful
    N = len(ecg_clean)
    # Calculate FFTs
    fft_clean = np.fft.fft(ecg_clean)
    fft_noisy = np.fft.fft(ecg_noisy)
    fft_filtered = np.fft.fft(ecg_filtered)

    # Calculate frequency axis (only need positive frequencies)
    freq = np.fft.fftfreq(N, d=1/fs)
    positive_freq_idxs = np.where((freq >= 0) & (freq <= fs / 2))[0]
    freq_pos = freq[positive_freq_idxs]

    # Extract positive frequency components
    fft_clean_pos = fft_clean[positive_freq_idxs]
    fft_noisy_pos = fft_noisy[positive_freq_idxs]
    fft_filtered_pos = fft_filtered[positive_freq_idxs]

    # Calculate Magnitude (in dB for better visualization)
    epsilon = 1e-12 # To avoid log10(0)
    mag_clean_db = 20 * np.log10(np.abs(fft_clean_pos) + epsilon)
    mag_noisy_db = 20 * np.log10(np.abs(fft_noisy_pos) + epsilon)
    mag_filtered_db = 20 * np.log10(np.abs(fft_filtered_pos) + epsilon)

    # Calculate Phase (unwrapped and converted to degrees) # MODIFIED HERE
    phase_clean_deg = np.angle(fft_clean_pos, deg=True)
    phase_noisy_deg = np.angle(fft_noisy_pos, deg=True)
    phase_filtered_deg = np.angle(fft_filtered_pos, deg=True)

    # Plot Magnitude
    fig_mag, ax_mag = plt.subplots(figsize=(14, 4))
    ax_mag.plot(freq_pos, mag_clean_db, label='Clean', color='blue', alpha=0.7, linewidth=1.0)
    ax_mag.plot(freq_pos, mag_noisy_db, label='Noisy', color='red', alpha=0.6, linewidth=1.0)
    ax_mag.plot(freq_pos, mag_filtered_db, label='Wiener Filtered', color='green', alpha=0.9, linewidth=1.5)
    ax_mag.set_xlabel("Frequency (Hz)")
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_title("Frequency Magnitude Spectrum")
    ax_mag.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_mag.legend()
    ax_mag.set_xlim(0, fs / 2)
    # ax_mag.set_ylim(bottom=np.min(mag_filtered_db)-10) # Optional: Adjust y-limits dynamically
    st.pyplot(fig_mag)

    # Plot Phase # MODIFIED HERE
    fig_phase, ax_phase = plt.subplots(figsize=(14, 4))
    ax_phase.plot(freq_pos, phase_clean_deg, label='Clean', color='blue', alpha=0.7, linewidth=1.0)
    ax_phase.plot(freq_pos, phase_noisy_deg, label='Noisy', color='red', alpha=0.6, linewidth=1.0)
    ax_phase.plot(freq_pos, phase_filtered_deg, label='Wiener Filtered', color='green', alpha=0.9, linewidth=1.5)
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("Phase (degrees)") # Updated Label
    ax_phase.set_title("Frequency Phase Spectrum")
    ax_phase.grid(True)
    ax_phase.legend()
    ax_phase.set_xlim(0, fs / 2)
    # Use automatic tick locator for better label spacing with degrees
    ax_phase.yaxis.set_major_locator(ticker.AutoLocator())
    ax_phase.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g')) # Standard number format
    # You could force more ticks if AutoLocator isn't enough:
    # ax_phase.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10)) # Example: Aim for 10 ticks
    st.pyplot(fig_phase)

else:
    st.warning("Frequency domain plots cannot be generated because the Wiener filter application failed.")

st.sidebar.markdown("---")
st.sidebar.info("NeuroKit2 & Wiener Filter Demo")