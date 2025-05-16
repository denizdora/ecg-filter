import streamlit as st
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
import matplotlib.ticker as ticker # For frequency plot formatting

# Set a fixed random state for reproducibility of simulation and noise
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --- Wiener Filter Implementation ---

# Cache the results of the filter calculation based on noisy signal, order, and symmetric flag
# Note: This implementation requires the *clean* signal for optimal coefficient calculation (Ideal Wiener Filter)
@st.cache_data # Cache the results of the filter calculation
def apply_wiener_filter(noisy_signal: np.ndarray, clean_signal: np.ndarray, order: int, symmetric: bool = False):
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
        tuple: (filtered_signal, dict_of_internals)
               filtered_signal (np.ndarray): The filtered signal.
               dict_of_internals (dict): Contains filter coefficients (h),
                                         regularized auto-correlation matrix (Rxx),
                                         and cross-correlation vector (Rxs).
        Returns (None, None) if filtering fails.
    """
    if order < 1:
        st.warning("Filter order (p) must be at least 1.")
        return None, None # Return None to indicate failure clearly

    N = len(noisy_signal)
    filter_length = order if not symmetric else 2 * order + 1

    if N <= filter_length:
         st.warning(f"Signal length ({N}) must be greater than filter length ({filter_length}). Adjust order or signal duration.")
         return None, None

    # --- Calculate Correlations (Biased estimates: divide by N) ---
    # Rxx: Autocorrelation of the noisy signal (x[n])
    # Required lags for causal: 0 to p-1
    # Required lags for symmetric: 0 to 2p (filter length - 1)
    phi_xx_full = np.correlate(noisy_signal, noisy_signal, mode='full')
    center_idx = N - 1 # Index corresponding to lag 0

    # Rxs: Cross-correlation between noisy signal (x[n]) and clean signal (s[n])
    # Required lags for causal: 0 to p-1
    # Required lags for symmetric: -p to +p
    phi_xs_full = np.correlate(noisy_signal, clean_signal, mode='full')

    try:
        if not symmetric:
            # --- Causal Wiener Filter ---
            # First column of Rxx (Toeplitz matrix) needs lags 0 to p-1
            autocorr_for_toeplitz = phi_xx_full[center_idx : center_idx + order] / N
            if len(autocorr_for_toeplitz) != order:
                 raise ValueError(f"Incorrect autocorrelation length for causal Rxx. Expected {order}, got {len(autocorr_for_toeplitz)}")
            Rxx = toeplitz(autocorr_for_toeplitz)

            # Rxs vector needs lags 0 to p-1
            Rxs = phi_xs_full[center_idx : center_idx + order] / N
            if len(Rxs) != order:
                 raise ValueError(f"Incorrect cross-correlation length for causal Rxs. Expected {order}, got {len(Rxs)}")

            filter_size = order

        else:
            # --- Symmetric (Non-Causal) Wiener Filter ---
            # Filter length L = 2p + 1
            L = filter_length
            # First column of Rxx (Toeplitz matrix) needs lags 0 to 2p (L-1)
            autocorr_for_toeplitz = phi_xx_full[center_idx : center_idx + L] / N
            if len(autocorr_for_toeplitz) != L:
                 raise ValueError(f"Incorrect autocorrelation length for symmetric Rxx. Expected {L}, got {len(autocorr_for_toeplitz)}")
            Rxx = toeplitz(autocorr_for_toeplitz)

            # Rxs vector needs lags -p to +p
            # Indices: center_idx - p to center_idx + p
            start_idx = center_idx - order
            end_idx = center_idx + order + 1
            if start_idx < 0 or end_idx > len(phi_xs_full):
                # This happens if signal is too short for the requested symmetric lag range
                raise ValueError(f"Cannot extract required cross-correlation lags (-{order} to +{order}) for symmetric filter. Signal length may be too short for this order.")
            Rxs = phi_xs_full[start_idx : end_idx] / N
            if len(Rxs) != L:
                 raise ValueError(f"Incorrect cross-correlation length for symmetric Rxs. Expected {L}, got {len(Rxs)}")

            filter_size = L

        # --- Solve for Filter Coefficients ---
        # Add regularization (small value to diagonal) to prevent singularity in Rxx
        # This is often necessary with real-world data or numerical precision issues
        epsilon = 1e-6 # Slightly larger epsilon can sometimes help stability
        Rxx_reg = Rxx + epsilon * np.identity(filter_size)

        # Solve the Wiener-Hopf equation: Rxx * h = Rxs
        h_wiener = np.linalg.solve(Rxx_reg, Rxs)

    except np.linalg.LinAlgError:
         st.error(f"Singular matrix encountered for Wiener filter (order={order}, symmetric={symmetric}). Cannot solve Wiener-Hopf equations. Try a different order, signal duration, or noise level.")
         return None, None # Indicate failure
    except ValueError as e:
         st.error(f"Error constructing Wiener filter components: {e}")
         return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during Wiener filter calculation: {e}")
        return None, None


    # --- Apply the Filter ---
    # Convolve the *noisy* signal with the filter coefficients
    # 'same' mode keeps the output length equal to the input length and handles centering
    # For the symmetric filter, 'same' mode implies the output corresponds to the
    # input sequence, appropriately shifted for the non-causal taps.
    filtered_signal = np.convolve(noisy_signal, h_wiener, mode='same')

    # Return filtered signal and internal details
    return filtered_signal, dict(
        h=h_wiener,        # impulse response
        Rxx=Rxx_reg,       # regularised autocorr matrix
        Rxs=Rxs,           # cross‑corr vector
        symmetric=symmetric, # Store type for display
        order=order # Store order for display
    )


# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="ECG Wiener Filter Demo", page_icon="📈")

st.title("ECG Denoising Demo: Wiener Filter")

st.markdown("""
This demo illustrates the concept of an **ideal Wiener filter** for ECG denoising.
An ideal Wiener filter requires access to the *clean* signal to compute the optimal filter coefficients.
In this simulation, we generate a synthetic clean ECG, add noise, and then apply the Wiener filter using the clean signal information to achieve maximum possible noise reduction in the Mean Squared Error sense.
""")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Parameters")
# Using unique keys for sliders for better stability in Streamlit
duration = st.sidebar.slider("Signal Duration (s)", min_value=10, max_value=30, value=20, step=5, key="sim_duration")
fs = 200
st.sidebar.write(f"Sampling Frequency: **{fs} Hz**")
noise_std = st.sidebar.slider("Noise Standard Deviation", min_value=0.0, max_value=0.4, value=0.2, step=0.05, key="noise_std")

st.sidebar.markdown("---") # Separator

st.sidebar.header("Wiener Filter Parameters")
wiener_order = st.sidebar.slider("Wiener Filter Order (p)", min_value=1, max_value=20, value=4, step=1, help="The number of filter taps (p for causal, 2p+1 for symmetric). Higher order means more complex filter, potentially better fit but risk of overfitting/instability.", key="wiener_order")
wiener_symmetric = st.sidebar.checkbox("Use Symmetric (Non-Causal) Filter", value=False, help="A symmetric filter uses past, present, and future samples (2p+1 taps). This is non-causal and requires buffering in real-time applications, but can provide better performance.", key="wiener_symmetric")

st.sidebar.markdown("---") # Separator

detailed = st.sidebar.checkbox("Show Detailed Explanations & Internals", value=False, key="show_detailed")

st.sidebar.markdown("---")
st.sidebar.info("Built with NeuroKit2 & Streamlit")

# --- Generate Signals ---
# Cached ECG simulation - Uses the fixed RANDOM_STATE
@st.cache_data
def generate_clean_ecg(duration, fs):
    try:
      # Specify heart rate variability for a more realistic signal
      return nk.ecg_simulate(duration=duration, sampling_rate=fs, method="ecgsyn", heart_rate=70, heart_rate_std=5, random_state=RANDOM_STATE)
    except Exception as e:
      st.error(f"Error generating clean ECG: {e}. Try different parameters.")
      # Fallback to a simple sine wave if generation fails
      st.warning("Falling back to a simple sine wave.")
      t = np.linspace(0, duration, int(fs * duration), endpoint=False)
      return 0.5 * np.sin(2 * np.pi * 1 * t) # 1 Hz sine wave (very basic)

ecg_clean = generate_clean_ecg(duration, fs)
time = np.linspace(0, duration, len(ecg_clean), endpoint=False)

# Add white Gaussian noise - Uses the fixed RANDOM_STATE
# We need to regenerate the noise whenever noise_std changes, even if ECG is cached
np.random.seed(RANDOM_STATE) # Reset seed to ensure same noise structure *given the parameters*
noise = np.random.normal(loc=0, scale=noise_std, size=len(ecg_clean))
ecg_noisy = ecg_clean + noise

# --- Apply Wiener Filter ---
st.header("Wiener Filter Results")

# Apply the filter and get results + internals
ecg_filtered, wiener_info = apply_wiener_filter(ecg_noisy, ecg_clean, wiener_order, wiener_symmetric)
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
mse_noisy = np.mean((ecg_noisy - ecg_clean)**2)
col_mse1.write(f"**MSE (Noisy vs Clean):** {mse_noisy:.6f}")

if ecg_filtered is not None:
    mse_filtered = np.mean((ecg_filtered - ecg_clean)**2)
    col_mse2.write(f"**MSE (Filtered vs Clean):** {mse_filtered:.6f}")
    # Avoid division by zero or near-zero MSE values
    if mse_noisy > 1e-9 and mse_filtered > 1e-9:
        improvement = mse_noisy / mse_filtered
        col_mse2.write(f"**MSE Improvement Factor:** {improvement:.2f}x")
    else:
         col_mse2.write(f"**MSE Improvement Factor:** N/A (MSE values too small)")
else:
    col_mse2.write("**MSE (Filtered vs Clean):** N/A (Filtering Failed)")
    col_mse2.write("**MSE Improvement Factor:** N/A")

# ---------- DETAILED EXPLANATION & INTERNALS SECTION ---------------------------------
if detailed:
    st.markdown("---") # Separator
    st.markdown("## 🔬 Detailed Explanation & Internals")

    if wiener_info is not None:
        st.markdown("""
        The ideal Wiener filter coefficients $h[n]$ are calculated by solving the **Wiener-Hopf Equation**:
        $$ \\mathbf R_{xx} \\mathbf h = \\mathbf r_{xs} $$
        where:
        *   $\\mathbf R_{xx}$ is the autocorrelation matrix of the noisy input signal $x[n]$.
        *   $\\mathbf r_{xs}$ is the cross-correlation vector between the noisy input signal $x[n]$ and the desired clean signal $s[n]$.
        *   $\\mathbf h$ is the vector of filter coefficients we solve for.

        The filter order $p$ (or $2p+1$ for symmetric) determines the size of the matrices/vectors ($p$ times $p$ or $(2p+1)$ times $(2p+1)$).
        A small regularization value ($\\epsilon = 10^{-6}$) is added to the diagonal of $\\mathbf R_{xx}$ to improve numerical stability and prevent singularity.
        """)

        col_detailed1, col_detailed2 = st.columns(2)

        # -- Impulse response ---------------------------------
        with col_detailed1:
            st.markdown("#### Wiener filter impulse response $h[n]$")
            st.markdown(f"Filter Length: {len(wiener_info['h'])} taps")

            fig_h, ax_h = plt.subplots(figsize=(6, 3))
            ax_h.stem(np.arange(len(wiener_info['h'])), wiener_info["h"]) # Use np.arange for tap index
            ax_h.set_xlabel("Tap index")
            ax_h.set_ylabel("Amplitude")
            ax_h.set_title("Wiener Filter Impulse Response")
            ax_h.grid(True, alpha=0.4)
            st.pyplot(fig_h)

        # -- Correlation matrices / vectors -----------------------------
        with col_detailed2:
            with st.expander("View Correlation Matrices / Vectors"):
                st.write("$\\mathbf R_{xx}$: Autocorrelation matrix of $x[n]$ (first 10×10 block shown)")
                st.write("Shape:", wiener_info["Rxx"].shape)
                st.dataframe(wiener_info["Rxx"][:10, :10].round(4)) # Show first 10x10 block

                st.write("$\\mathbf r_{xs}$: Cross-correlation vector between $x[n]$ and $s[n]$")
                st.write("Shape:", wiener_info["Rxs"].shape)
                # Display Rxs as a column vector for clarity if not too long, otherwise show head
                if len(wiener_info["Rxs"]) <= 20: # Display full vector if short
                     st.dataframe(wiener_info["Rxs"].reshape(-1, 1).round(4))
                else: # Display head and tail if long
                     st.dataframe(np.vstack([wiener_info["Rxs"][:10], wiener_info["Rxs"][-10:]]).round(4),
                                   column_names=["value"],
                                   row_index=["0", "1", ..., "9", "...", f"{len(wiener_info['Rxs'])-10}", ..., f"{len(wiener_info['Rxs'])-1}"])


    else:
        st.warning("Detailed internals are not available because the Wiener filter could not be computed.")

# --- Plotting ---
st.markdown("---") # Separator
st.subheader("Time Domain Signal Plots")

# Plot 1: Overlay
fig_compare, ax_compare = plt.subplots(figsize=(14, 4))
ax_compare.plot(time, ecg_noisy, label=f"Noisy (MSE={mse_noisy:.4f})", color="red", alpha=0.7, linewidth=1.0)
if ecg_filtered is not None:
    mse_filtered = np.mean((ecg_filtered - ecg_clean)**2) # Recalculate MSE just for plot label
    ax_compare.plot(time, ecg_filtered, label=f"Wiener Filtered ({filter_params_str}, MSE={mse_filtered:.4f})", color="green", alpha=0.9, linewidth=1.5)
else:
     # Plot a placeholder if filtering failed
     ax_compare.plot([], [], label="Wiener Filtered (Failed)", color="green") # Empty plot entry
ax_compare.plot(time, ecg_clean, label="Clean ECG", color="blue", linestyle="--", alpha=0.8, linewidth=1.0)
ax_compare.set_xlabel("Time (s)")
ax_compare.set_ylabel("Amplitude")
ax_compare.set_title("Overlay: Clean, Noisy, and Wiener Filtered ECG")
ax_compare.grid(True, which='both', linestyle='--', linewidth=0.5)
ax_compare.legend()
ax_compare.set_xlim(0, duration)
st.pyplot(fig_compare)

# Optional zoom-in toggle
zoom_plot = st.checkbox("Show Zoomed-in Signal Plot", value=True, key="show_zoom_plot")
if zoom_plot:
    max_zoom = min(duration, 10.0) # Limit max zoom duration to a reasonable value
    zoom_duration = st.slider("Zoom Duration (s)", min_value=0.5, max_value=max_zoom, value=min(3.0, max_zoom), step=0.25, key="signal_zoom_slider")
    idx_zoom = int(zoom_duration * fs)

    fig_zoom, ax_zoom = plt.subplots(figsize=(14, 4))
    ax_zoom.plot(time[:idx_zoom], ecg_noisy[:idx_zoom], label="Noisy ECG", color="red", alpha=0.7)
    if ecg_filtered is not None:
        ax_zoom.plot(time[:idx_zoom], ecg_filtered[:idx_zoom], label=f"Wiener Filtered ({filter_params_str})", color="green", alpha=0.9, linewidth=1.5)
    else:
        ax_zoom.plot([], [], label="Wiener Filtered (Failed)", color="green") # Empty plot entry
    ax_zoom.plot(time[:idx_zoom], ecg_clean[:idx_zoom], label="Clean ECG", color="blue", linestyle="--", alpha=0.8)
    ax_zoom.set_title(f"Zoomed Signal View: First {zoom_duration:.2f} Seconds")
    ax_zoom.set_xlabel("Time (s)")
    ax_zoom.set_ylabel("Amplitude")
    ax_zoom.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_zoom.legend()
    ax_zoom.set_xlim(0, zoom_duration) # Ensure x-axis matches slider
    st.pyplot(fig_zoom)
