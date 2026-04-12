######################################################################
# Author: Rohan Dahale, Nick Conroy, Date: 24 March 2026
######################################################################

import os

# Environment settings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import argparse
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import h5py
import ehtim as eh
import pandas as pd
from scipy import interpolate
import scipy.ndimage as ndimage
from scipy.stats import truncnorm
from scipy.ndimage import label, center_of_mass
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import label, center_of_mass
from mpl_toolkits.axes_grid1 import make_axes_locatable
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ProcessPoolExecutor
import functools
import glob
from tqdm import tqdm


# Constants
RADPERUAS = 4.848136811094136e-12
GM_C3 = 20.46049 / 3600  # hours
M_PER_RAD = 41139576306.70914

# Plot style settings
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["font.family"] = "sans-serif"


def create_parser():
    p = argparse.ArgumentParser(description="Calculate and visualize Pattern Speed.")
    p.add_argument('-d', '--data', type=str, required=True, help='Path to UVFITS data file.')
    p.add_argument('-i', '--input', type=str, nargs='+', required=True, help='Glob pattern(s) or list of reconstruction HDF5 files.')
    p.add_argument('--truthmv', type=str, default=None, help='Path to truth HDF5 file (optional).')
    p.add_argument('-o', '--outpath', type=str, default='./patternspeed_output', help='Output path prefix (without extension).')
    p.add_argument('--tstart', type=float, default=None, help='Start time (in UT hours) for data')
    p.add_argument('--tstop', type=float, default=None, help='Stop time (in UT hours) for data')
    p.add_argument('-n', '--ncores', type=int, default=32, help='Number of cores')
    p.add_argument('--nsamples', type=int, default=1000, help='Number of MCMC samples (default: 1000, set to 0 to skip MCMC)')
    return p

# -----------------------------------------------------------------------------
# RingFitter Class
# -----------------------------------------------------------------------------
class RingFitter:
    """Class for fitting and extracting ring parameters from EHT images."""
    
    def __init__(self, fov=200*eh.RADPERUAS, npix=200):
        self.fov = fov
        self.npix = npix
    
    def fit_from_image(self, image, **kwargs):
        """Extract ring parameters from an image."""
        # Extract parameters for finding center
        center_x = kwargs.pop('center_x', None)
        center_y = kwargs.pop('center_y', None)
        search_radius_min = kwargs.pop('search_radius_min', 10)
        search_radius_max = kwargs.pop('search_radius_max', 100)
        nrays_search = kwargs.pop('nrays_search', 25)
        nrs_search = kwargs.pop('nrs_search', 50)
        fov_search = kwargs.pop('fov_search', 0.1)
        n_search = kwargs.pop('n_search', 20)
        image_blur_fwhm = kwargs.pop('image_blur_fwhm', 2.0*eh.RADPERUAS)
        image_threshold = kwargs.pop('image_threshold', 0.05)
        
        # Find the center of the ring if not provided
        # Use a large global search area for finding the center so it doesn't fail on shifted images
        if center_x is None or center_y is None:
            center_x, center_y = self._find_center(
                image,
                search_radius_min=10,
                search_radius_max=100,
                nrays_search=nrays_search,
                nrs_search=nrs_search,
                fov_search=fov_search,
                n_search=n_search,
                blur_fwhm=image_blur_fwhm,
                threshold=image_threshold
            )
        
        # Extract all ring parameters with remaining kwargs
        params = self._extract_ring_parameters(image, center_x, center_y, min_radius=search_radius_min, max_radius=search_radius_max, **kwargs)
        
        # Calculate center coordinates relative to image center
        fov = kwargs.pop('fov', self.fov)
        x0 = center_x - fov/2/eh.RADPERUAS
        y0 = center_y - fov/2/eh.RADPERUAS
        
        result = {
            "x0": x0,
            "y0": y0,
            "r": params["D"]/2,
            "r_err": params["Derr"]/2,
            "xc_pix": center_x, # pixel coordinates
            "yc_pix": center_y
        }
        return result
    
    def _find_center(self, image, search_radius_min=10, search_radius_max=100,
                    nrays_search=25, nrs_search=50, fov_search=0.1, n_search=20,
                    blur_fwhm=2.0*eh.RADPERUAS, threshold=0.05):
        image_blur = image.blur_circ(blur_fwhm)
        image_mod = image_blur.threshold(cutoff=threshold)
        xc, yc = eh.features.rex.findCenter(
            image_mod, 
            rmin_search=search_radius_min, 
            rmax_search=search_radius_max,
            nrays_search=nrays_search, 
            nrs_search=nrs_search,
            fov_search=fov_search, 
            n_search=n_search
        )
        return xc, yc
    
    def _extract_ring_parameters(self, image, center_x, center_y, 
                                min_radius=5, max_radius=50, 
                                n_angles=360, n_radial=100,
                                return_full_data=False):
        # Setup radial and angular grid
        angles = np.linspace(0, 360, n_angles)
        angles_rad = np.deg2rad(angles)
        radial_points = np.linspace(0, max_radius, n_radial)
        dr = radial_points[1] - radial_points[0]
        
        # Create interpolation function for image using RectBivariateSpline
        x = np.arange(image.xdim) * image.psize / eh.RADPERUAS
        y = np.arange(image.ydim) * image.psize / eh.RADPERUAS
        z = image.imarr()
        image_interp = interpolate.RectBivariateSpline(y, x, z)
        
        # Sample image in polar coordinates
        radial_image = np.zeros([n_radial, n_angles])
        r_mesh, angle_mesh = np.meshgrid(radial_points, angles_rad)
        x_mesh = r_mesh * np.sin(angle_mesh) + center_x
        y_mesh = r_mesh * np.cos(angle_mesh) + center_y
        
        for r in range(n_radial):
            # Use the vectorized evaluation method of RectBivariateSpline
            vals = [image_interp(y_mesh[i][r], x_mesh[i][r], grid=False) for i in range(len(angles))]
            radial_image[r, :] = np.array(vals)
        radial_image = np.fliplr(radial_image)
        
        # Calculate ring parameters at each angle
        r_peak_values = []
        
        # Use far radius as intensity floor reference
        far_radius_idx = np.argmin(np.abs(radial_points - max_radius))
        intensity_floor = radial_image[far_radius_idx, :].mean()
        
        # Calculate peak radius and width at each angle
        for angle_idx in range(len(angles)):
            intensity_profile = radial_image[:, angle_idx] - intensity_floor
            intensity_profile[np.where(radial_points < min_radius)] = 0
            
            peak_idx = np.argmax(intensity_profile)
            r_peak = radial_points[peak_idx]
            
            # Refine peak location with quadratic interpolation
            if peak_idx > 0 and peak_idx < n_radial - 1:
                nearby_values = intensity_profile[peak_idx-1:peak_idx+2]
                r_peak = self._quad_interp_radius(r_peak, dr, nearby_values)[0]
            
            r_peak_values.append(r_peak)
        
        # Calculate average ring diameter and width
        ring_diameter = np.mean(r_peak_values) * 2
        diameter_error = np.std(r_peak_values) * 2
        
        result = {
            "D": ring_diameter,
            "Derr": diameter_error
        }
        return result
    
    def _quad_interp_radius(self, r_max, dr, val_list):
        v_L, v_max, v_R = val_list
        denom = (2 * (v_L + v_R - 2*v_max))
        if denom == 0:
            return r_max, v_max
        rpk = r_max + dr*(v_L - v_R) / denom
        vpk = 8*v_max*(v_L + v_R) - (v_L - v_R)**2 - 16*v_max**2
        vpk /= (8*(v_L + v_R - 2*v_max))
        return (rpk, vpk)

# -----------------------------------------------------------------------------
# Analysis Functions
# -----------------------------------------------------------------------------
def prepare_movie_data(im_list, times, fov, npix):
    # Dimensions
    N = npix
    dx = fov / N / RADPERUAS # pixel size
    dt = np.diff(times).mean() / GM_C3
    
    # Stack images
    I_stack = np.array([im.imarr() for im in im_list]) # (time, y, x)
    
    # Transpose from (time, y, x) to (y, x, time)
    Iall = np.transpose(I_stack, (1, 2, 0)) # (y, x, time)
    
    # Smoothing
    sigma = 20 / (dx * (2*np.sqrt(2*np.log(2)))) # 20 uas FWHM
    sIall = ndimage.gaussian_filter(Iall, sigma=(sigma, sigma, 0))
    
    return sIall, Iall, dt, dx

def map_coordinates_vectorized(data, coords):
    """
    Vectorized wrapper for scipy.ndimage.map_coordinates
    data: (ny, nx, nt) or (ny, nx)
    coords: (3, N) -> [y, x, t] or (2, N) -> [y, x]
    """
    return ndimage.map_coordinates(data, coords, order=1, mode='nearest')

def sample_cylinder(sIall, ring_params, dx, x_shift=0, y_shift=0, r_shift=0):
    # Dimensions from data
    # sIall is (y, x, time)
    ny, nx, nt = sIall.shape
    
    # Ring params in pixels (base)
    xc = ring_params['xc_pix'] # center in pixel coordinates
    yc = ring_params['yc_pix']
    
    # Base physical coords
    x0 = ring_params['x0']
    y0 = ring_params['y0']
    r_val = ring_params['r']
    
    # Apply shifts
    x0_new = x0 + x_shift
    y0_new = y0 + y_shift
    r_new = r_val + r_shift
    
    # Convert back to pixels
    center_index = nx / 2.0 
    
    xc_new_pix = (x0_new) / dx + center_index
    yc_new_pix = (y0_new) / dx + center_index
    r_new_pix = r_new / dx
    
    # Circle sampling points
    ntheta = 180
    theta = np.linspace(-np.pi, np.pi, ntheta, endpoint=False)
    pa = theta + 0.5*np.pi 
    
    # Calculate cylinder sampling coordinates
    # icirc (y), jcirc (x)
    icirc = yc_new_pix + r_new_pix*np.sin(pa)
    jcirc = xc_new_pix + r_new_pix*np.cos(pa)
    
    # Vectorized Sampling
    # We want to sample at (icirc, jcirc) for every time step t [0...nt-1]
    # Coordinates for map_coordinates: (ndim, npoints)
    # y coordinates: repeat icirc for each time step
    # x coordinates: repeat jcirc for each time step
    # t coordinates: repeat time index for each angle
    
    # Grid construction
    # y (ntheta) -> tile nt times -> (nt*ntheta)
    # x (ntheta) -> tile nt times -> (nt*ntheta)
    # t (nt)     -> repeat ntheta times -> (nt*ntheta)
    # We want result (nt, ntheta).
    # Order: for t in 0..nt: for angle in 0..ntheta
    # So t indices should be 0,0..0 (ntheta times), 1,1..1 (ntheta times)
    
    T_coords = np.repeat(np.arange(nt), ntheta)
    Y_coords = np.tile(icirc, nt)
    X_coords = np.tile(jcirc, nt)
    
    coords = np.vstack([Y_coords, X_coords, T_coords])
    
    # Order=1 (linear interpolation)
    # Mode='nearest' to match clamping behavior (or 'constant', cval=0)
    samples = ndimage.map_coordinates(sIall, coords, order=1, mode='nearest')
    
    qs = samples.reshape((nt, ntheta))
             
    return qs


def compute_autocorrelation(qs):
    """
    Compute centered autocorrelation function.
    Cyclic in PA (angle wrapping via FFT), non-cyclic in time (no time wrapping).
    Based on Nick Conroy's autocorr_unwrapped_fft method.
    
    Input:  qs shape (Nt, Nphi) -- time x angle
    Output: racf shape (2*Nt-1, Nphi) -- time-lag x angle-shift, peak normalized to 1
            qsn shape (Nt, Nphi) -- mean-subtracted cylinder
    """
    # Mean subtract
    qsn = np.copy(qs)
    qsn -= qsn.mean(axis=0, keepdims=True)  # subtract temporal mean at each angle
    qsn -= qsn.mean(axis=1, keepdims=True)  # subtract azimuthal mean at each time
    
    Nt, Nphi = qsn.shape
    R = np.zeros((2 * Nt - 1, Nphi), dtype=float)
    
    for j in range(2 * Nt - 1):
        tau_idx = j - (Nt - 1)  # time lag index
        
        # Get overlapping time slices (no time wrapping)
        if tau_idx >= 0:
            f1 = qsn[:Nt - tau_idx, :]   # (noverlap, Nphi)
            f2 = qsn[tau_idx:, :]         # (noverlap, Nphi)
        else:
            f1 = qsn[-tau_idx:, :]        # (noverlap, Nphi)
            f2 = qsn[:Nt + tau_idx, :]    # (noverlap, Nphi)
        
        noverlap = f1.shape[0]
        if noverlap == 0:
            continue
        
        # FFT along angle axis (axis=1) for cyclic PA wrapping
        FB = np.fft.fft(f1, axis=1)
        FC = np.fft.fft(f2, axis=1)
        cross = np.fft.ifft(FB * np.conj(FC), axis=1).real
        
        # Sum over overlapping time points -> (Nphi,)
        numerator = cross.sum(axis=0)
        
        # Map FFT ordering to direct-sum ordering
        numerator = np.roll(numerator[::-1], 1)
        
        # Per-lag normalization (prevents values > 1 at edges)
        sigma1 = np.sqrt(np.mean(f1**2))
        sigma2 = np.sqrt(np.mean(f2**2))
        if sigma1 == 0 or sigma2 == 0:
            continue
        norm = Nphi * noverlap * sigma1 * sigma2
        R[j, :] = numerator / norm
    
    # Center PA shifts so zero-shift is at the middle
    R = np.roll(R, shift=-Nphi // 2, axis=1)
    
    # Normalize peak to 1
    max_val = np.max(R)
    if max_val > 0:
        R /= max_val

    R = R[ (R.shape[0]//4) : (3*R.shape[0]//4) , : ]  ### NEW
    
    return R, qsn

def calculate_pattern_speed(racf, dt, dtheta=2.0, xi_crit_factor=3.0):

    racf_std = np.std(racf)
    xi_crit = xi_crit_factor * racf_std
    
    racf_cut = np.copy(racf)
    
    # Filter connected region
    labels_map, num_features = label((racf > xi_crit).astype(int))
    center_idx = (racf.shape[0]//2, racf.shape[1]//2)
    
    # Check initial region width
    non_zero_columns = 0
    non_zero_rows = 0
    if labels_map[center_idx] != 0:
         Q = labels_map == labels_map[center_idx]
         non_zero_columns = np.count_nonzero(np.sum(Q, axis=0))
         non_zero_rows = np.count_nonzero(np.sum(Q, axis=1))
    
    # Fallback: if center not in threshold or region too thin (< 5 pixels)
    if (labels_map[center_idx] == 0) or (non_zero_columns < 5) or (non_zero_rows < 5):
         target_width = 5
         col_offset = target_width // 2
         row_offset = target_width // 2
         
         # Determine edge column/row (ensure in bounds)
         edge_col = min(center_idx[1] + col_offset, racf.shape[1] - 1)
         edge_row = min(center_idx[0] + row_offset, racf.shape[0] - 1)
         
         # Set xi_crit to max value at edge column/row to ensure width >= target_width
         xi_crit_col = np.max(racf[:, int(edge_col)])
         xi_crit_row = np.max(racf[int(edge_row), :])
         xi_crit = min(xi_crit_col, xi_crit_row)
         
         # Restrict mask to target strip (mimicking cylinder.py fallback)
         col_indices = np.arange(racf.shape[1])
         row_indices = np.arange(racf.shape[0])
         strip_mask_col = np.abs(col_indices - center_idx[1]) <= col_offset
         strip_mask_row = np.abs(row_indices - center_idx[0]) <= row_offset
         
         threshold_mask = (racf >= xi_crit) & strip_mask_row[:, np.newaxis] & strip_mask_col[np.newaxis, :]
         mask_int = threshold_mask.astype(int)
         
         labels_map, num_features = label(mask_int)
         
         # Final safety check
         if labels_map[center_idx] == 0:
              return 0.0, np.zeros_like(racf), np.zeros_like(labels_map, dtype=bool)

         Q = labels_map == labels_map[center_idx]
    else:
         Q = labels_map == labels_map[center_idx]
    
    # Moments
    ts = np.linspace(-len(racf)/2, len(racf)/2, len(racf), endpoint=False)
    phis = np.linspace(-len(racf[0])/2, len(racf[0])/2, len(racf[0]), endpoint=False)
    
    delta_t = ts[1] - ts[0]
    delta_phi = phis[1] - phis[0]
    
    # Meshgrid
    T_mesh, Phi_mesh = np.meshgrid(ts, phis, indexing='ij')
    
    # Apply mask
    racf_cut[~Q] = 0.0
    
    # Weighted sums
    moment_t = np.sum(racf_cut * T_mesh**2)
    moment_t_phi = np.sum(racf_cut * T_mesh * Phi_mesh)
    
    if moment_t == 0:
        pattern_speed = 0
    else:
        pattern_speed = moment_t_phi / moment_t
        # Units
        pattern_speed = pattern_speed * dtheta / dt
        
    return pattern_speed, racf_cut, Q

def determine_xi_crit_factor(path):
    if 'truth' in path:
        if 'hs' in path:
            return 1.67
        else:
            return 2.25
    elif 'resolve' in path:
        return 1.40
    elif 'doghit' in path:
        return 0.30
    elif 'ehtim' in path:
        return 0.40
    elif 'kine' in path:
        return 0.30
    elif 'ngmem' in path:
        return 0.25
    else:
        return 0.6

def run_mcmc(sIall, ring_params, dx, dt, n_samples, xi_crit_factor_base, racf_best_std):
    # Setup MCMC
    # Sigma fit from xi_crit RMSE curves
    sigma = 0.7
    
    # Calculate base absolute threshold (mimicking cylinder.py logic)
    abs_base = xi_crit_factor_base * racf_best_std
    
    # xi_crit bounds (0, 1) in absolute units
    # Truncated normal for xi_crit
    a, b = (0 - abs_base) / sigma, (1 - abs_base) / sigma
    xi_crit_abs_samples = truncnorm.rvs(a, b, loc=abs_base, scale=sigma, size=n_samples)
    
    # Perturb Ring Parameters (x, y, r)
    # rerr is used for std of x, y, r perturbation
    rerr = ring_params.get('r_err', 1.0) # default if missing
    
    x_factor_samples = np.random.normal(0, rerr, n_samples)
    y_factor_samples = np.random.normal(0, rerr, n_samples)
    r_factor_samples = np.random.normal(0, rerr, n_samples)
    
    ps_samples = []
    
    for i in range(n_samples):
        # Sample cylinder with perturbed parameters
        qs = sample_cylinder(sIall, ring_params, dx, 
                           x_shift=x_factor_samples[i], 
                           y_shift=y_factor_samples[i], 
                           r_shift=r_factor_samples[i])
        
        # Compute Autocorr
        racf, _ = compute_autocorrelation(qs)
        
        # Calculate pattern speed
        # Convert absolute sample to factor for this specific realization
        current_std = np.std(racf)
        factor = xi_crit_abs_samples[i] / current_std
        
        ps, _, _ = calculate_pattern_speed(racf, dt, dtheta=2.0, xi_crit_factor=factor)
        ps_samples.append(ps)
        
    ps_samples = np.array(ps_samples)
    
    # Statistics
    mean_ps = np.mean(ps_samples)
    std_ps = np.std(ps_samples)
    percentiles = np.percentile(ps_samples, [15.865, 50, 84.135]) 
    median = percentiles[1]
    
    # Calculate uncertainties relative to the median (1 sigma)
    median_plus = percentiles[2] - median
    median_minus = median - percentiles[0]
    
    # Mode
    counts, bin_edges = np.histogram(ps_samples, bins=50)
    modal_bin_index = np.argmax(counts)
    mode_value = 0.5 * (bin_edges[modal_bin_index] + bin_edges[modal_bin_index + 1])
    
    return {
        'samples': ps_samples,
        'mean': mean_ps,
        'std': std_ps,
        'median': median,
        'median_plus_sigma': median_plus,
        'median_minus_sigma': median_minus,
        'mode': mode_value
    }

# -----------------------------------------------------------------------------
# Main Processing
# -----------------------------------------------------------------------------
def process_movie(path, times, fov, npix, n_samples=0):
    # Load movie
    mv = eh.movie.load_hdf5(path)
    
    # Uniform time sampling
    min_t = times.min()
    max_t = times.max()
    
    # Time stamps every two minutes
    step_hr = 2*1.0/60.0
    uniform_times = np.arange(min_t, max_t + 1e-5, step_hr)
    ntimes = len(uniform_times) 
    
    # Create list of regridded images
    im_list = []
    
    mv.reset_interp(bounds_error=False)
    for t in uniform_times:
        im = mv.get_image(t)
        im = im.blur_circ(0).regrid_image(fov, npix)
        #This flip is a hack to make plots correct pattern speed sign
        im.imvec = np.flipud(im.imarr()).flatten() 
        im_list.append(im)
        
    # Use merged average frame for fitting (preserves metadata)
    # Create a new movie from the regridded images for averaging
    merged_mov_for_mean = eh.movie.merge_im_list(im_list)
    mean_im = merged_mov_for_mean.avg_frame()
    
    rf = RingFitter(fov=fov, npix=npix)
    ring_params = rf.fit_from_image(
        mean_im, 
        search_radius_min=10, search_radius_max=100, 
        nrays_search=25, nrs_search=50
    )
    
    # Prepare Data
    sIall, Iall, dt, dx = prepare_movie_data(im_list, uniform_times, fov, npix)
    
    # Main Computation (Best Bet)
    qs = sample_cylinder(sIall, ring_params, dx)
    qs_raw = sample_cylinder(Iall, ring_params, dx) # Compute raw cylinder from unsmoothed data
    
    racf, qs_norm = compute_autocorrelation(qs)
    
    # Determine xi_crit factor
    xi_crit_factor = determine_xi_crit_factor(path)
    
    ps, racf_cut, mask = calculate_pattern_speed(racf, dt, dtheta=2.0, xi_crit_factor=xi_crit_factor)
    
    # MCMC
    mcmc_res = None
    if n_samples > 0:
        print(f"Running MCMC with {n_samples} samples...")
        # Calculate std of the best-bet racf for MCMC baseline
        racf_best_std = np.std(racf)
        mcmc_res = run_mcmc(sIall, ring_params, dx, dt, n_samples, xi_crit_factor, racf_best_std=racf_best_std)
    
    return {
        'mean_im': mean_im,
        'ring_params': ring_params,
        'qs': qs,
        'qs_raw': qs_raw,
        'qs_norm': qs_norm,
        'racf': racf,
        'racf_cut': racf_cut,
        'mask': mask,
        'ps': ps,
        'dt': dt,
        'times_len': ntimes,
        'times_dt': dt,
        'mcmc': mcmc_res
    }

def add_colorbar(mappable, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return ax.figure.colorbar(mappable, cax=cax)

def add_colorbar(mappable, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return ax.figure.colorbar(mappable, cax=cax)

def add_sci_colorbar(mappable, ax):
    from matplotlib import ticker
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = ax.figure.colorbar(mappable, cax=cax)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    cb.formatter = formatter
    cb.update_ticks()
    return cb

def plot_results(output_dir, recon_res, truth_res=None):
    from matplotlib import gridspec
    import matplotlib.colors as mcolors

    rows = 2 if truth_res else 1
    cols = 5
    
    fig = plt.figure(figsize=(6*cols, 5*rows))
    width_ratios = [1]*cols
    gs = gridspec.GridSpec(rows, cols, width_ratios=width_ratios, wspace=0.4, hspace=0.3)
    
    datasets = []
    titles = []
    if truth_res:
         datasets.append(truth_res)
         titles.append("Truth")
    datasets.append(recon_res)
    titles.append("Recon")
    
    # Pre-calculate ranges for Pass/Fail Check
    intervals = {}
    
    # We first iterate to collect MCMC stats if they exist
    for i, res in enumerate(datasets):
         title_prefix = titles[i]
         if res.get('mcmc'):
             mcmc = res['mcmc']
             samples = mcmc['samples']
             best_ps = res['ps']
             
             # Calculate Best Bet Errors (RMS)
             samples_above = samples[samples > best_ps]
             samples_below = samples[samples < best_ps]
             
             best_plus = 0.0
             if len(samples_above) > 0:
                 best_plus = np.sqrt(np.mean((samples_above - best_ps)**2))
                 
             best_minus = 0.0
             if len(samples_below) > 0:
                 best_minus = np.sqrt(np.mean((samples_below - best_ps)**2))
                 
             intervals[title_prefix] = (best_ps - best_minus, best_ps + best_plus)
    
    pass_fail_status = None
    if "Truth" in intervals and "Recon" in intervals:
        t_min, t_max = intervals["Truth"]
        r_min, r_max = intervals["Recon"]
        
        # Check overlap
        overlap = max(t_min, r_min) <= min(t_max, r_max)
        pass_fail_status = overlap
    
    for i, res in enumerate(datasets):
        title_prefix = titles[i]
        
        # 1. Mean Image
        ax1 = fig.add_subplot(gs[i, 0])
        
        # Use mean_im from result (it is an ehtim image object or pre-averaged array)
        if isinstance(res['mean_im'], eh.image.Image):
            im_data = res['mean_im'].imarr()
        else:
             im_data = res['mean_im'] # Assumed to be array in aggregated case
        
        # If aggregated, we might need to handle it differently if it's not an ehtim object
        # but for now let's assume we pass the array or similar.
        
        fov_uas = 200
        extent = [-fov_uas/2, fov_uas/2, -fov_uas/2, fov_uas/2]
        
        im1 = ax1.imshow(im_data, cmap='afmhot', origin='lower', extent=extent)
        ax1.set_title(title_prefix)
        ax1.set_xlabel(r'$x\, [\mu{\rm as}]$')
        ax1.set_ylabel(r'$y\, [\mu{\rm as}]$')
        
        # Ring params
        rp = res['ring_params']
        x0 = rp['x0']
        y0 = rp['y0']
        r = rp['r']
        r_err = rp['r_err']
        
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Mean ring
        xc = x0 + r*np.cos(theta)
        yc = y0 + r*np.sin(theta)
        ax1.plot(xc, yc, 'b--', lw=1)
        
        # Shaded region
        r_inner = max(0, r - r_err)
        r_outer = r + r_err
        
        xc_in = x0 + r_inner*np.cos(theta)
        yc_in = y0 + r_inner*np.sin(theta)
        xc_out = x0 + r_outer*np.cos(theta)
        yc_out = y0 + r_outer*np.sin(theta)
        
        xy_outer = np.column_stack((xc_out, yc_out))
        xy_inner = np.column_stack((xc_in[::-1], yc_in[::-1])) 
        xy_poly = np.vstack((xy_outer, xy_inner))
        
        ax1.fill(xy_poly[:,0], xy_poly[:,1], color='blue', alpha=0.3)
        
        add_sci_colorbar(im1, ax1)
        
        # 2. Raw Cylinder
        ax2 = fig.add_subplot(gs[i, 1])
        qs_raw = res['qs_raw']
        frames, thetas = qs_raw.shape
        dt = res['dt']
        
        im2 = ax2.imshow(qs_raw.T, cmap='afmhot', aspect='auto', origin='lower',
                       extent=[0, frames*dt, -180, 180])
        ax2.set_title(title_prefix + ' Raw Cylinder')
        ax2.set_xlabel(r'$t [G M/c^3]$')
        ax2.set_ylabel(r'$\mathrm{PA} [{\rm deg}]$')
        ax2.set_ylim(-180, 180)
        add_sci_colorbar(im2, ax2)
        
        # 3. Smoothed Normalized Cylinder
        ax3 = fig.add_subplot(gs[i, 2])
        qs_norm = res['qs_norm']
        title_norm_cyl = "Truth Smooth Norm Cyl" if title_prefix == "Truth" else "Recon Smooth Norm Cyl"
        
        im3 = ax3.imshow(qs_norm.T, cmap='afmhot', aspect='auto', origin='lower',
                       extent=[0, frames*dt, -180, 180])
        ax3.set_title(title_norm_cyl)
        ax3.set_xlabel(r'$t [G M/c^3]$')
        ax3.set_ylabel(r'$\mathrm{PA} [{\rm deg}]$')
        ax3.set_ylim(-180, 180)
        add_sci_colorbar(im3, ax3)
        
        # 4. Autocorrelation
        ax4 = fig.add_subplot(gs[i, 3])
        racf = res['racf']
        dtheta = 2.0 
        
        extent_acf = [-racf.shape[0]*dt/2, racf.shape[0]*dt/2, -racf.shape[1]*dtheta/2, racf.shape[1]*dtheta/2]
        
        im4 = ax4.imshow(racf.T, cmap='YlGnBu_r', aspect='auto', origin='lower', extent=extent_acf)
        ax4.set_title(title_prefix + ' Autocorr') 
        ax4.set_xlabel(r'$\Delta t [G M/c^3]$')
        ax4.set_ylabel(r'$\Delta \mathrm{PA} [{\rm deg}]$')
        ax4.set_ylim(-180, 180) 
        
        ax4.contour(res['mask'].T, extent=extent_acf, levels=[0.5], colors='purple', origin='lower')
        
        x_vals = np.linspace(extent_acf[0], extent_acf[1], 100)
        y_vals = res['ps'] * x_vals
        ax4.plot(x_vals, y_vals, 'g--', lw=2, label=f'Slope {res["ps"]:.2f}')
        
        # Shaded region for slope (Ensemble Std)
        ps_std = res.get('ps_std', 0)
        if ps_std > 0:
             y_vals_upper = (res['ps'] + ps_std) * x_vals
             y_vals_lower = (res['ps'] - ps_std) * x_vals
             ax4.fill_between(x_vals, y_vals_lower, y_vals_upper, color='green', alpha=0.3, label=f'Std {ps_std:.2f}')
             
        ax4.legend()
        add_colorbar(im4, ax4)
        
        # Save separate outputs
        type_str = title_prefix.lower()
        prefix = output_dir
        
        np.save(f'{prefix}_patternspeed_{type_str}_autocorr.npy', racf)
        np.save(f'{prefix}_patternspeed_{type_str}_smoothed_norm_cylinder.npy', qs_norm)
        
        # Save Limits for Autocorr
        xlim_acf = [-len(racf[:, 0])*dt/2., len(racf[:, 0])*dt/2.]
        ylim_acf = [-len(racf[0, :])*dtheta/2, len(racf[0, :])*dtheta/2.]
        np.save(f'{prefix}_patternspeed_{type_str}_autocorr_xlim.npy', xlim_acf)
        np.save(f'{prefix}_patternspeed_{type_str}_autocorr_ylim.npy', ylim_acf)
        
        # 5. Histogram (MCMC)
        ax5 = fig.add_subplot(gs[i, 4])
        title_samples = "Truth Samples" if title_prefix == "Truth" else "Recon Samples"
        
        # Pass/Fail in Title for Recon
        if title_prefix == "Recon" and pass_fail_status is not None:
             status_str = "PASS" if pass_fail_status else "FAIL"
             color_str = "green" if pass_fail_status else "red"
             title_samples += f" ({status_str})"
             ax5.set_title(title_samples, color=color_str, fontweight='bold')
        else:
             ax5.set_title(title_samples)
        
        if res.get('mcmc'):
            mcmc = res['mcmc']
            samples = mcmc['samples']
            
            ax5.hist(samples, bins=50, histtype='step', color='k')
            
            best_ps = res['ps']
            median = mcmc['median']
            med_plus = mcmc['median_plus_sigma']
            med_minus = mcmc['median_minus_sigma']
            
            # Recalculate Best Errors (needed for array saving too)
            samples_above = samples[samples > best_ps]
            samples_below = samples[samples < best_ps]
            
            best_plus = 0.0
            if len(samples_above) > 0:
                best_plus = np.sqrt(np.mean((samples_above - best_ps)**2))
                
            best_minus = 0.0
            if len(samples_below) > 0:
                best_minus = np.sqrt(np.mean((samples_below - best_ps)**2))
            
            label_med = f'Median: {median:.2f}' + r'$^{+%.2f}_{-%.2f}$' % (med_plus, med_minus)
            label_best = f'Best Bet: {best_ps:.2f}' + r'$^{+%.2f}_{-%.2f}$' % (best_plus, best_minus)
            
            ax5.axvline(median, color='g', ls='-', label=label_med)
            ax5.axvline(best_ps, color='r', ls='--', label=label_best)
            
            ax5.set_xlabel(r'$\Omega_p$')
            ax5.legend(fontsize='small')
            
            np.save(f'{prefix}_patternspeed_{type_str}_mcmc_samples.npy', samples)
            
            # Save Uncertainty Array
            # [bestbet_ps, plus_sigma, minus_sigma, median, median_plus, median_minus, mode, mode_plus, mode_minus]
            mode_val = mcmc['mode']
            
            # Calculate Mode percentiles
            # p[1] = median, p[2] = median + med_plus, p[0] = median - med_minus
            p_high = median + med_plus
            p_low = median - med_minus
            
            mode_plus = p_high - mode_val
            mode_minus = mode_val - p_low
            
            uncertainty_array = np.array([
                 best_ps, best_plus, best_minus,
                 median, med_plus, med_minus,
                 mode_val, mode_plus, mode_minus
            ])
            np.save(f'{prefix}_patternspeed_{type_str}_uncertainty.npy', uncertainty_array)
            
        else:
            ax5.text(0.5, 0.5, "No MCMC", ha='center', va='center')

    #plt.tight_layout()
    fig.savefig(f'{output_dir}_patternspeed_summary.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

def main():
    args = create_parser().parse_args()
    
    # Expand glob patterns for input files
    input_files = []
    for pattern in args.input:
        matched = glob.glob(pattern)
        if matched:
            input_files.extend(matched)
        else:
            # If no match, treat as literal filename
            input_files.append(pattern)
    
    if not input_files:
        print("No input files found matching the pattern(s).")
        return
        
    print(f"Found {len(input_files)} input file(s).")

    # Load observation to determine time range
    print(f"Loading observation: {args.data}")
    # Suppress output
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
             obs = eh.obsdata.load_uvfits(args.data)
             obs.add_scans()
    
    obslist = obs.split_obs()
    obs_times = np.array([o.data['time'][0] for o in obslist])
    
    # Time filtering
    if args.tstart is not None or args.tstop is not None:
        ts = args.tstart if args.tstart is not None else obs_times.min()
        te = args.tstop if args.tstop is not None else obs_times.max()
        
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                obs = obs.flag_UT_range(UT_start_hour=ts, UT_stop_hour=te, output='flagged')
                obs.add_scans()
                obslist = obs.split_obs()
                
        if len(obslist) == 0:
             print("No data remaining after time flagging.")
             return
        obs_times = np.array([o.data['time'][0] for o in obslist])
    # Settings
    npix = 200
    fov = 200 * eh.RADPERUAS
    


    # --- Parallel Processing ---
    max_workers = args.ncores
    print(f"Starting parallel processing with {max_workers} workers...")
    
    # Partial function to fix constant arguments
    process_func = functools.partial(process_movie, 
                                     times=obs_times, 
                                     fov=fov, 
                                     npix=npix, 
                                     n_samples=args.nsamples)

    recon_results_list = []
    # If single file and 1 core, avoid pool overhead? Not critical, but safe.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_gen = executor.map(process_func, input_files)
        for res in tqdm(results_gen, total=len(input_files), desc="Processing files"):
            if res is not None:
                recon_results_list.append(res)
                
    if not recon_results_list:
        print("No results generated.")
        return

    # --- Aggregation ---
    print("Aggregating results...")
    
    def aggregate_results(res_list):
        n = len(res_list)
        if n == 0: return None
        
        # Base result from first item
        agg = res_list[0].copy()
        
        if n == 1:
            return agg
            
        # Scalar/Array Mean/Std
        # pattern speed
        ps_vals = np.array([r['ps'] for r in res_list])
        agg['ps'] = np.mean(ps_vals)
        agg['ps_std'] = np.std(ps_vals)
        
        # Ring Params
        r_vals = np.array([r['ring_params']['r'] for r in res_list])
        agg['ring_params']['r'] = np.mean(r_vals)
        agg['ring_params']['r_std'] = np.std(r_vals)
        
        # Derr error should be correctly propagated
        r_err_vals = np.array([r['ring_params']['r_err'] for r in res_list])
        avg_r_err = np.mean(r_err_vals)
        total_r_err = np.sqrt(avg_r_err**2 + agg['ring_params']['r_std']**2)
        agg['ring_params']['r_err'] = total_r_err

        # MCMC Samples - Stack
        if agg.get('mcmc'):
             all_samples = np.concatenate([r['mcmc']['samples'] for r in res_list])
             
             # Recompute stats on stacked samples
             mean_ps = np.mean(all_samples)
             std_ps = np.std(all_samples)
             percentiles = np.percentile(all_samples, [15.865, 50, 84.135]) 
             median = percentiles[1]
             median_plus = percentiles[2] - median
             median_minus = median - percentiles[0]
             
             counts, bin_edges = np.histogram(all_samples, bins=50)
             modal_bin_index = np.argmax(counts)
             mode_value = 0.5 * (bin_edges[modal_bin_index] + bin_edges[modal_bin_index + 1])
             
             agg['mcmc'] = {
                 'samples': all_samples,
                 'mean': mean_ps,
                 'std': std_ps,
                 'median': median,
                 'median_plus_sigma': median_plus,
                 'median_minus_sigma': median_minus,
                 'mode': mode_value
             }
        
        # Mean Image
        all_imarrs = np.array([r['mean_im'].imarr() for r in res_list])
        mean_imarr = np.mean(all_imarrs, axis=0)
        agg['mean_im'] = mean_imarr # Pass as array
        
        # Autocorrelation
        all_racfs = np.array([r['racf'] for r in res_list])
        agg['racf'] = np.mean(all_racfs, axis=0)
        
        # Raw Cylinders
        all_qs_raw = np.array([r['qs_raw'] for r in res_list])
        agg['qs_raw'] = np.mean(all_qs_raw, axis=0)
        
        # Norm Cylinders
        all_qs_norm = np.array([r['qs_norm'] for r in res_list])
        agg['qs_norm'] = np.mean(all_qs_norm, axis=0)
        
        # Mask - Recompute based on mean racf
        if input_files:
            xi_crit_factor = determine_xi_crit_factor(input_files[0])
        else:
            xi_crit_factor = 0.6
            
        agg_dt = res_list[0]['dt'] if 'dt' in res_list[0] else 1.0
        _, _, mask = calculate_pattern_speed(agg['racf'], agg_dt, dtheta=2.0, xi_crit_factor=xi_crit_factor)
        agg['mask'] = mask

        return agg

    final_recon_res = aggregate_results(recon_results_list)

    # Truth Processing (if requested)
    truth_res = None
    if args.truthmv:
        print(f"Processing Truth: {args.truthmv}")
        # Truth usually single file
        truth_res = process_movie(args.truthmv, obs_times, fov, npix, n_samples=args.nsamples)
        
    print(f"Saving results to {args.outpath}...")
    plot_results(args.outpath, final_recon_res, truth_res)
    
    # Save statistics
    stats_dict = {
        'ps': final_recon_res['ps'],
        'ring_r': final_recon_res['ring_params']['r'],
        'ring_r_err': final_recon_res['ring_params']['r_err']
    }
    if 'ps_std' in final_recon_res:
        stats_dict['ps_std'] = final_recon_res['ps_std']
        
    np.save(f'{args.outpath}_patternspeed_stats.npy', stats_dict)
    print("Done!")

if __name__ == "__main__":
    main()
