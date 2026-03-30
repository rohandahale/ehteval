######################################################################
# Author: Rohan Dahale, Date: 09 December 2025
# Based on:  Script's from Freek Roelofs, Marianna Foschini
######################################################################

import os

# Set environment variables
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import ehtim as eh
import ehtplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import argparse
import os
import multiprocessing
from functools import partial
import subprocess
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set plot style
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["font.family"] = "sans-serif"

def create_parser():
    p = argparse.ArgumentParser(description="Visualize reconstruction results: Total Intensity GIF, Linear Polarization GIF, and Visibility Variance Plot.")
    p.add_argument('-d', '--data', type=str, required=True, help='Path to UVFITS data file.')
    p.add_argument('-i', '--input', type=str, required=True, help='Path to reconstruction HDF5 file.')
    p.add_argument('--truthmv', type=str, default=None, help='Path to truth HDF5 file (optional).')
    p.add_argument('-o', '--outpath', type=str, default='./visualize_output', help='Output path prefix (without extension).')
    p.add_argument('--tstart', type=float, default=None, help='Start time (in UT hours) for data')
    p.add_argument('--tstop', type=float, default=None, help='Stop time (in UT hours) for data')
    p.add_argument('--fps', type=int, default=10, help='Frames per second for GIFs.')
    p.add_argument('-n', '--ncores', type=int, default=16, help='Number of cores for parallel processing.')
    p.add_argument('--skip-total', action='store_true', help='Skip generating total intensity GIF.')
    p.add_argument('--skip-lp', action='store_true', help='Skip generating linear polarization GIF.')
    p.add_argument('--skip-visvar', action='store_true', help='Skip generating visibility variance plot.')
    return p

def process_obs_local(obs, recon_path, truth_path=None, tstart=None, tstop=None):
    obs.add_scans()
    obslist = obs.split_obs()
    
    obs_times = np.array([o.data['time'][0] for o in obslist])
    data_min_t = obs_times.min()
    data_max_t = obs_times.max()
    print(f"Data time range: {data_min_t:.3f} - {data_max_t:.3f} h")
    
    if tstart is not None or tstop is not None:
        ts = tstart if tstart is not None else data_min_t
        te = tstop if tstop is not None else data_max_t
        print(f"Time flagging data to use in range: {ts:.3f} - {te:.3f} h")
        
        with open(os.devnull, 'w') as devnull:
             with redirect_stdout(devnull), redirect_stderr(devnull):
                 obs = obs.flag_UT_range(UT_start_hour=ts, UT_stop_hour=te, output='flagged')
                 obs.add_scans()
                 obslist = obs.split_obs()
        
        if not obslist:
             print("No data remaining after time flagging.")
             return obs, None
        
        obs_times = np.array([o.data['time'][0] for o in obslist])
        print(f"New data time range: {obs_times.min():.3f} - {obs_times.max():.3f} h")
        
    filtered_times = obs_times
    print(f"Using {len(filtered_times)} observation times.")
    
    # Check Movie ranges for warning
    recon_mv = eh.movie.load_hdf5(recon_path)
    min_t_list = [min(recon_mv.times)]
    max_t_list = [max(recon_mv.times)]
    
    if truth_path:
        truth_mv = eh.movie.load_hdf5(truth_path)
        min_t_list.append(min(truth_mv.times))
        max_t_list.append(max(truth_mv.times))
        
    movie_min_t = max(min_t_list)
    movie_max_t = min(max_t_list)
    print(f"Effective Movie time range: {movie_min_t:.3f} - {movie_max_t:.3f} h")

    if movie_min_t > filtered_times.min() or movie_max_t < filtered_times.max():
         print("Warning: Movie times do not span the whole duration of data. Extrapolation will be used.")

    return obs, filtered_times

def process_frame_worker(t, movie, fov, npix):
    im = movie.get_image(t)
    im = im.blur_circ(fwhm_i=0, fwhm_pol=0).regrid_image(fov, npix)
    I = im.imvec.reshape(im.ydim, im.xdim)
    Q = im.qvec.reshape(im.ydim, im.xdim)
    U = im.uvec.reshape(im.ydim, im.xdim)
    
    return {'I': I, 'Q': Q, 'U': U, 'time': t, 'rf': im.rf, 'psize': im.psize}

def process_movie_parallel(movie, times, fov, npix, ncores=1):
    pool = multiprocessing.Pool(processes=ncores)
    func = partial(process_frame_worker, movie=movie, fov=fov, npix=npix)
    results = pool.map(func, times)
    pool.close()
    pool.join()
    return results

def compute_static_dynamic(frames_data):
    I_stack = np.array([f['I'] for f in frames_data])
    Q_stack = np.array([f['Q'] for f in frames_data])
    U_stack = np.array([f['U'] for f in frames_data])
    
    static_I = np.median(I_stack, axis=0)
    static_Q = np.median(Q_stack, axis=0)
    static_U = np.median(U_stack, axis=0)
    
    processed_frames = []
    for i in range(len(frames_data)):
        dyn_I = I_stack[i] - static_I
        dyn_Q = Q_stack[i] - static_Q
        dyn_U = U_stack[i] - static_U
        
        processed_frames.append({
            'total': {'I': I_stack[i], 'Q': Q_stack[i], 'U': U_stack[i]},
            'dynamic': {'I': dyn_I, 'Q': dyn_Q, 'U': dyn_U},
            'static': {'I': static_I, 'Q': static_Q, 'U': static_U},
            'time': frames_data[i]['time'],
            'rf': frames_data[i]['rf'],
            'psize': frames_data[i]['psize']
        })
        
    return processed_frames, {'I': static_I, 'Q': static_Q, 'U': static_U}

def add_scale_bar(ax, fov, color='white'):
    bar_length = fov / 4.0
    # Top left: x is positive (left), y is positive (top).
    # x axis is inverted: [fov/2, -fov/2]
    # Position: 10% padding from bottom-left corner
    x_start = fov/2 * 0.9 
    x_end = x_start - bar_length
    y_pos = -fov/2 * 0.9
    
    ax.plot([x_start, x_end], [y_pos, y_pos], color=color, linewidth=2)
    ax.text((x_start + x_end)/2, y_pos + fov*0.02, f"{int(bar_length)} $\mu$as", 
            color=color, ha='center', va='bottom', fontsize=14)

def get_tb(frame_data):
    rf = frame_data['rf']
    psize = frame_data['psize']
    return 3.254e13 / (rf**2 * psize**2) / 1e10

# --- Parallel Rendering Functions ---

def render_total_frame(args):
    # Support both 8-tuple (legacy) and 9-tuple (with label)
    if len(args) == 9:
        idx, recon_frame, truth_frame, time_val, fov, max_I, max_dyn, tmp_dir, label = args
    else:
        idx, recon_frame, truth_frame, time_val, fov, max_I, max_dyn, tmp_dir = args
        label = 'Reconstruction'
    
    if label is None: label = 'Reconstruction'
    
    has_truth = truth_frame is not None
    nrows = 2 if has_truth else 1
    ncols = 3
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 6 * nrows))
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.95, hspace=0.1, wspace=0.1)
    
    if nrows == 1:
        axes = np.array([axes])
        
    if has_truth:
        row_labels = ['Truth', label]
        datasets = [truth_frame, recon_frame]
    else:
        row_labels = [label]
        datasets = [recon_frame]
        
    col_labels = ['Total', 'Dynamic', 'Static']
    lims = [fov/2, -fov/2, -fov/2, fov/2]
    
    for i, ax_row in enumerate(axes):
        ax_row[0].set_ylabel(row_labels[i], fontsize=18)
        for j, ax in enumerate(ax_row):
            if i == 0:
                ax.set_title(col_labels[j], fontsize=18)
            ax.set_xticks([])
            ax.set_yticks([])

    for i, d in enumerate(datasets):
        tb_factor = get_tb(d)
        
        # Total
        im_tot = axes[i, 0].imshow(d['total']['I'] * tb_factor, cmap='afmhot_us', vmin=0, vmax=max_I, extent=lims)
        
        # Dynamic
        dyn_data = d['dynamic']['I'] * tb_factor
        im_dyn = axes[i, 1].imshow(dyn_data, cmap='coolwarm', vmin=-max_dyn, vmax=max_dyn, extent=lims)
        
        # Contour for Reconstruction Dynamic (i=1 if has_truth)
        if has_truth and i == 1:
            truth_dyn = truth_frame['dynamic']['I'] * get_tb(truth_frame)
            levels = [0.3 * max_dyn]
            axes[i, 1].contour(np.abs(truth_dyn), levels=levels, extent=lims, colors='black', alpha=0.7, linewidths=1, origin='upper')
        
        # Static
        im_stat = axes[i, 2].imshow(d['static']['I'] * tb_factor, cmap='afmhot_us', vmin=0, vmax=max_I, extent=lims)
        
        # Colorbars
        if i == nrows - 1:
            # Column 0
            divider0 = make_axes_locatable(axes[i, 0])
            cax0 = divider0.append_axes("bottom", size="5%", pad=0.05)
            fig.colorbar(im_tot, cax=cax0, orientation='horizontal', label='$T_B$ ($10^{10}$ K)')
            
            # Column 1
            divider1 = make_axes_locatable(axes[i, 1])
            cax1 = divider1.append_axes("bottom", size="5%", pad=0.05)
            fig.colorbar(im_dyn, cax=cax1, orientation='horizontal', label='$T_B$ ($10^{10}$ K)')
            
            # Column 2
            divider2 = make_axes_locatable(axes[i, 2])
            cax2 = divider2.append_axes("bottom", size="5%", pad=0.05)
            fig.colorbar(im_stat, cax=cax2, orientation='horizontal', label='$T_B$ ($10^{10}$ K)')
            
    # Add scale bar to top-left panel
    add_scale_bar(axes[0, 0], fov)

    plt.suptitle(f"{time_val:.2f} UT", y=0.98, fontsize=22, color='black')
    
    fname = os.path.join(tmp_dir, f"frame_{idx:04d}.png")
    plt.savefig(fname, dpi=100)
    plt.close(fig)
    return fname

def plot_vectors(ax, I, Q, U, cmap_bg='binary', vmin=None, vmax=None, is_dynamic=False, tb_factor=1.0, lims=None):
    # Background Intensity
    im = ax.imshow(I * tb_factor, cmap=cmap_bg, vmin=vmin, vmax=vmax, extent=lims, origin='upper')
    
    skip = 10
    ny, nx = I.shape
    fov = lims[0] - lims[1] # approx
    pixel_size = fov / nx
    y, x = np.mgrid[slice(-fov/2, fov/2, pixel_size), slice(-fov/2, fov/2, pixel_size)]
    
    amp = np.sqrt(Q**2 + U**2)
    scal = np.max(amp) * 0.5
    
    angle = np.angle(Q + 1j*U)
    vx = (-np.sin(angle/2) * amp/scal)
    vy = (np.cos(angle/2) * amp/scal)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mfrac = amp / np.abs(I)
        
    pcut = 0.1
    Imax = np.max(np.abs(I))
    QUmax = np.max(amp)
    
    mask = (np.abs(I) < pcut * Imax) | (amp < pcut * QUmax)
    
    mfrac_m = np.ma.masked_where(mask, mfrac)
    vx_m = np.ma.masked_where(mask, vx)
    vy_m = np.ma.masked_where(mask, vy)
    
    # Quiver
    cnorm = Normalize(vmin=0.0, vmax=0.5)
    q = ax.quiver(-x[::skip, ::skip], -y[::skip, ::skip], 
                  vx_m[::skip, ::skip], vy_m[::skip, ::skip],
                  mfrac_m[::skip, ::skip], 
                  cmap='rainbow',
                  norm=cnorm,
                  headlength=0, headwidth=1, pivot='mid', scale=16, width=0.01)
    return q, im

def render_lp_frame(args):
    # Support both 8-tuple (legacy) and 9-tuple (with label)
    if len(args) == 9:
        idx, recon_frame, truth_frame, time_val, fov, max_I, max_dyn, tmp_dir, label = args
    else:
        idx, recon_frame, truth_frame, time_val, fov, max_I, max_dyn, tmp_dir = args
        label = 'Reconstruction'
    
    if label is None: label = 'Reconstruction'
    
    has_truth = truth_frame is not None
    nrows = 2 if has_truth else 1
    ncols = 3
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 6 * nrows))
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.95, hspace=0.1, wspace=0.1)
    
    if nrows == 1:
        axes = np.array([axes])
        
    if has_truth:
        row_labels = ['Truth', label]
        datasets = [truth_frame, recon_frame]
    else:
        row_labels = [label]
        datasets = [recon_frame]
        
    col_labels = ['Total', 'Dynamic', 'Static']
    lims = [fov/2, -fov/2, -fov/2, fov/2]
    
    for i, ax_row in enumerate(axes):
        ax_row[0].set_ylabel(row_labels[i], fontsize=18)
        for j, ax in enumerate(ax_row):
            if i == 0:
                ax.set_title(col_labels[j], fontsize=18)
            ax.set_xticks([])
            ax.set_yticks([])

    for i, d in enumerate(datasets):
        tb_factor = get_tb(d)
        
        # Total: Background Tb (afmhot), Vectors |m|
        q_tot, im_tot = plot_vectors(axes[i, 0], d['total']['I'], d['total']['Q'], d['total']['U'], 
                     cmap_bg='afmhot', vmin=0, vmax=max_I, tb_factor=tb_factor, lims=lims)
        
        # Dynamic: Background Tb (coolwarm), Vectors |m|
        q_dyn, im_dyn = plot_vectors(axes[i, 1], d['dynamic']['I'], d['dynamic']['Q'], d['dynamic']['U'], 
                     cmap_bg='coolwarm', vmin=-max_dyn, vmax=max_dyn, is_dynamic=True, tb_factor=tb_factor, lims=lims)
        
        # Contour for Reconstruction Dynamic (i=1 if has_truth)
        if has_truth and i == 1:
            truth_dyn = truth_frame['dynamic']['I'] * get_tb(truth_frame)
            levels = [0.3 * max_dyn]
            axes[i, 1].contour(np.abs(truth_dyn), levels=levels, extent=lims, colors='black', alpha=0.7, linewidths=1, origin='upper')

        # Static: Background Tb (afmhot), Vectors |m|
        q_stat, im_stat = plot_vectors(axes[i, 2], d['static']['I'], d['static']['Q'], d['static']['U'], 
                     cmap_bg='afmhot', vmin=0, vmax=max_I, tb_factor=tb_factor, lims=lims)
        
        if i == nrows - 1:
             # Horizontal colorbars
             
             # Column 0: Tb (Total)
             divider0 = make_axes_locatable(axes[i, 0])
             cax0 = divider0.append_axes("bottom", size="5%", pad=0.05)
             fig.colorbar(im_tot, cax=cax0, orientation='horizontal', label='$T_B$ ($10^{10}$ K)')
             
             # Column 1: Tb (Dynamic)
             divider1 = make_axes_locatable(axes[i, 1])
             cax1 = divider1.append_axes("bottom", size="5%", pad=0.05)
             fig.colorbar(im_dyn, cax=cax1, orientation='horizontal', label='$T_B$ ($10^{10}$ K)')
             
             # Column 2: |m| (Static)
             divider2 = make_axes_locatable(axes[i, 2])
             cax2 = divider2.append_axes("bottom", size="5%", pad=0.05)
             fig.colorbar(q_stat, cax=cax2, orientation='horizontal', label='$|m|$')

    # Add scale bar to top-left panel
    add_scale_bar(axes[0, 0], fov)

    plt.suptitle(f"{time_val:.2f} UT", y=0.98, fontsize=22, color='black')
    
    fname = os.path.join(tmp_dir, f"frame_{idx:04d}.png")
    plt.savefig(fname, dpi=100)
    plt.close(fig)
    return fname

def generate_gif_parallel(render_func, recon_data, truth_data, times, outpath, fps, fov, ncores, suffix, label='Reconstruction'):
    print(f"Generating {suffix} GIF in parallel...")
    
    # Calculate limits first
    max_I = 0
    max_dyn = 0
    
    for d in recon_data:
        tb = get_tb(d)
        max_I = max(max_I, np.max(d['total']['I']) * tb)
        max_dyn = max(max_dyn, np.max(np.abs(d['dynamic']['I'])) * tb)
        
    if truth_data:
        for d in truth_data:
            tb = get_tb(d)
            max_I = max(max_I, np.max(d['total']['I']) * tb)
            max_dyn = max(max_dyn, np.max(np.abs(d['dynamic']['I'])) * tb)
            
    # Create temp directory
    tmp_dir = f"{outpath}_tmp_{suffix}"
    if os.path.exists(tmp_dir):
        import shutil
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    
    # Prepare arguments
    tasks = []
    for i, t in enumerate(times):
        r_frame = recon_data[i]
        t_frame = truth_data[i] if truth_data else None
        tasks.append((i, r_frame, t_frame, t, fov, max_I, max_dyn, tmp_dir, label))
        
    # Run parallel rendering
    pool = multiprocessing.Pool(processes=ncores)
    pool.map(render_func, tasks)
    pool.close()
    pool.join()
    
    # Stitch with ffmpeg
    out_file = f"{outpath}_{suffix}.gif"
    if os.path.exists(out_file):
        os.remove(out_file)
        
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(tmp_dir, 'frame_%04d.png'),
        '-vf', 'split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
        out_file
    ]
    
    #print(f"Stitching frames with ffmpeg: {' '.join(cmd)}")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir)
    print(f"Saved {out_file}")

def compute_vis_worker(frame_tuple, uv_points, fov, npix):
    I, Q, U = frame_tuple
    im = eh.image.Image(I, psize=fov/npix, ra=17.761121055553343, dec=-29.00784305556, rf=230000000000.0, source='SGRA', mjd=57850)
    im.qvec = Q.flatten()
    im.uvec = U.flatten()
    vis, visQ, visU, _ = im.sample_uv(uv_points)
    return vis, visQ, visU

def compute_variance_parallel(frames_data, uv_points, fov, npix, ncores=1):
    pool = multiprocessing.Pool(processes=ncores)
    func = partial(compute_vis_worker, uv_points=uv_points, fov=fov, npix=npix)
    inputs = [(f['total']['I'], f['total']['Q'], f['total']['U']) for f in frames_data]
    results = pool.map(func, inputs)
    pool.close()
    pool.join()
    
    # results is list of (vis, visQ, visU)
    visI = np.array([r[0] for r in results])
    visQ = np.array([r[1] for r in results])
    visU = np.array([r[2] for r in results])
    visP = visQ + 1j * visU
    
    def get_var(vis_stack):
        # Amp Variance
        amp_stack = np.abs(vis_stack)
        mean_amp = np.mean(amp_stack, axis=0)
        std_amp = np.std(amp_stack, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            var_amp = std_amp / mean_amp
        var_amp[mean_amp == 0] = 0
        
        # Phase Variance
        angles = np.angle(vis_stack)
        phasors = np.exp(1j * angles)
        phasors[np.abs(vis_stack) == 0] = 0
        mean_phasor = np.mean(phasors, axis=0)
        R = np.abs(mean_phasor)
        var_phase = 1 - R
            
        return var_amp, var_phase
        
    var_amp_I, var_phase_I = get_var(visI)
    var_amp_Q, var_phase_Q = get_var(visQ)
    var_amp_U, var_phase_U = get_var(visU)
    var_amp_P, var_phase_P = get_var(visP)
    
    return (var_amp_I, var_phase_I), (var_amp_Q, var_phase_Q), (var_amp_U, var_phase_U), (var_amp_P, var_phase_P)

def plot_variance(recon_data, truth_data, obs, outpath, fov, npix, ncores, label='Reconstruction'):
    print("Plotting Visibility Variance...")
    
    U = np.linspace(-10.0e9, 10.0e9, npix)
    V = np.linspace(-10.0e9, 10.0e9, npix)
    UU, VV = np.meshgrid(U, V)
    UV = np.vstack((UU.flatten(), VV.flatten())).T

    # Calculate mask based on max UV radius from observations
    u_obs = obs.data['u']
    v_obs = obs.data['v']
    max_baseline = np.max(np.sqrt(u_obs**2 + v_obs**2))
    grid_radii = np.sqrt(UV[:, 0]**2 + UV[:, 1]**2)
    mask = grid_radii <= max_baseline
    
    # Calculate variances
    r_vars = compute_variance_parallel(recon_data, UV, fov, npix, ncores)
    
    has_truth = truth_data is not None
    if has_truth:
        t_vars = compute_variance_parallel(truth_data, UV, fov, npix, ncores)

    # Save Variance Matrices
    print("Saving Variance Matrices to .npy...")
    # r_vars: [ (amp_I, phase_I), (amp_Q, phase_Q), ... ]
    # pols order: I, Q, U, P
    pols = ['I', 'Q', 'U', 'P']
    
    # Recon
    for i, pol in enumerate(pols):
        np.save(f"{outpath}_recon_amp_var_{pol}.npy", r_vars[i][0].reshape(npix, npix))
        np.save(f"{outpath}_recon_phase_var_{pol}.npy", r_vars[i][1].reshape(npix, npix))
        
    # Truth
    if has_truth:
        for i, pol in enumerate(pols):
            np.save(f"{outpath}_truth_amp_var_{pol}.npy", t_vars[i][0].reshape(npix, npix))
            np.save(f"{outpath}_truth_phase_var_{pol}.npy", t_vars[i][1].reshape(npix, npix))
        
    # Added two extra rows for the Log-Ratio Metrics (Amp and Phase) if truth exists
    if has_truth:
        nrows = 6 
    else:
        nrows = 2

    ncols = 4
    
    # Use gridspec to create a column for colorbars
    fig = plt.figure(figsize=(16, 4 * nrows))
    gs = fig.add_gridspec(nrows, ncols + 1, width_ratios=[1, 1, 1, 1, 0.05])
    
    extent = [10, -10, -10, 10]
    
    col_names = ['I', 'Q', 'U', 'P']
    
    # Re-plotting with global row limits
    for i in range(nrows):
        # Determine max/min for the row
        vmin = 0
        vmax = 1
        cmap = 'binary_us'
        
        if has_truth:
            if i < 2: # Amp Rows (Truth, Recon)
                all_amp = []
                for j in range(ncols):
                    all_amp.append(t_vars[j][0])
                    all_amp.append(r_vars[j][0])
                vmax = np.max([np.max(d[mask]) for d in all_amp])
                cmap = 'binary_us'
            
            elif i < 4: # Phase Rows (Truth, Recon)
                all_phase = []
                for j in range(ncols):
                    all_phase.append(t_vars[j][1])
                    all_phase.append(r_vars[j][1])
                vmax = np.max([np.max(d[mask]) for d in all_phase])
                cmap = 'inferno'
            
            elif i == 4: # Amplitude Log-Ratio (Hallucination Metric)
                all_ratios = []
                epsilon = 1e-12 # Prevent div by zero
                for j in range(ncols):
                    # Metric: log10( Recon / Truth )
                    # + epsilon to avoid log(0) or div/0 issues
                    num = r_vars[j][0] + epsilon
                    den = t_vars[j][0] + epsilon
                    ratio = np.log10(num / den)
                    all_ratios.append(ratio)
                
                # Center the colormap around 0
                max_abs_val = np.max([np.max(np.abs(d[mask])) for d in all_ratios])
                vmax = max_abs_val
                vmin = -max_abs_val
                cmap = 'RdBu_r' # Red = Hallucination (Recon > Truth), Blue = Over-smooth
            
            elif i == 5: # Phase Log-Ratio (Hallucination Metric)
                all_ratios = []
                epsilon = 1e-12 # Prevent div by zero
                for j in range(ncols):
                    # Metric: log10( Recon / Truth )
                    # + epsilon to avoid log(0) or div/0 issues
                    num = r_vars[j][1] + epsilon
                    den = t_vars[j][1] + epsilon
                    ratio = np.log10(num / den)
                    all_ratios.append(ratio)
                
                # Center the colormap around 0
                max_abs_val = np.max([np.max(np.abs(d[mask])) for d in all_ratios])
                vmax = max_abs_val
                vmin = -max_abs_val
                cmap = 'RdBu_r' # Red = Hallucination (Recon > Truth), Blue = Over-smooth

        else:
            if i == 0:
                all_amp = [r_vars[j][0] for j in range(ncols)]
                vmax = np.max([np.max(d[mask]) for d in all_amp])
                cmap = 'binary_us'
            else:
                all_phase = [r_vars[j][1] for j in range(ncols)]
                vmax = np.max([np.max(d[mask]) for d in all_phase])
                cmap = 'inferno'
                
        for j in range(ncols):
            ax = fig.add_subplot(gs[i, j])
            
            if has_truth:
                if i == 0: data = t_vars[j][0]
                elif i == 1: data = r_vars[j][0]
                elif i == 2: data = t_vars[j][1]
                elif i == 3: data = r_vars[j][1]
                elif i == 4: 
                    epsilon = 1e-12
                    data = np.log10((r_vars[j][0] + epsilon) / (t_vars[j][0] + epsilon))
                elif i == 5:
                    epsilon = 1e-12
                    data = np.log10((r_vars[j][1] + epsilon) / (t_vars[j][1] + epsilon))
            else:
                if i == 0: data = r_vars[j][0]
                elif i == 1: data = r_vars[j][1]
                
            im = ax.imshow(data.reshape(npix, npix), cmap=cmap, extent=extent, 
                           origin='lower', vmin=vmin, vmax=vmax)
            
            # Add max value text
            # For the ratio plot, we care about the most extreme deviation (min or max)
            if i == 4 or i == 5:
                # current_max = np.max(data[mask])
                # current_min = np.min(data[mask])
                # Show whichever is larger in magnitude
                val_to_show = np.max(data[mask]) #current_max if abs(current_max) > abs(current_min) else current_min
                ax.text(0.95, 0.95, f"peak: {val_to_show:.2f}", transform=ax.transAxes,
                        ha='right', va='top', color='black', fontsize=14)
            else:
                current_max = np.max(data[mask])
                ax.text(0.95, 0.95, f"max: {current_max:.2f}", transform=ax.transAxes,
                        ha='right', va='top', color='red', fontsize=14)
            
            if i == 0:
                ax.set_title(col_names[j], fontsize=18)
            if j == 0:
                if has_truth:
                    lbls = ['Truth Amp Var', f'{label} Amp Var', 
                            'Truth Phase Var', f'{label} Phase Var', 
                            f'Log10({label[0]}/T) Amp', f'Log10({label[0]}/T) Phase'] 
                else:
                    lbls = [f'{label} Amp Var', f'{label} Phase Var']
                ax.set_ylabel(lbls[i], fontsize=14)
                
            ax.plot(u_obs/1e9, v_obs/1e9, '.', color='tab:orange', alpha=0.3) # Lowered alpha for better visibility
            ax.plot(-u_obs/1e9, -v_obs/1e9, '.', color='tab:orange', alpha=0.3)
            ax.set_xlim(10, -10)
            ax.set_ylim(-10, 10)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0 and j == 0:
                # Add scale bar
                bar_len = 5
                x_start = 9 
                x_end = x_start - bar_len
                y_pos = -9 
                ax.plot([x_start, x_end], [y_pos, y_pos], color='blue', linewidth=2)
                ax.text((x_start + x_end)/2, y_pos + 0.5, f"{bar_len} G$\lambda$", 
                        color='blue', ha='center', va='bottom', fontsize=14)
            
        # Add colorbar in the last column of gridspec
        cax = fig.add_subplot(gs[i, ncols])
        fig.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.savefig(f'{outpath}_visvar.png', bbox_inches='tight')
    plt.close(fig)

def main():
    args = create_parser().parse_args()
    
    obs = eh.obsdata.load_uvfits(args.data)
    
    obs, times = process_obs_local(obs, args.input, args.truthmv, args.tstart, args.tstop)
    if times is None:
        return
    
    npix = 160
    fov = 160 * eh.RADPERUAS
    
    recon_mv = eh.movie.load_hdf5(args.input)
    recon_mv.reset_interp(bounds_error=False)
    recon_frames = process_movie_parallel(recon_mv, times, fov, npix, args.ncores)
    recon_frames = [f for f in recon_frames if f is not None]
    
    truth_frames = None
    if args.truthmv:
        truth_mv = eh.movie.load_hdf5(args.truthmv)
        truth_mv.reset_interp(bounds_error=False)
        truth_frames = process_movie_parallel(truth_mv, times, fov, npix, args.ncores)
        truth_frames = [f for f in truth_frames if f is not None]
        
    recon_processed, recon_static = compute_static_dynamic(recon_frames)
    truth_processed, truth_static = None, None
    if truth_frames:
        truth_processed, truth_static = compute_static_dynamic(truth_frames)
        
    outdir = os.path.dirname(args.outpath)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        
    # Parallel GIF generation
    if not args.skip_total:
        t0 = time.time()
        generate_gif_parallel(render_total_frame, recon_processed, truth_processed, times, args.outpath, args.fps, 160, args.ncores, "total")
        t1 = time.time()
        print(f"Total Intensity GIF took {t1 - t0:.2f} seconds")
    else:
        print("Skipping Total Intensity GIF (--skip-total)")
    
    if not args.skip_lp:
        t0 = time.time()
        generate_gif_parallel(render_lp_frame, recon_processed, truth_processed, times, args.outpath, args.fps, 160, args.ncores, "lp")
        t1 = time.time()
        print(f"Linear Polarization GIF took {t1 - t0:.2f} seconds")
    else:
        print("Skipping Linear Polarization GIF (--skip-lp)")
    
    if not args.skip_visvar:
        t0 = time.time()
        plot_variance(recon_processed, truth_processed, obs, args.outpath, fov, npix, args.ncores)
        t1 = time.time()
        print(f"Visibility variance plot took {t1 - t0:.2f} seconds")
    else:
        print("Skipping Visibility Variance Plot (--skip-visvar)")
    
    print("Done!")

if __name__ == "__main__":
    main()
