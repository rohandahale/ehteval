######################################################################
# Author: Rohan Dahale, Date: 17 Dec 2025
######################################################################

import os
import sys
import re
import warnings
import numpy as np
import pandas as pd
import ehtim as eh
import argparse
from concurrent.futures import ProcessPoolExecutor
import functools
import glob
from tqdm import tqdm
import subprocess
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import c, k_B
from astropy.stats import circmean, circstd
from contextlib import redirect_stdout, redirect_stderr
import tempfile
import shutil
import gc

# Suppress all warnings
warnings.filterwarnings('ignore')

# Set environment variables to single-threaded
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Set plot style
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["font.family"] = "sans-serif"

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data', type=str, required=True, help='UVFITS data file')
    p.add_argument('--truthmv', type=str, default=None, help='Truth HDF5 movie file (optional)')
    p.add_argument('--input', type=str, nargs='+', required=True, help='Glob pattern(s) or list of HDF5 model files')
    p.add_argument('-o', '--outpath', type=str, default='./vida_pol', help='Output prefix (without extension)')
    p.add_argument('--tstart', type=float, default=None, help='Start time (in UT hours) for data')
    p.add_argument('--tstop', type=float, default=None, help='Stop time (in UT hours) for data')
    p.add_argument('-n', '--ncores', type=int, default=32, help='Total number of cores to use')
    p.add_argument('--blur', type=float, default=0.0, help='Blur (fwhm) in microarcseconds for reconstructions')
    p.add_argument('--blur-truth', type=float, default=0.0, help='Blur (fwhm) in microarcseconds for truth only')
    p.add_argument('--no-regrid', action='store_true', help='Disable regridding of image before fitting')
    p.add_argument('--maxiters', type=int, default=20000, help='Maximum iterations for VIDA optimizer')
    p.add_argument('--stride', type=int, default=10, help='Stride for checkponting and parallel batching (Julia)')
    return p

def kill_julia_process():
    """Kills any existing vida_pol.jl processes AND orphaned Julia workers to free memory."""
    try:
        # Kill the main vida_pol.jl process
        subprocess.run(["pkill", "-f", "vida_pol.jl"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Kill orphaned Julia worker processes spawned by Distributed.addprocs()
        subprocess.run(["pkill", "-f", "julia.*--worker"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Give processes time to terminate
        import time
        time.sleep(2)
    except Exception:
        pass

def run_julia_on_temp(input_movie_path, output_csv, times, procs=1, blur=0.0, no_regrid=False, maxiters=20000, stride=10):
    """
    1. Force kills previous Julia processes.
    2. Loads input_movie_path using ehtim.
    3. Interpolates/extracts frames at `times`.
    4. Saves to a temporary HDF5 file.
    5. Runs vida_pol.jl on the temp file.
    6. Cleans up temp file.
    """
    # Ensure fresh start for memory
    kill_julia_process()

    if os.path.exists(output_csv):
        print(f"Skipping Julia run, {output_csv} exists.")
        return output_csv
    
    # Create temp directory in output path
    out_dir = os.path.dirname(os.path.abspath(output_csv))
    temp_dir = os.path.join(out_dir, "temp_vida")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        
    base_name = os.path.basename(output_csv).replace('.csv', '.hdf5')
    temp_h5 = os.path.join(temp_dir, base_name)
    
    # Load and Interpolate
    # We need to suppress ehtim output
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            mv = eh.movie.load_hdf5(input_movie_path)
            mv.reset_interp(bounds_error=False)

            # Extract frames
            frames = []
            for t in times:
                im = mv.get_image(t)
                im.vvec = np.zeros_like(im.ivec)
                frames.append(im)
            
            # Create new movie
            new_mv = eh.movie.merge_im_list(frames)
            new_mv.save_hdf5(temp_h5)
            
            # Cleanup objects to free memory immediately
            del mv, new_mv, frames
            gc.collect()
        
    # Run Julia
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vida_pol.jl")
    
    cmd = [
        "julia",
        f"--procs={procs}",
        script_path,
        "--input", temp_h5,
        "--outname", output_csv,
        "--stride", str(stride),
        "--blur", str(blur),
        "--fevals", str(maxiters)
        ]
    if no_regrid:
        cmd.append("--regrid") # pass flag to DISABLE regridding (logic inverted in julia)

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    
    # Cleanup temp file
    if os.path.exists(temp_h5):
        os.remove(temp_h5)
        
    # Remove temp directory
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)

    return output_csv

def main():
    args = create_parser().parse_args()

    input_files = []
    for pattern in args.input:
        matched = glob.glob(pattern)
        if matched:
            input_files.extend(matched)
        else:
            input_files.append(pattern)
            
    if not input_files:
        print("No input files found.")
        return

    print(f"Found {len(input_files)} input file(s).")
    
    # Grid times logic from REx
    print(f"Loading observation: {args.data}")
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            obs = eh.obsdata.load_uvfits(args.data)
            obs.add_scans()
            obslist = obs.split_obs()
            
    obs_times = np.array([o.data['time'][0] for o in obslist])
    data_min_t = obs_times.min()
    data_max_t = obs_times.max()
    print(f"Data time range: {data_min_t:.3f} - {data_max_t:.3f} h")

    if args.tstart is not None or args.tstop is not None:
        tstart = args.tstart if args.tstart is not None else data_min_t
        tstop = args.tstop if args.tstop is not None else data_max_t
        print(f"Time flagging data to use in range: {tstart:.3f} - {tstop:.3f} h")
        
        with open(os.devnull, 'w') as devnull:
             with redirect_stdout(devnull), redirect_stderr(devnull):
                 obs = obs.flag_UT_range(UT_start_hour=tstart, UT_stop_hour=tstop, output='flagged')
                 obs.add_scans()
                 obslist = obs.split_obs()
        
        if not obslist:
            print("No data remaining after time flagging.")
            return

        obs_times = np.array([o.data['time'][0] for o in obslist])
        print(f"New data time range: {obs_times.min():.3f} - {obs_times.max():.3f} h")

    times = obs_times
    print(f"Processing {len(times)} time steps.")

    # Resource Allocation
    # Resource Allocation
    n_files = len(input_files)
    # Default to sequential Python processing to save memory
    python_workers = 1 
    julia_procs = args.ncores 
    if julia_procs < 1: julia_procs = 1
        
    print(f"Parallel Config: {python_workers} Python workers, each launching Julia with {julia_procs} procs.")

    # Process Truth first
    truth_csv = None
    if args.truthmv:
        print("Processing truth movie...")
        truth_csv = args.outpath + "_vida_truth.csv"
        # Sequential run, use full ncores for julia
        if not os.path.exists(truth_csv):
             run_julia_on_temp(args.truthmv, truth_csv, times, procs=args.ncores, blur=args.blur_truth, no_regrid=args.no_regrid, maxiters=args.maxiters, stride=args.stride)
        else:
            print(truth_csv)
            print("Truth CSV exists, skipping.")

    # Process Input Files
    tasks = []
    
    for i, f in enumerate(input_files):
        # Extract snapshot number (e.g., '049') from filename like '..._049.hdf5'
        base = os.path.basename(f).replace('.hdf5', '').replace('.h5', '')
        match = re.search(r'_(\d+)$', base)
        if match:
            snap_num = match.group(1)
            csv_name = f"{args.outpath}_vida_{snap_num}.csv"
        else:
            # Fallback: use index if no snapshot number found
            csv_name = f"{args.outpath}_vida_{i}.csv"
        tasks.append((f, csv_name))

    results_csvs = []
    
    process_func = functools.partial(run_julia_on_temp, times=times, procs=julia_procs, blur=args.blur, no_regrid=args.no_regrid, maxiters=args.maxiters, stride=args.stride)
    
    with ProcessPoolExecutor(max_workers=python_workers) as executor:
        futures = {executor.submit(process_func, f, csv): csv for f, csv in tasks}
        
        for future in tqdm(futures, total=len(tasks), desc="Processing movies"):
            res = future.result()
            if res:
                results_csvs.append(res)
    
    if not results_csvs:
        print("No results generated.")
        return

    # Load and Aggregation
    df_first = pd.read_csv(results_csvs[0])
    numeric_cols = df_first.select_dtypes(include=[np.number]).columns
    
    # Read truth
    df_truth = None
    if truth_csv and os.path.exists(truth_csv):
        df_truth = pd.read_csv(truth_csv)
    
    all_dfs = [pd.read_csv(csv) for csv in results_csvs]
    is_bayesian = len(all_dfs) > 1
    
    # Helper to align
    aligned_data = {col: np.zeros((len(all_dfs), len(times))) * np.nan for col in numeric_cols}
    
    for i, df in enumerate(all_dfs):
        if len(df) == len(times):
            for col in numeric_cols:
                if col in df.columns:
                    aligned_data[col][i, :] = df[col]
        else:
            print(f"Warning: Result {i} has {len(df)} rows, expected {len(times)}. Mismatch.")
            n = min(len(df), len(times))
            for col in numeric_cols:
                if col in df.columns:
                     aligned_data[col][i, :n] = df[col][:n]

    # Compute stats
    final_df = pd.DataFrame({'time': times})
    
    for col in numeric_cols:
        if col == 'time': continue
        vals = aligned_data[col]
        
        if is_bayesian:
            # Check for angular columns
            if 'ξ' in col or 'evpa' in col:
                # Circular stats in RADIANS (ξ is radians in CSV, evpa might be too? netevpa in julia returns radians usually)
                # Julia: netevpa(img) = evpa(mean(img)). evpa returns radians in Comrade/VLBISkyModels.
                final_df[f'{col}_mean'] = circmean(vals, axis=0) # defaults to low=-pi, high=pi
                final_df[f'{col}_std'] = circstd(vals, axis=0)
            else:
                final_df[f'{col}_mean'] = np.nanmean(vals, axis=0)
                final_df[f'{col}_std'] = np.nanstd(vals, axis=0)
        else:
            final_df[col] = vals[0, :]
            
    # Special handling for beta2 stats from samples if available
    if is_bayesian:
        re_key = 're_betalp_2'
        im_key = 'im_betalp_2'
        if re_key in numeric_cols and im_key in numeric_cols:
             re_vals = aligned_data[re_key]
             im_vals = aligned_data[im_key]
             beta2_vals = re_vals + 1j * im_vals
             
             # Abs std
             final_df['beta2_abs_std'] = np.nanstd(np.abs(beta2_vals), axis=0)
             
             # Angle std (Circular)
             angles = np.angle(beta2_vals)
             final_df['beta2_angle_std'] = np.rad2deg(circstd(angles, axis=0))

    # Add Truth (assume truth result also matches row-wise)
    if df_truth is not None:
         if len(df_truth) == len(times):
             for col in numeric_cols:
                 if col in df_truth.columns and col != 'time':
                     final_df[f'{col}_truth'] = df_truth[col]
         else:
             # Basic Interp for truth
             t_truth = df_truth['time']
             for col in numeric_cols:
                 if col in df_truth.columns and col != 'time':
                     final_df[f'{col}_truth'] = np.interp(times, t_truth, df_truth[col])

    # Derived Quantities for Plotting
    def get_complex(df, prefix, suffix='_mean'):
        re = df.get(f're_{prefix}{suffix}', None)
        im = df.get(f'im_{prefix}{suffix}', None)
        if re is not None and im is not None:
            return re + 1j * im
        return None

    suffix = '_mean' if is_bayesian else ''
    beta2 = get_complex(final_df, 'betalp_2', suffix) # re_betalp_2_mean
    
    if beta2 is not None:
        final_df[f'beta2_abs{suffix}'] = np.abs(beta2)
        final_df[f'beta2_angle{suffix}'] = np.rad2deg(np.angle(beta2))
        # stds are already calculated in aggregation loop if Bayesian

    if df_truth is not None:
        beta2_truth = get_complex(final_df, 'betalp_2', '_truth')
        if beta2_truth is not None:
             final_df[f'beta2_abs_truth'] = np.abs(beta2_truth)
             final_df[f'beta2_angle_truth'] = np.rad2deg(np.angle(beta2_truth))

    if is_bayesian:
        r0 = final_df.get('r0_mean', None)
        sig = final_df.get('σ_mean', None)
    else:
        r0 = final_df.get('r0', None)
        sig = final_df.get('σ', None)
        
    rad2uas = 1/eh.RADPERUAS
    
    if r0 is not None:
        final_df[f'd{suffix}'] = 2 * r0 * rad2uas
        if is_bayesian:
             r0_std = final_df.get('r0_std', None)
             if r0_std is not None:
                 final_df['d_std'] = 2 * r0_std * rad2uas

        if sig is not None:
             final_df[f'w{suffix}'] = sig * 2.355 * rad2uas
             if is_bayesian:
                 sig_std = final_df.get('σ_std', None)
                 if sig_std is not None:
                     final_df['w_std'] = sig_std * 2.355 * rad2uas

    if df_truth is not None:
        r0_t = final_df.get('r0_truth', None)
        sig_t = final_df.get('σ_truth', None)
        if r0_t is not None:
            final_df['d_truth'] = 2 * r0_t * rad2uas
            if sig_t is not None:
                 final_df['w_truth'] = sig_t * 2.355 * rad2uas

            # Calculate true_D_truth
            if 'd_truth' in final_df.columns and 'w_truth' in final_df.columns:
                d_t = final_df['d_truth']
                w_t = final_df['w_truth']
                with np.errstate(divide='ignore', invalid='ignore'):
                     ratio1 = w_t / d_t
                     denom1 = 1 - (1/(4*np.log(2))) * (ratio1**2)
                     final_df['true_D_truth'] = d_t / denom1

    # Calculate true_D for recon
    suffix = '_mean' if is_bayesian else ''
    if f'd{suffix}' in final_df.columns and f'w{suffix}' in final_df.columns:
        d_val = final_df[f'd{suffix}']
        w_val = final_df[f'w{suffix}']
        
        with np.errstate(divide='ignore', invalid='ignore'):
             ratio2 = w_val / d_val
             denom2 = 1 - (1/(4*np.log(2))) * (ratio2**2)
             final_df[f'true_D{suffix}'] = d_val / denom2
             
        # Error Prop if Bayesian
        if is_bayesian and f'd_std' in final_df.columns and f'w_std' in final_df.columns:
             d_err = final_df['d_std']
             w_err = final_df['w_std']
             
             # Ported from rex.py
             ln2 = np.log(2)
             ratio = w_val / d_val
             ratio_sq = ratio**2
             common_denominator = (1 - (1 / (4 * ln2)) * ratio_sq)**2
             partial_d = (1 - (3 / (4 * ln2)) * ratio_sq) / common_denominator
             partial_w = ((1 / (2 * ln2)) * ratio) / common_denominator
             d_err_term_sq = np.square(partial_d * d_err)
             w_err_term_sq = np.square(partial_w * w_err)
             final_df['true_D_std'] = np.sqrt(d_err_term_sq + w_err_term_sq)

    # Calculate A and PA (papeak)
    # A = s_1 / 2
    # PA = ξ_1 (converted to degrees)
    
    # Check for s_1 and ξ_1 (using greek xi)
    # Note: Column names in pandas might depend on encoding, but usually match source.
    # Julia source uses ξ.
    
    # Helper for extracting/calculating metric + error + truth
    def calc_derived_metric(name, source_col, func, err_func=None):
        suffix = '_mean' if is_bayesian else ''
        
        # Recon
        col = f'{source_col}{suffix}'
        if col in final_df.columns:
            dest_name = f'{name}{suffix}'
            final_df[dest_name] = func(final_df[col])
            
            # Error
            if is_bayesian and err_func:
                 src_std = f'{source_col}_std'
                 if src_std in final_df.columns:
                     final_df[f'{name}_std'] = err_func(final_df[src_std])
                     
        # Truth
        if df_truth is not None:
            col_t = f'{source_col}_truth'
            if col_t in final_df.columns:
                final_df[f'{name}_truth'] = func(final_df[col_t])

    # Asymmetry
    # A = s_1 / 2
    calc_derived_metric('A', 's_1', lambda x: x / 2.0, lambda err: err / 2.0)
    
    # PA
    # ξ_1 is in radians.
    # We want degrees in [-180, 180]. 
    def wrap_180(x):
        return ((x + 180) % 360) - 180

    calc_derived_metric('PA', 'ξ_1', lambda x: wrap_180(np.degrees(x)), lambda err: np.degrees(err))

    # Pass Percentages 
    pass_percentages = {}
    threshold_arrays = {}
    
    if df_truth is not None:
        # Define thresholds
        abs_thresholds = {
            'd': 5.0,
            'w': 5.0,
            'evpa': 26.0
        }

        rel_thresholds = ['true_D', 'A']
        # Pol magnitudes 10%
        pol_mags = ['m_net', 'm_avg', 'beta2_abs', 'v_net']
        for pm in pol_mags:
             suffix_check = '_mean' if is_bayesian else ''
             if f'{pm}{suffix_check}' in final_df.columns:
                 rel_thresholds.append(pm)

        metrics_to_check = ['d', 'w', 'beta2_angle', 'PA']
        metrics_to_check.extend(rel_thresholds)

        # PA Threshold modulated by amplitude A
        A0 = 0.7184071604180173
        pa_threshold0 = 26.0
        if 'A_truth' in final_df.columns:
             truth_A = final_df['A_truth']
             truth_A_safe = np.where(truth_A == 0, 1e-6, truth_A)
             pa_threshold_arr = pa_threshold0 * A0 / truth_A_safe
             threshold_arrays['PA'] = pa_threshold_arr

        # beta2_angle threshold modulated by beta2_abs, floored at 26
        beta2_abs_0 = 0.05
        beta2_angle_threshold0 = 26.0
        beta2_angle_threshold_arr = None
        if 'beta2_abs_truth' in final_df.columns:
             truth_beta2_abs = final_df['beta2_abs_truth']
             truth_beta2_abs_safe = np.where(truth_beta2_abs == 0, 1e-6, truth_beta2_abs)
             beta2_angle_threshold_arr = beta2_angle_threshold0 * beta2_abs_0 / truth_beta2_abs_safe
             beta2_angle_threshold_arr = np.maximum(beta2_angle_threshold_arr, beta2_angle_threshold0)
             threshold_arrays['beta2_angle'] = beta2_angle_threshold_arr

        for metric in metrics_to_check:
            recon_col = f'{metric}_mean' if is_bayesian else metric
            truth_col = f'{metric}_truth'
            std_col = f'{metric}_std'

            if recon_col not in final_df.columns or truth_col not in final_df.columns:
                continue

            recon_val = final_df[recon_col]
            truth_val = final_df[truth_col]

            # Threshold
            if metric == 'PA':
                thres = pa_threshold_arr
            elif metric == 'beta2_angle' and beta2_angle_threshold_arr is not None:
                thres = beta2_angle_threshold_arr
            elif metric in abs_thresholds:
                thres = np.full_like(truth_val, abs_thresholds[metric])
            elif metric in rel_thresholds:
                thres = 0.10 * np.abs(truth_val)
            else:
                thres = np.zeros_like(truth_val)
            
            threshold_arrays[metric] = thres
            
            # Diff
            if metric in ['beta2_angle', 'evpa', 'PA']:
                # Angular difference
                diff = np.abs(recon_val - truth_val)
                diff = np.minimum(diff, 360 - diff)
            else:
                diff = np.abs(recon_val - truth_val)
                
            # Condition
            uncertainty = 0
            if is_bayesian and std_col in final_df.columns:
                uncertainty = final_df[std_col]
            
            pass_condition = (diff - uncertainty) <= thres
            pass_pct = np.count_nonzero(pass_condition) / len(recon_val) * 100
            pass_percentages[metric] = pass_pct
            final_df[f'pass_percent_{metric}'] = pass_pct
        
        # Save thresholds to dataframe
        for metric, thres in threshold_arrays.items():
            final_df[f'{metric}_threshold'] = thres

    out_csv = args.outpath + "_vida.csv"
    final_df.to_csv(out_csv, index=False)
    print(f"Saved aggregated results to {out_csv}")
    
    # Plotting Logic (Combined)
    times_plot = final_df['time']
    
    def plot_metric(ax, metric_base, label_prefix='', unit='', wrap=False):
        col_mean = f'{metric_base}_mean'
        col_std = f'{metric_base}_std'
        col_truth = f'{metric_base}_truth'
        
        if not is_bayesian:
             col_mean = metric_base
             col_std = None 
             
        if col_truth in final_df.columns:
             truth_label = 'Truth'
             ax.plot(times_plot, final_df[col_truth], 'k-', label=truth_label, alpha=0.8)
             
             # Shading
             if metric_base in threshold_arrays:
                 thres = threshold_arrays[metric_base]
                 base_truth = final_df[col_truth]
                 ax.fill_between(times_plot, base_truth - thres, base_truth + thres, color='black', alpha=0.15)
             
        if col_mean in final_df.columns:
            val = final_df[col_mean]
            
            label = 'Recon'
            if metric_base in pass_percentages:
                label += f" (Pass: {pass_percentages[metric_base]:.1f}%)"
            
            if is_bayesian and col_std is not None and col_std in final_df.columns:
                err = final_df[col_std]
                # Filter NaNs for plotting just in case
                ax.errorbar(times_plot, val, yerr=err, fmt='-o', label=label, capsize=3, alpha=0.7)
            else:
                 ax.plot(times_plot, val, '-o', label=label, alpha=0.7)
                 
        if unit:
            ax.set_ylabel(f'{metric_base} ({unit})')
        else:
            ax.set_ylabel(f'{metric_base}')
            
        ax.legend()

    # Combined Plot (2x4) 
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(32,8), sharex=True)
    plot_metric(ax[0,0], 'd', unit='uas')
    plot_metric(ax[0,1], 'w', unit='uas')
    plot_metric(ax[0,2], 'true_D', unit='uas') 
    plot_metric(ax[0,3], 'A')
    
    plot_metric(ax[1,0], 'm_net')
    plot_metric(ax[1,1], 'beta2_abs')
    plot_metric(ax[1,2], 'beta2_angle', unit='deg')
    plot_metric(ax[1,3], 'PA', unit='deg')

    for a in ax.flatten(): a.set_xlabel('Time (UTC)')
    plt.tight_layout()
    plt.savefig(args.outpath + '_vida.png', bbox_inches='tight', dpi=300)
    print(f"Saved plot to {args.outpath}_vida.png")
    
    # Cleanup memory
    kill_julia_process()

if __name__ == "__main__":
    main()
