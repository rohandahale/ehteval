import os
import re
import sys
import yaml
import argparse
import subprocess
import shutil
import time
import gc
from glob import glob

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_command(cmd, description):
    print(f"\n--- Running {description} ---")
    print("Command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"{description} completed successfully.")
        # Cleanup
        gc.collect()
        time.sleep(5)
    except subprocess.CalledProcessError as e:
        print(f"{description} failed with error code {e.returncode}.")

def main():
    parser = argparse.ArgumentParser(description="Driver script for Sgr A* Dynamics Evaluation")
    parser.add_argument('config', nargs='?', default='params.yml', help='Path to params.yml configuration file')
    args = parser.parse_args()

    # Load Configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found.")
        sys.exit(1)
        
    config = load_config(args.config)
    
    # Setup Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, 'src')
    
    submission_dir = config['submission_dir']
    results_dir = config['results_dir']
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Copy config file to results directory with versioning
    existing_params = glob(os.path.join(results_dir, 'params_v*.yml'))
    max_version = 0
    for p in existing_params:
        match = re.search(r'params_v(\d+)\.yml', os.path.basename(p))
        if match:
            max_version = max(max_version, int(match.group(1)))
    
    next_version = max_version + 1
    new_params_path = os.path.join(results_dir, f'params_v{next_version}.yml')
    shutil.copy(args.config, new_params_path)
    print(f"Copied configuration to {new_params_path}")
        
    # Parameters (Single Choice where applicable)
    models = config['models']
    data_band = config['data_band']
    scattering = config['scattering']
    recon_band = config['recon_band']
    pipeline = config['pipeline']
    is_bayesian = config['is_bayesian']
    
    ncores = str(config['ncores'])
    
    # Optional Time Range
    tstart = config.get('tstart') 
    tstop = config.get('tstop')
    noise = config.get('noise')
    
    # Overwrite Flag
    overwrite = config.get('overwrite', False)

    # Iterate over Models only (other params are fixed for this run)
    for model in models:
        print(f"\n{'='*80}")
        print(f"Processing: Model={model}, Pipeline={pipeline} (Bayesian={is_bayesian})")
        print(f"Params: Data Band={data_band}, Recon Band={recon_band}, Scattering={scattering}")
        print(f"{'='*80}")
        
        # 1. Define Output Directory
        # "Outpath is <results_dir>/<model>/<model>_<pipeline>"
        
        # Determine if we should use truth based on model
        sgra_variants = ["SGRA", "sgra", "Sgra", "SgrA"]
        # Use truth only if model is NOT one of the SGRA variants
        use_truth_for_model = model not in sgra_variants
        
        out_prefix = os.path.join(results_dir, f"{model}_{pipeline}")
        
        # 2. File Paths construction
        
        # Data File
        data_fname = config['data_format'].format(
            model=model, 
            data_band=data_band, 
            scattering=scattering
        )
        data_path = os.path.join(submission_dir, data_fname)
        
        # Truth File
        truth_fname = config['truth_format'].format(
            model=model, 
            data_band=data_band, 
            scattering=scattering
        )
        truth_path = os.path.join(submission_dir, truth_fname)
        
        # Reconstruction Files
        input_arg = None
        visualize_input = None
        
        if is_bayesian:
            # Bayesian Mode
            recon_fname_bayes_glob = config['recon_format_bayesian'].format(
                model=model, 
                recon_band=recon_band, 
                scattering=scattering, 
                pipeline=pipeline
            )
            recon_path_bayes_glob = os.path.join(submission_dir, recon_fname_bayes_glob)
            
            # Check for matches
            matched_files = glob(recon_path_bayes_glob)
            if not matched_files:
                print(f"Skipping: No Bayesian input files found matching: {recon_path_bayes_glob}")
                continue
                
            input_arg = matched_files
            print(f"Found {len(matched_files)} Bayesian input files.")
            
            # For visualize, we need the mean file
            recon_fname_bayes_mean = config['recon_format_bayesian_mean'].format(
                model=model, 
                recon_band=recon_band, 
                scattering=scattering, 
                pipeline=pipeline
            )
            recon_path_bayes_mean = os.path.join(submission_dir, recon_fname_bayes_mean)
            
            if os.path.exists(recon_path_bayes_mean):
                visualize_input = recon_path_bayes_mean
            else:
                print(f"Warning: Mean file for Bayesian reconstruction not found: {recon_path_bayes_mean}")
                visualize_input = None
                
        else:
            # Non-Bayesian Mode
            recon_fname_non_bayes = config['recon_format_non_bayesian'].format(
                model=model, 
                recon_band=recon_band, 
                scattering=scattering, 
                pipeline=pipeline
            )
            recon_path_non_bayes = os.path.join(submission_dir, recon_fname_non_bayes)
            
            if not os.path.exists(recon_path_non_bayes):
                print(f"Skipping: Non-Bayesian input file not found: {recon_path_non_bayes}")
                continue
                
            input_arg = [recon_path_non_bayes]
            visualize_input = recon_path_non_bayes # Visualize uses the single recon file
            print(f"Found Non-Bayesian input: {recon_path_non_bayes}")

        if not os.path.exists(data_path):
             print(f"Skipping: Data file not found: {data_path}")
             continue
             
        # 3. Execution
        
        # Helper to build basic command
        def build_cmd(script_name, input_val, output_val, use_truth=True, extra_args=[]):
            script_path = os.path.join(src_dir, script_name)
            cmd = ['python', script_path, 
                   '-d', data_path, 
                   '-n', ncores]
            
            if script_name in ['visualize.py', 'patternspeed.py']:
                flag = '-i'
            else:
                flag = '--input'
            
            cmd.append(flag)
            if isinstance(input_val, list):
                cmd.extend(input_val)
            else:
                cmd.append(input_val)
                
            cmd.extend(['-o', output_val])
            
            if use_truth:
                cmd.extend(['--truthmv', truth_path])
                
            if tstart is not None:
                cmd.extend(['--tstart', str(tstart)])
            if tstop is not None:
                cmd.extend(['--tstop', str(tstop)])
                
            cmd.extend(extra_args)
            return cmd

        # a) Preprocess HDF5
        if config['run_steps']['preprocess_hdf5']:
            preprocess_inputs = []
            
            # Use the visualization input (Mean file in Bayesian mode, single recon in others)
            if visualize_input and os.path.exists(visualize_input):
                preprocess_inputs.append(visualize_input)
            
            # Add Truth if applicable and exists
            if use_truth_for_model and os.path.exists(truth_path):
                preprocess_inputs.append(truth_path)

            cmd = ['python', os.path.join(src_dir, 'preprocess_hdf5.py'),
                   '-d', data_path,
                   '--input'] + preprocess_inputs + [
                   '-o', out_prefix,
                   '-n', ncores]
            if tstart is not None: cmd.extend(['--tstart', str(tstart)])
            if tstop is not None: cmd.extend(['--tstop', str(tstop)])
            run_command(cmd, "Preprocess HDF5")

        # b) Chisq
        if config['run_steps']['chisq']:
            if not overwrite and os.path.exists(f"{out_prefix}_chisq.csv"):
                print(f"Skipping Chi-Squared: Output {out_prefix}_chisq.csv already exists.")
            else:
                cmd = build_cmd('chisq.py', input_arg, out_prefix, use_truth=False) 
                run_command(cmd, "Chi-Squared")

        # c) Hotspot
        if config['run_steps']['hotspot']:
            if not overwrite and os.path.exists(f"{out_prefix}_hotspot.csv"):
                print(f"Skipping Hotspot: Output {out_prefix}_hotspot.csv already exists.")
            else:
                cmd = build_cmd('hotspot.py', input_arg, out_prefix, use_truth=use_truth_for_model)
                run_command(cmd, "Hotspot Feature Extraction")

        # d) Nxcorr
        if config['run_steps']['nxcorr']:
            if not use_truth_for_model:
                print("Skipping Nxcorr: Truth movie not available/required for this model.")
            elif not overwrite and os.path.exists(f"{out_prefix}_total_nxcorr.csv"):
                print(f"Skipping Nxcorr: Output {out_prefix}_total_nxcorr.csv already exists.")
            else:
                cmd = build_cmd('nxcorr.py', input_arg, out_prefix, use_truth=use_truth_for_model)
                run_command(cmd, "Nxcorr Analysis")

        # e) Pattern Speed
        if config['run_steps']['patternspeed']:
            if not overwrite and os.path.exists(f"{out_prefix}_patternspeed_summary.png"):
                print(f"Skipping Pattern Speed: Output {out_prefix}_patternspeed_summary.png already exists.")
            else:
                cmd = build_cmd('patternspeed.py', input_arg, out_prefix, use_truth=use_truth_for_model)
                run_command(cmd, "Pattern Speed")

        # e2) Pattern Speed v2
        if config['run_steps'].get('patternspeed_v2', False):
            # Create subfolder inside results directory
            v2_dir = os.path.join(results_dir, "patternspeed_v2")
            if not os.path.exists(v2_dir):
                os.makedirs(v2_dir)
            
            v2_prefix = os.path.join(v2_dir, f"{model}_{pipeline}")
            
            if not overwrite and os.path.exists(f"{v2_prefix}_patternspeed_summary.png"):
                print(f"Skipping Pattern Speed v2: Output {v2_prefix}_patternspeed_summary.png already exists.")
            else:
                cmd = build_cmd('patternspeed_v2.py', input_arg, v2_prefix, use_truth=use_truth_for_model)
                run_command(cmd, "Pattern Speed v2")
            
        # f) Rex
        if config['run_steps']['rex']:
             if not overwrite and os.path.exists(f"{out_prefix}_rex.csv"):
                 print(f"Skipping REx: Output {out_prefix}_rex.csv already exists.")
             else:
                 cmd = build_cmd('rex.py', input_arg, out_prefix, use_truth=use_truth_for_model)
                 run_command(cmd, "Ring Extraction (REx)")
             
        # f2) Vida Pol
        if config['run_steps'].get('vida_pol', False):
            if not overwrite and os.path.exists(f"{out_prefix}_vida.csv"):
                print(f"Skipping VIDA Pol: Output {out_prefix}_vida.csv already exists.")
            else:
                vida_blur = config.get('vida_pol_blur', 0.0)
                vida_extra_args = ['--blur-truth', str(vida_blur)]
                cmd = build_cmd('vida_pol.py', input_arg, out_prefix, use_truth=use_truth_for_model, extra_args=vida_extra_args)
                run_command(cmd, "VIDA Polarization Extraction")

             
        # g) Visualize
        if config['run_steps']['vizualize']:
            total_gif_exists = os.path.exists(f"{out_prefix}_total.gif")
            lp_gif_exists = os.path.exists(f"{out_prefix}_lp.gif")
            visvar_exists = os.path.exists(f"{out_prefix}_visvar.png")
            
            if not overwrite and total_gif_exists and lp_gif_exists and visvar_exists:
                print(f"Skipping Visualization: All outputs already exist.")
            else:
                if visualize_input:
                    # Build extra args to skip already existing outputs
                    viz_extra_args = []
                    if not overwrite:
                        if total_gif_exists:
                            viz_extra_args.append('--skip-total')
                            print(f"  Will skip _total.gif (already exists)")
                        if lp_gif_exists:
                            viz_extra_args.append('--skip-lp')
                            print(f"  Will skip _lp.gif (already exists)")
                        if visvar_exists:
                            viz_extra_args.append('--skip-visvar')
                            print(f"  Will skip _visvar.png (already exists)")
                    
                    cmd = build_cmd('visualize.py', visualize_input, out_prefix, use_truth=use_truth_for_model, extra_args=viz_extra_args)
                    run_command(cmd, "Visualization")
                else:
                    print("Skipping Visualization: No valid input file (Mean file missing for Bayesian?).")

if __name__ == "__main__":
    main()
