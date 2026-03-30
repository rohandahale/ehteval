#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_data.py — Generate synthetic datasets from model movies.

Reads syndata_config.yaml and orchestrates the full data generation pipeline:
  Step 1: Generate caltable (once, reused for all models)
  Step 2: Generate uncalibrated data (per model × band)
  Step 3: Pre-imaging calibration (per model × band × scattering variant)
  Step 4: Generate ground truth movies

Usage:
    python generate_data.py --config data_config.yaml
    python generate_data.py --config data_config.yaml --step uncal
    python generate_data.py --config data_config.yaml --step preimcal --models mring+hsCW
    python generate_data.py --config data_config.yaml --step groundtruth

Author: Rohan Dahale
"""

import os
import gc
import sys
import yaml
import time
import shutil
import argparse
import numpy as np

# Add parent directory so we can import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.syntheticdata import (
    GainConfig, load_obs_ref,
    generate_caltable, make_synthetic_dataset,
    apply_preimcal, make_ground_truth,
    get_uncal_basename,
)


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    """Load and return YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# ═══════════════════════════════════════════════════════════════════════
#  Step 1: Caltable Generation
# ═══════════════════════════════════════════════════════════════════════

def step_caltable(config: dict):
    """Generate calibration table (once for all models)."""
    import ehtim as eh

    caltable_config = config['caltable']
    if not caltable_config.get('generate_new', False):
        caltable_dir = caltable_config['dir']
        if os.path.isdir(caltable_dir):
            print(f"  [OK] Using existing caltable: {caltable_dir}")
            return caltable_dir
        else:
            print(f"  [WARN] Caltable dir not found: {caltable_dir}")
            print(f"      Set caltable.generate_new: true in config to generate.")
            return None

    print("\n" + "=" * 60)
    print("  Step 1: Generating calibration table")
    print("=" * 60)

    # Load first model as reference
    model_list = config['models']['generate']
    model_dir = config['output']['model_dir']
    first_model = os.path.join(model_dir, f"{model_list[0]}.hdf5")

    if not os.path.exists(first_model):
        print(f"  [ERROR] First model not found: {first_model}")
        print(f"     Run generate_models.py first.")
        return None

    mov = eh.movie.load_hdf5(first_model)
    mov.reset_interp(bounds_error=False)

    # Load observation reference
    obs_config = config['observation']
    uvfits_path = os.path.join(obs_config['uvfits_dir'],
                                obs_config['uvfits_files']['LO'])
    obs_ref = load_obs_ref(uvfits_path,
                            obs_config.get('array_file', './arrays/EHT2017.txt'))

    # Gain configuration
    gain_config = GainConfig.from_config(config)

    # Generate
    seeds = config['seeds']
    corruption = config['corruption']
    output_dir = caltable_config['dir']

    caltable_dir = generate_caltable(
        mov, obs_ref, gain_config, output_dir,
        seed=seeds['observation_noise'],
        add_th_noise=corruption['add_thermal_noise'],
        ampcal=corruption['ampcal'],
        phasecal=corruption['phasecal'],
        rlgaincal=corruption.get('rlgaincal', True),
        dterm=corruption.get('dterm', False),
        array_file=obs_config.get('array_file', './arrays/EHT2017.txt'))

    del mov
    gc.collect()
    return caltable_dir


# ═══════════════════════════════════════════════════════════════════════
#  Step 2: Uncalibrated Data Generation
# ═══════════════════════════════════════════════════════════════════════

def step_uncal(config: dict, models: list = None):
    """Generate uncalibrated synthetic data for each model × band."""
    print("\n" + "=" * 60)
    print("  Step 2: Generating uncalibrated data")
    print("=" * 60)

    model_list = models or config['models']['generate']
    obs_config = config['observation']
    time_config = config['time']
    corruption = config['corruption']
    seeds = config['seeds']
    compute = config.get('compute', {})
    model_dir = config['output']['model_dir']
    uncal_dir = os.path.join(config['output']['data_dir'], 'uncal')
    caltable_dir = config['caltable']['dir']
    bands = obs_config.get('bands', ['LO'])

    os.makedirs(uncal_dir, exist_ok=True)

    total = len(model_list) * len(bands)
    count = 0

    for model_name in model_list:
        for band in bands:
            count += 1
            model_path = os.path.join(model_dir, f"{model_name}.hdf5")

            if not os.path.exists(model_path):
                print(f"\n  [{count}/{total}] [WARN] Model not found: {model_path}")
                continue

            print(f"\n  [{count}/{total}] {model_name} / {band}")

            uvfits_file = obs_config['uvfits_files'].get(band)
            uvfits_path = os.path.join(obs_config['uvfits_dir'], uvfits_file)
            obs_ref = load_obs_ref(uvfits_path,
                                    obs_config.get('array_file',
                                                   './arrays/EHT2017.txt'))

            make_synthetic_dataset(
                model_path=model_path,
                obs_ref=obs_ref,
                output_dir=uncal_dir,
                caltable_dir=caltable_dir,
                band=band,
                tstart=time_config['tstart'],
                tstop=time_config['tstop'],
                tshift=time_config.get('tshift', 0.0),
                apply_scat=corruption['apply_scattering'],
                rngseed=seeds['scattering_screen'],
                seed=seeds['observation_noise'],
                add_th_noise=corruption['add_thermal_noise'],
                gain=True,
                ampcal=corruption.get('ampcal', False),
                phasecal=corruption.get('phasecal', False),
                dterm=corruption.get('dterm', False),
                nproc_scatter=compute.get('nproc_scatter', 32),
                save_scattered_movie=True,
                save_avg_frame=True)

            gc.collect()


# ═══════════════════════════════════════════════════════════════════════
#  Step 3: Pre-imaging Calibration
# ═══════════════════════════════════════════════════════════════════════

def step_preimcal(config: dict, models: list = None):
    """Run pre-imaging calibration pipeline for all models."""
    print("\n" + "=" * 60)
    print("  Step 3: Pre-imaging calibration")
    print("=" * 60)

    model_list = models or config['models']['generate']
    obs_config = config['observation']
    preimcal_config = config['preimcal']
    time_config = config['time']
    data_dir = config['output']['data_dir']
    uncal_dir = os.path.join(data_dir, 'uncal')
    bands = obs_config.get('bands', ['LO'])
    tint_options = preimcal_config.get('tint_options', [60])
    variants = preimcal_config.get('variants', {})

    # Time window configurations
    time_windows = {
        'fulltime': {
            'tstart': time_config['tstart'],
            'tstop': time_config['tstop'],
            'do_timeflag': False,
        },
        'besttime': {
            'tstart': time_config['besttime_start'],
            'tstop': time_config['besttime_stop'],
            'do_timeflag': True,
        },
    }

    corruption = config.get('corruption', {})

    for model_name in model_list:
        for band in bands:
            basename = get_uncal_basename(
                model_name, band, 
                corruption.get('apply_scattering', True),
                corruption.get('ampcal', False),
                corruption.get('phasecal', False),
                corruption.get('dterm', False),
                corruption.get('add_thermal_noise', True)
            )
            
            uncal_path = os.path.join(uncal_dir, f"{basename}.uvfits")
            scat_movie = os.path.join(uncal_dir, f"{basename}.hdf5")

            if not os.path.exists(uncal_path):
                print(f"  [WARN] Uncal data not found: {uncal_path}")
                continue

            for tw_name, tw_params in time_windows.items():
                for tint in tint_options:
                    for var_name, var_params in variants.items():
                        out_dir = os.path.join(
                            data_dir, tw_name, 'data',
                            f"netcal+tavg{tint}s")
                        netcal_dir = os.path.join(
                            out_dir,
                            f"{model_name}_{band}_{var_name}_netcal_caltable")

                        params = {
                            'do_static_netcal': preimcal_config.get(
                                'do_static_netcal', False),
                            'do_lc_netcal': preimcal_config.get(
                                'do_lc_netcal', True),
                            'zbl': preimcal_config.get('zbl', 2.7),
                            'netcal_caltable_dir': netcal_dir,
                            'is_normalized': False,
                            'is_deblurred': False,
                            'nproc': 32,
                            'do_LMTcal': preimcal_config.get(
                                'do_LMTcal', False),
                            'LMTcal_fwhm': preimcal_config.get(
                                'LMTcal_fwhm', 60.0),
                            'do_JCMTcal': preimcal_config.get(
                                'do_JCMTcal', False),
                            'tint': tint,
                            'do_mergebands': False,
                            'syserr': preimcal_config.get('syserr', -1),
                            'ref_optype': var_params.get('ref_optype', None),
                            'ref_scale': preimcal_config.get(
                                'ref_scale', 1.0),
                            'do_deblurr': var_params.get(
                                'do_deblurr', False),
                            'do_psd_noise': preimcal_config.get(
                                'do_psd_noise', False),
                            'do_timeflag': tw_params['do_timeflag'],
                            'tstart': tw_params['tstart'],
                            'tstop': tw_params['tstop'],
                        }

                        print(f"\n  [preimcal] {model_name}/{band} "
                              f"→ {tw_name}/netcal+tavg{tint}s/{var_name}")

                        try:
                            apply_preimcal(
                                uvfits_path=uncal_path,
                                output_dir=out_dir,
                                scattered_movie_path=scat_movie if os.path.exists(scat_movie) else None,
                                preimcal_params=params,
                                netcal_caltable_dir=netcal_dir)
                        except Exception as e:
                            print(f"  [ERROR] Error: {e}")

                        gc.collect()


# ═══════════════════════════════════════════════════════════════════════
#  Step 4: Ground Truth Generation
# ═══════════════════════════════════════════════════════════════════════

def step_groundtruth(config: dict, models: list = None):
    """Generate ground truth movies for each model."""
    print("\n" + "=" * 60)
    print("  Step 4: Ground truth generation")
    print("=" * 60)

    model_list = models or config['models']['generate']
    obs_config = config['observation']
    preimcal_config = config['preimcal']
    data_dir = config['output']['data_dir']
    uncal_dir = os.path.join(data_dir, 'uncal')
    bands = obs_config.get('bands', ['LO'])
    variants = preimcal_config.get('variants', {})

    time_windows = ['fulltime', 'besttime']

    corruption = config.get('corruption', {})

    for model_name in model_list:
        for band in bands:
            basename = get_uncal_basename(
                model_name, band, 
                corruption.get('apply_scattering', True),
                corruption.get('ampcal', False),
                corruption.get('phasecal', False),
                corruption.get('dterm', False),
                corruption.get('add_thermal_noise', True)
            )
            
            scat_movie = os.path.join(uncal_dir, f"{basename}.hdf5")
            if not os.path.exists(scat_movie):
                print(f"  [WARN] Scattered movie not found: {scat_movie}")
                continue

            for tw_name in time_windows:
                truth_dir = os.path.join(data_dir, tw_name, 'groundtruth')
                print(f"\n  [groundtruth] {model_name}/{band} → {tw_name}")

                try:
                    make_ground_truth(
                        scattered_movie_path=scat_movie,
                        output_dir=truth_dir,
                        model_name=model_name,
                        band=band,
                        preimcal_params={'variants': variants})
                except Exception as e:
                    print(f"  [ERROR] Error: {e}")

                gc.collect()


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def create_parser():
    p = argparse.ArgumentParser(
        description='Generate synthetic EHT datasets.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  caltable    - Generate calibration table (once)
  uncal       - Generate uncalibrated synthetic data
  preimcal    - Run pre-imaging calibration
  groundtruth - Generate ground truth movies
  all         - Run all steps in sequence

Examples:
  python generate_data.py --config data_config.yaml
  python generate_data.py --config data_config.yaml --step uncal
  python generate_data.py --config data_config.yaml --step preimcal --models mring+hsCW
        """)
    p.add_argument('--config', type=str, default='data_config.yaml',
                   help='Path to YAML configuration file')
    p.add_argument('--step', type=str, default='all',
                   choices=['caltable', 'uncal', 'preimcal', 'groundtruth', 'all'],
                   help='Pipeline step to run')
    p.add_argument('--models', type=str, nargs='+', default=None,
                   help='Specific models to process (default: all from config)')
    p.add_argument('--bands', type=str, nargs='+', default=None,
                   help='Override bands (default: from config)')
    return p


def main():
    args = create_parser().parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    # Override bands if specified
    if args.bands:
        config['observation']['bands'] = args.bands

    print(f"\n{'='*60}")
    print(f"  Synthetic Data Generation Pipeline")
    print(f"  Config: {args.config}")
    print(f"  Step:   {args.step}")
    if args.models:
        print(f"  Models: {args.models}")
    print(f"{'='*60}")

    # Copy config to output directory
    data_dir = config['output']['data_dir']
    os.makedirs(data_dir, exist_ok=True)
    config_copy = os.path.join(data_dir, 'data_config.yaml')
    shutil.copy2(args.config, config_copy)

    total_t0 = time.time()

    if args.step in ('caltable', 'all'):
        step_caltable(config)

    if args.step in ('uncal', 'all'):
        step_uncal(config, args.models)

    if args.step in ('preimcal', 'all'):
        step_preimcal(config, args.models)

    if args.step in ('groundtruth', 'all'):
        step_groundtruth(config, args.models)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"  [OK] Pipeline complete in {total_elapsed:.1f}s")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
