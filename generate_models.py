#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_models.py — Generate synthetic model movies.

Reads syndata_config.yaml and uses src/models.py to produce model movies
for the specified model parameters. 

Each model is saved as:
    <outdir>/<modelname>.hdf5       — movie
    <outdir>/<modelname>.fits       — time-averaged frame
    <outdir>/<modelname>_params.yaml — generation parameters
    <outdir>/gifs/<modelname>.gif   — animated plots
    <outdir>/avg/<modelname>.png    — static average plot

Usage:
    python generate_models.py --config syndata_config.yaml
    python generate_models.py --config syndata_config.yaml --models mring+hsCW point
    python generate_models.py --config syndata_config.yaml --dry-run
"""

import os
import gc
import sys
import yaml
import time
import shutil
import argparse
import numpy as np
import ehtim as eh

# Add parent directory so we can import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models import (
    # Static frame builders
    make_crescent_frame, make_ring_frame, make_disk_frame,
    make_edisk_frame, make_double_frame, make_point_frame,
    # Movie builders
    make_static_movie, make_mring_hs_movie, make_mring_hs_pol_movie,
    make_mring_hs_cross_movie, make_mring_hs_incoh_movie,
    make_mring_hs_not_center_movie, make_varbeta2_movie,
    # GRMHD
    load_grmhd, compose_grmhd_hotspot,
    # Utilities
    extract_movie_window, save_model, validate_params,
)
from src.syntheticdata import load_obs_ref as _load_obs_ref

# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_obs_ref(config: dict):
    """Load the reference observation from config."""
    uvfits_dir = config['observation']['uvfits_dir']
    lo_file = config['observation']['uvfits_files']['LO']
    uvfits_path = os.path.join(uvfits_dir, lo_file)
    array_file = config['observation'].get('array_file', './arrays/EHT2017.txt')
    return _load_obs_ref(uvfits_path, array_file)

def get_base_params(config: dict) -> dict:
    """Retrieve fallback global parameters (fov, npix, seed)."""
    return {
        'fov_uas': config['image']['fov_uas'],
        'npix': config['image']['npix'],
        'besttime_start': config['time']['besttime_start'],
        'besttime_stop': config['time']['besttime_stop'],
        'linpol_frac': config['polarization']['mbreve_mean'],
        'circpol_frac': config['polarization']['vbreve_mean'],
        'seed': config.get('seeds', {}).get('model_generation', 42),
    }

# ═══════════════════════════════════════════════════════════════════════
#  Model Registry
# ═══════════════════════════════════════════════════════════════════════

def build_model_registry() -> dict:
    """Map type strings (from YAML) to builder functions.
    
    Returns:
        dict: type -> (movie_builder, frame_builder)
    """
    registry = {}

    # Static models
    registry['crescent'] = (make_static_movie, make_crescent_frame)
    registry['ring'] = (make_static_movie, make_ring_frame)
    registry['disk'] = (make_static_movie, make_disk_frame)
    registry['edisk'] = (make_static_movie, make_edisk_frame)
    registry['double'] = (make_static_movie, make_double_frame)
    registry['point'] = (make_static_movie, make_point_frame)

    # Dynamic models
    registry['mring_hs'] = (make_mring_hs_movie, None)
    registry['mring_hs_cross'] = (make_mring_hs_cross_movie, None)
    registry['mring_hs_not_center'] = (make_mring_hs_not_center_movie, None)
    registry['mring_hs_incoh'] = (make_mring_hs_incoh_movie, None)
    registry['mring_hs_pol'] = (make_mring_hs_pol_movie, None)
    registry['varbeta2'] = (make_varbeta2_movie, None)

    # GRMHD based
    registry['grmhd'] = ('grmhd', None)
    registry['grmhd_hs'] = ('grmhd+hs', None)

    return registry


# ═══════════════════════════════════════════════════════════════════════
#  Model Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_single_model(model_name: str, recipe: dict, registry: dict,
                           base_params: dict, obs_ref, outdir: str, config: dict) -> str:
    """Build a single model requested from the YAML recipe."""
    mtype = recipe.get('type')
    if not mtype or mtype not in registry:
        print(f"  [WARN] Unknown model type '{mtype}' for recipe '{model_name}'. Skipping.")
        return None

    # Merge global defaults + recipe
    params = dict(base_params)
    params.update(recipe)

    # Validate required parameters before attempting to build
    missing = validate_params(mtype, params)
    if missing:
        print(f"  [ERROR] Model '{model_name}' (type '{mtype}') is missing required params: {missing}. Skipping.")
        return None

    output_path = os.path.join(outdir, f"{model_name}.hdf5")
    builder = registry[mtype]

    # Seed RNG for reproducibility before any stochastic model construction
    seed = params.get('seed', 42)
    np.random.seed(seed)

    t0 = time.time()
    
    if builder[0] == 'grmhd':
        source_key = recipe.get('source_key', model_name)
        grmhd_path = config['models']['grmhd_sources'].get(source_key)
        if not grmhd_path:
            print(f"  [WARN] GRMHD source path not found for key: {source_key}")
            return None
            
        print(f"  Loading GRMHD: {grmhd_path}")
        mov = load_grmhd(grmhd_path, params)
        
    elif builder[0] == 'grmhd+hs':
        base_model_name = recipe.get('base_model')
        if not base_model_name:
            print("  [WARN] grmhd+hs type requires 'base_model' specified.")
            return None
            
        # Try loading generated base model first, then fall back to loading from source directly
        base_hdf5_path = os.path.join(outdir, f"{base_model_name}.hdf5")
        if os.path.exists(base_hdf5_path):
            print(f"  Loading compiled base GRMHD from: {base_hdf5_path}")
            grmhd_mov = eh.movie.load_hdf5(base_hdf5_path)
            grmhd_mov.reset_interp(bounds_error=False)
        else:
            print(f"  Loading raw base GRMHD source...")
            source_key = config['models']['recipes'].get(base_model_name, {}).get('source_key', base_model_name)
            grmhd_path = config['models']['grmhd_sources'].get(source_key)
            base_recipe = dict(base_params)
            base_recipe.update(config['models']['recipes'].get(base_model_name, {}))
            grmhd_mov = load_grmhd(grmhd_path, base_recipe)

        mov = compose_grmhd_hotspot(grmhd_mov, params, obs_ref)
        del grmhd_mov
        gc.collect()

    elif builder[1] is not None:
        # Static models: builder[0] is make_static_movie, builder[1] is the frame_func
        movie_func, frame_func = builder
        print(f"  Building static movie ({mtype})")
        mov = movie_func(frame_func, params, obs_ref)
        
    else:
        # Dynamic geometric models: builder[0] is make_xxx_movie
        movie_func, _ = builder
        print(f"  Building dynamic movie ({mtype})")
        mov = movie_func(params, obs_ref)

    elapsed = time.time() - t0
    print(f"  [TIME] Built in {elapsed:.1f}s, saving to {output_path}")

    # Remove the generic model_name if it conflicts
    params['model_name'] = model_name
    no_plots = config.get('no_plots', False)
    fps = config.get('fps', 10)
    skip_visvar = config.get('skip_visvar', True)
    save_model(mov, output_path, params, obs_ref=obs_ref, no_plots=no_plots, fps=fps, skip_visvar=skip_visvar)

    del mov
    gc.collect()

    return output_path


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def create_parser():
    p = argparse.ArgumentParser(description='Generate EHT synthetic model movies.')
    p.add_argument('--config', type=str, default='models_config.yaml',
                   help='Path to YAML configuration file')
    p.add_argument('--models', type=str, nargs='+', default=None,
                   help='Specific models to generate (default: all from config.models.generate)')
    p.add_argument('--outdir', type=str, default=None,
                   help='Override output directory (default: from config)')
    p.add_argument('--no-plots', action='store_true',
                   help='Disable generating high-quality GIF plots using visualize.py')
    p.add_argument('--fps', type=int, default=10,
                   help='FPS for generated GIFs (default: 10)')
    p.add_argument('--skip-visvar', action='store_true', default=True,
                   help='Skip generating visibility variance plot (default: True)')
    p.add_argument('--dry-run', action='store_true',
                   help='Print what would be generated without executing')
    return p


def main():
    args = create_parser().parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    
    outdir = args.outdir or config['output']['model_dir']
    os.makedirs(outdir, exist_ok=True)

    model_names_to_build = args.models or config['models'].get('generate', [])
    recipes = config['models'].get('recipes', {})

    print(f"\n{'='*60}")
    print(f"  Model Generation Pipeline")
    print(f"  Config: {args.config}")
    print(f"  Output: {outdir}")
    print(f"  Models requested: {len(model_names_to_build)}")
    print(f"{'='*60}\n")

    registry = build_model_registry()

    if args.dry_run:
        print("  DRY RUN — would generate:")
        for name in model_names_to_build:
            if name in recipes:
                print(f"    {name:30s} [Recipe OK: type={recipes[name].get('type')}]")
            else:
                print(f"    {name:30s} [ERROR] Recipe missing in config!")
        return

    # Back up config to output directory
    config_copy = os.path.join(outdir, 'syndata_config_snapshot.yaml')
    shutil.copy2(args.config, config_copy)

    print("Loading reference observation...")
    obs_ref = load_obs_ref(config)
    
    base_params = get_base_params(config)
    config['no_plots'] = args.no_plots
    config['fps'] = args.fps
    config['skip_visvar'] = args.skip_visvar

    total_t0 = time.time()
    results = []
    
    for i, model_name in enumerate(model_names_to_build, 1):
        print(f"\n[{i}/{len(model_names_to_build)}] Generating: {model_name}")
        
        recipe = recipes.get(model_name)
        if not recipe:
            print(f"  [ERROR] No recipe defined in config for model '{model_name}'. Skipping.")
            continue
            
        try:
            result_path = generate_single_model(
                model_name, recipe, registry, base_params, obs_ref, outdir, config)
        except Exception as e:
            print(f"  [ERROR] Failed to generate '{model_name}': {e}")
            result_path = None

        results.append((model_name, result_path))

    total_elapsed = time.time() - total_t0
    successes = len([r for r in results if r[1] is not None])
    print(f"\n{'='*60}")
    print(f"  [OK] Done! Generated {successes} / {len(model_names_to_build)} models in {total_elapsed:.1f}s")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
