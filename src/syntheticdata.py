#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
syntheticdata.py — Core synthetic data generation engine for EHT observations.

This module provides the generalized pipeline for:
  1. Scattering: diffractive blur + refractive screen
  2. Caltable generation / loading
  3. Observation simulation (observe_same with calibration tables)
  4. Pre-imaging calibration integration
  5. Ground truth generation

All corruption parameters are explicit function arguments.
Fixed seeds are propagated to every stochastic step.
Memory efficient: designed for one-model-at-a-time processing.

Author: Rohan Dahale
"""

import os
import gc
import shutil
import random
import numpy as np
import ehtim as eh
import ehtim.scattering as so
import scipy.interpolate as interp
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════
#  Gain / D-term Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GainConfig:
    """Station-based gain amplitude offsets and uncertainties."""
    gain_offset: Dict[str, float] = field(default_factory=lambda: {
        'AA': 0.0, 'AP': 0.0, 'AZ': 0.0, 'JC': 0.0,
        'LM': 0.0, 'SM': 0.0, 'SP': 0.0, 'SR': 0.0})
    gain_sigma: Dict[str, float] = field(default_factory=lambda: {
        'AA': 0.016, 'AP': 0.072, 'AZ': 0.059, 'JC': 0.047,
        'LM': 0.108, 'SM': 0.008, 'SP': 0.054, 'SR': 0.0})
    dterm_offset: Dict[str, float] = field(default_factory=lambda: {
        'AA': 0.005, 'AP': 0.005, 'AZ': 0.01, 'LM': 0.01,
        'PV': 0.01, 'SM': 0.005, 'JC': 0.01, 'SP': 0.01, 'SR': 0.01})

    @classmethod
    def from_config(cls, config: dict) -> 'GainConfig':
        """Create GainConfig from the 'gains' section of config yaml."""
        gains = config.get('gains', {})
        return cls(
            gain_offset=gains.get('offset', {}),
            gain_sigma=gains.get('sigma', {}),
            dterm_offset=gains.get('dterm_offset', {}))


# ═══════════════════════════════════════════════════════════════════════
#  Observation Reference Loading
# ═══════════════════════════════════════════════════════════════════════

def load_obs_ref(uvfits_path: str,
                 array_file: str = './arrays/EHT2017.txt') -> 'eh.obsdata.Obsdata':
    """Load observation reference and copy correct mount types.

    Args:
        uvfits_path: path to reference uvfits file.
        array_file: path to EHT array description file.

    Returns:
        ehtim.Obsdata with correct telescope metadata.
    """
    obs = eh.obsdata.load_uvfits(uvfits_path, remove_nan=True)
    eht = eh.array.load_txt(array_file)

    t_obs = list(obs.tarr['site'])
    t_eht = list(eht.tarr['site'])
    t_conv = {'AA': 'ALMA', 'AP': 'APEX', 'SM': 'SMA', 'JC': 'JCMT',
              'AZ': 'SMT', 'LM': 'LMT', 'PV': 'PV', 'SP': 'SPT'}

    for t in t_conv:
        if t in obs.tarr['site']:
            for key in ['fr_par', 'fr_elev', 'fr_off', 'dl', 'dr']:
                obs.tarr[key][t_obs.index(t)] = eht.tarr[key][t_eht.index(t_conv[t])]

    return obs


# ═══════════════════════════════════════════════════════════════════════
#  Scattering
# ═══════════════════════════════════════════════════════════════════════

def apply_scattering(mov, rngseed: int = 1,
                     nproc: int = 32) -> 'eh.movie.Movie':
    """Apply diffractive blur + refractive screen to a movie.

    Args:
        mov: input ehtim.Movie.
        rngseed: seed for the epsilon screen.
        nproc: number of processes for scattering computation.

    Returns:
        Scattered ehtim.Movie.
    """
    eps = so.MakeEpsilonScreen(mov.xdim, mov.ydim, rngseed=rngseed)
    scattering_model = so.ScatteringModel()
    mov_scat = scattering_model.Scatter_Movie(mov, eps, processes=nproc)
    mov_scat.reset_interp(bounds_error=False)
    return mov_scat


# ═══════════════════════════════════════════════════════════════════════
#  Caltable Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_caltable(mov, obs_ref,
                      gain_config: GainConfig,
                      output_dir: str,
                      seed: int = 1,
                      add_th_noise: bool = True,
                      ampcal: bool = False,
                      phasecal: bool = False,
                      rlgaincal: bool = True,
                      dterm: bool = False,
                      array_file: str = './arrays/EHT2017.txt') -> str:
    """Generate and save calibration table.

    This should be run ONCE and reused for all models.
    Uses observe_same to create a caltable, then saves it.

    Args:
        mov: reference movie (first model in the set).
        obs_ref: reference observation.
        gain_config: station-based gain configuration.
        output_dir: directory to save caltable.
        seed: RNG seed for noise/gain generation.
        add_th_noise: whether to add thermal noise.
        ampcal: if True, amplitude is perfectly calibrated.
        phasecal: if True, phase is perfectly calibrated.
        rlgaincal: if True, R/L gains are equal.
        dterm: if True, add polarization leakage.
        array_file: path to EHT array file.

    Returns:
        Path to saved caltable directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Prepare movie metadata
    mov.rf = obs_ref.rf
    mov.ra = obs_ref.ra
    mov.dec = obs_ref.dec
    mov.mjd = obs_ref.mjd

    # Resample movie to observation time grid
    start_time = obs_ref.tstart
    end_time = obs_ref.tstop
    dtime = np.median(np.diff(mov.times))
    nt = int((end_time - start_time) // dtime)
    times = np.linspace(start_time, end_time, nt)

    im_list = []
    for it in range(nt):
        im = mov.get_image(times[it])
        im.time = times[it]
        im_list.append(im)
    mov_resampled = eh.movie.merge_im_list(im_list)

    # Configure corruption
    dcal = not dterm
    frcal = True
    dterm_offset = gain_config.dterm_offset if dterm else 0

    # Generate observation with full corruption → saves caltable
    temp_dir = output_dir + f"/seed{seed:04d}_{random.randint(0, int(1e8)):08d}"

    obs = mov_resampled.observe_same(
        obs_ref, ttype='nfft',
        add_th_noise=add_th_noise,
        ampcal=ampcal, phasecal=phasecal,
        stabilize_scan_phase=True, stabilize_scan_amp=True,
        gain_offset=gain_config.gain_offset,
        gainp=gain_config.gain_sigma,
        jones=True, inv_jones=False,
        dcal=dcal, frcal=frcal, rlgaincal=rlgaincal,
        neggains=False,
        dterm_offset=dterm_offset,
        caltable_path=temp_dir,
        seed=seed, sigmat=0.25)

    # Rename temporary caltable
    caltable_src = temp_dir + "_simdata_caltable"
    caltable_dst = output_dir
    if os.path.isdir(caltable_src):
        # Copy contents
        for item in os.listdir(caltable_src):
            s = os.path.join(caltable_src, item)
            d = os.path.join(caltable_dst, item)
            if os.path.isfile(s):
                shutil.copy2(s, d)
        shutil.rmtree(caltable_src)

    # Clean up temp
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Plot gains for diagnostic
    try:
        ct = eh.caltable.load_caltable(obs, caltable_dst + "/")
        ct.plot_gains(list(np.sort(list(ct.data.keys()))),
                      yscale='linear',
                      export_pdf=os.path.join(caltable_dst, "gains.pdf"),
                      show=False, rangex=[7, 15], rangey=[0.9, 1.2])
        plt.close()
        ct.plot_dterms(list(np.sort(list(ct.data.keys()))),
                       export_pdf=os.path.join(caltable_dst, "dterms.pdf"),
                       show=False)
        plt.close()
    except Exception as e:
        print(f"  Warning: could not plot gains/dterms: {e}")

    print(f"  [OK] Caltable saved to: {caltable_dst}")
    return caltable_dst


# ═══════════════════════════════════════════════════════════════════════
#  Observation Simulation
# ═══════════════════════════════════════════════════════════════════════

def simulate_observation(mov, obs_ref,
                         caltable_dir: str,
                         seed: int = 1,
                         add_th_noise: bool = True,
                         gain: bool = True,
                         dterm: bool = False) -> 'eh.obsdata.Obsdata':
    """Generate synthetic observation using a saved caltable.

    Corresponds to generate_observations_caldir in the old code.

    Args:
        mov: model movie.
        obs_ref: reference observation.
        caltable_dir: path to calibration table directory.
        seed: RNG seed.
        add_th_noise: add thermal noise.
        gain: apply gain corruption from caltable.
        dterm: apply D-term leakage.

    Returns:
        ehtim.Obsdata with synthetic visibilities.
    """
    frcal = True
    dcal = not dterm

    # Sync movie metadata with observation
    mov.rf = obs_ref.rf
    mov.ra = obs_ref.ra
    mov.dec = obs_ref.dec
    mov.mjd = obs_ref.mjd

    # Resample movie to observation time grid
    start_time = obs_ref.tstart
    end_time = obs_ref.tstop
    dtime = np.median(np.diff(mov.times))
    nt = int((end_time - start_time) // dtime)
    times = np.linspace(start_time, end_time, nt)

    im_list = []
    for it in range(nt):
        im = mov.get_image(times[it])
        im.time = times[it]
        im_list.append(im)
    mov_resampled = eh.movie.merge_im_list(im_list)

    # Generate noise-free observation
    obs_nonoise = mov_resampled.observe_same(obs_ref, add_th_noise=False)
    obs_nonoise.source = 'SGRA'

    # Load and apply caltable
    ct = eh.caltable.load_caltable(obs_nonoise, caltable_dir + "/")
    obs_nonoise.tarr = ct.tarr

    # Add D-term leakage
    if dterm:
        obsdata_dterms = eh.observing.obs_simulate.add_jones_and_noise(
            obs_nonoise, add_th_noise=False, ampcal=True,
            phasecal=True, dcal=dcal, frcal=frcal,
            dterm_offset=0., seed=seed)
        obs_dterms = obs_nonoise.copy()
        obs_dterms.data = obsdata_dterms
    else:
        obs_dterms = obs_nonoise.copy()

    # Apply complex gains
    if gain:
        obs_gains = ct.applycal(obs_dterms)
    else:
        obs_gains = obs_dterms

    # Add thermal noise
    if add_th_noise:
        obsdata = eh.observing.obs_simulate.add_jones_and_noise(
            obs_gains, add_th_noise=True, ampcal=True,
            phasecal=True, dcal=True, frcal=True)
        obs_final = obs_gains.copy()
        obs_final.data = obsdata
    else:
        obs_final = obs_gains.copy()

    return obs_final


def get_uncal_basename(model_name: str, band: str, apply_scat: bool, 
                       ampcal: bool, phasecal: bool, 
                       dterm: bool, add_th_noise: bool) -> str:
    """Consolidated logic for uncalibrated synthetic data naming."""
    flags = []
    if apply_scat: flags.append('scat')
    if not ampcal: flags.append('gA')
    if not phasecal: flags.append('gP')
    if dterm: flags.append('dt')
    if add_th_noise: flags.append('th')
    
    flag_str = "_" + "_".join(flags) if flags else ""
    return f"{model_name}_{band}{flag_str}"


def get_cal_basename(uncal_basename: str, params: dict) -> str:
    """Consolidated logic for pre-imaging calibration data naming."""
    flags = []
    if params.get('do_static_netcal', False) or params.get('do_lc_netcal', False):
        flags.append('netcal')
    if params.get('do_LMTcal', False):
        flags.append('LMTcal')
    if params.get('do_JCMTcal', False):
        flags.append('JCMTcal')
    if 'tint' in params:
        flags.append(f"tavg{int(params['tint'])}")
    if params.get('do_mergebands', False):
        flags.append('merge')
    if params.get('syserr', -1) > 0:
        flags.append('syserr')
    if params.get('ref_optype', None) is not None:
        flags.append(f"ref{params['ref_optype']}")
    if params.get('do_psd_noise', False):
        flags.append('psd')
    
    do_deblurr = params.get('do_deblurr', False)
    ref_optype = params.get('ref_optype', None)
    if not do_deblurr and ref_optype is None:
        flags.append("onsky")
    elif do_deblurr and ref_optype is None:
        flags.append("deblur")
    elif do_deblurr and ref_optype is not None:
        flags.append("dsct")

    flag_str = "_" + "_".join(flags) if flags else "_cal"
    # replace any trailing _onsky / _scat if they compound weirdly? 
    # Simply append to the uncal basename:
    return f"{uncal_basename}{flag_str}"


# ═══════════════════════════════════════════════════════════════════════
#  Full Pipeline
# ═══════════════════════════════════════════════════════════════════════

def make_synthetic_dataset(
        model_path: str,
        obs_ref,
        output_dir: str,
        caltable_dir: str,
        band: str = 'LO',
        tstart: float = 9.0,
        tstop: float = 15.0,
        tshift: float = 0.0,
        apply_scat: bool = True,
        rngseed: int = 1,
        seed: int = 1,
        add_th_noise: bool = True,
        gain: bool = True,
        ampcal: bool = False,
        phasecal: bool = False,
        dterm: bool = False,
        nproc_scatter: int = 32,
        save_scattered_movie: bool = True,
        save_avg_frame: bool = True,
        rerun: bool = False) -> str:
    """End-to-end: load model → extract window → scatter → simulate obs → save.

    Args:
        model_path: path to model HDF5 movie.
        obs_ref: reference observation.
        output_dir: directory for output files.
        caltable_dir: path to calibration table.
        band: frequency band label ('LO' or 'HI').
        tstart, tstop: UT hours for time window.
        tshift: time offset for source movie.
        apply_scat: whether to apply scattering.
        rngseed: seed for scattering screen.
        seed: seed for observation noise.
        add_th_noise: add thermal noise.
        gain: apply gains from caltable.
        ampcal: if False, apply amplitude corruptions from caltable.
        phasecal: if False, apply phase corruptions from caltable.
        dterm: apply D-term leakage.
        nproc_scatter: processes for scattering.
        save_scattered_movie: save the scattered movie HDF5.
        save_avg_frame: save the average frame FITS.
        rerun: if False, skip if output exists.

    Returns:
        Path to output uvfits file.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_name = os.path.basename(model_path).replace('.hdf5', '')
    
    # If gain=False, then effectively NO gains are applied (ampcal=True, phasecal=True means perfect)
    eff_ampcal = ampcal if gain else True
    eff_phasecal = phasecal if gain else True
    
    basename = get_uncal_basename(model_name, band, apply_scat, eff_ampcal, eff_phasecal, dterm, add_th_noise)
    output_uvfits = os.path.join(output_dir, f"{basename}.uvfits")

    if os.path.exists(output_uvfits) and not rerun:
        print(f"  [SKIP] Output exists, skipping: {output_uvfits}")
        return output_uvfits

    # Load model movie
    print(f"  Loading model: {model_path}")
    from src.models import extract_movie_window
    mov_org = eh.movie.load_hdf5(model_path)
    mov_org.reset_interp(bounds_error=False)

    # Extract time window
    print("  Extracting time window")
    mov = extract_movie_window(mov_org, tstart, tstop, tshift, obs_ref)
    mov.reset_interp(bounds_error=False)
    del mov_org
    gc.collect()

    # Apply scattering
    if apply_scat:
        print("  Applying scattering")
        mov_scat = apply_scattering(mov, rngseed=rngseed, nproc=nproc_scatter)
    else:
        mov_scat = mov
    del mov
    gc.collect()

    # Save scattered movie and avg frame
    if save_scattered_movie:
        print(f"  Saving scattered movie & avg to {output_dir}")
        mov_scat.save_hdf5(os.path.join(output_dir, f"{basename}.hdf5"))
        if save_avg_frame:
            im = mov_scat.avg_frame()
            im.save_fits(os.path.join(output_dir, f"{basename}.fits"))

    # Simulate observation
    print("  Simulating observation")
    obs = simulate_observation(
        mov_scat, obs_ref, caltable_dir,
        seed=seed, add_th_noise=add_th_noise,
        gain=gain, dterm=dterm)

    obs.add_scans()
    obs.save_uvfits(output_uvfits)
    print(f"  [OK] Saved: {output_uvfits}")

    del mov_scat, obs
    gc.collect()

    return output_uvfits


# ═══════════════════════════════════════════════════════════════════════
#  Pre-imaging Calibration Integration
# ═══════════════════════════════════════════════════════════════════════

def apply_preimcal(uvfits_path: str,
                   output_dir: str,
                   scattered_movie_path: str = None,
                   preimcal_params: dict = None,
                   netcal_caltable_dir: str = None) -> str:
    """Apply pre-imaging calibration pipeline to a synthetic uvfits file.

    Delegates to the preimcal module. The order of operations follows
    the established preimcal pipeline.

    Args:
        uvfits_path: path to uncalibrated uvfits.
        output_dir: output directory.
        scattered_movie_path: path to scattered movie HDF5 (for lightcurve).
        preimcal_params: dict of preimcal parameters.
        netcal_caltable_dir: directory for netcal caltable output.

    Returns:
        Path to calibrated uvfits.
    """
    # Import preimcal from src/preimcal.py (same directory as this module)
    from src import preimcal

    params = dict(preimcal_params) if preimcal_params else {}
    os.makedirs(output_dir, exist_ok=True)

    obs = eh.obsdata.load_uvfits(uvfits_path)

    # Build lightcurve from scattered movie if available
    if scattered_movie_path and params.get('do_lc_netcal', False):
        mv = eh.movie.load_hdf5(scattered_movie_path)
        lc_time = np.array(mv.times)
        lc_flux = np.array([im.total_flux() for im in mv.im_list()])
        params['lc_function'] = interp.UnivariateSpline(
            lc_time, lc_flux, ext=3, k=4, s=0)
        del mv
        gc.collect()

    if netcal_caltable_dir:
        params['netcal_caltable_dir'] = netcal_caltable_dir
    elif 'netcal_caltable_dir' not in params:
        params['netcal_caltable_dir'] = output_dir

    # Run pipeline
    obs_cal = preimcal.preim_pipeline(obs, **params)

    # Determine output naming from scattering variant and preimcal steps
    uncal_basename = os.path.basename(uvfits_path).replace('.uvfits', '')
    cal_basename = get_cal_basename(uncal_basename, params)
    output_path = os.path.join(output_dir, f"{cal_basename}.uvfits")
    obs_cal.save_uvfits(output_path)

    del obs, obs_cal
    gc.collect()

    return output_path


# ═══════════════════════════════════════════════════════════════════════
#  Ground Truth Generation
# ═══════════════════════════════════════════════════════════════════════

def make_ground_truth(scattered_movie_path: str,
                      output_dir: str,
                      model_name: str,
                      band: str,
                      preimcal_params: dict) -> dict:
    """Generate onsky/deblur/dsct ground truth movies.

    For each scattering variant, apply the corresponding deblurring
    and save the resulting movie.

    Args:
        scattered_movie_path: path to scattered movie HDF5.
        output_dir: output directory for truth movies.
        model_name: model identifier.
        band: frequency band label.
        preimcal_params: preimcal parameters dict.

    Returns:
        Dict mapping {variant_name: filepath}.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    mov = eh.movie.load_hdf5(scattered_movie_path)
    mov.reset_interp(bounds_error=False)

    variants = preimcal_params.get('variants', {
        'onsky': {'do_deblurr': False},
        'deblur': {'do_deblurr': True},
        'dsct': {'do_deblurr': True},
    })

    sm = so.ScatteringModel()

    for variant_name, variant_params in variants.items():
        do_deblurr = variant_params.get('do_deblurr', False)

        if do_deblurr:
            # Deblur the movie
            truth_list = []
            for im in tqdm(mov.im_list(),
                           desc=f"  Truth ({variant_name})"):
                im_deblurred = sm.Deblur_Image(im.switch_polrep("stokes"))
                truth_list.append(im_deblurred)
            truth_mov = eh.movie.merge_im_list(truth_list)
        else:
            truth_mov = mov

        output_path = os.path.join(
            output_dir,
            f"{model_name}_{band}_{variant_name}_truth.hdf5")
        truth_mov.save_hdf5(output_path)
        results[variant_name] = output_path

    del mov
    gc.collect()

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Batch Processing
# ═══════════════════════════════════════════════════════════════════════

def run_batch(config: dict,
              models: List[str] = None,
              step: str = 'all') -> None:
    """Process multiple models in serial. Memory-efficient.

    Args:
        config: full configuration dict.
        models: list of model names (default: all from config).
        step: pipeline step to run ('caltable', 'uncal', 'preimcal',
              'groundtruth', 'all').
    """
    if models is None:
        models = config['models']['generate']

    obs_config = config['observation']
    time_config = config['time']
    corruption = config['corruption']
    seeds = config['seeds']
    output = config['output']
    compute = config.get('compute', {})

    bands = obs_config.get('bands', ['LO'])
    uvfits_dir = obs_config['uvfits_dir']

    for model_name in models:
        for band in bands:
            uvfits_file = obs_config['uvfits_files'].get(band)
            if not uvfits_file:
                continue

            uvfits_path = os.path.join(uvfits_dir, uvfits_file)
            array_file = obs_config.get('array_file', './arrays/EHT2017.txt')
            obs_ref = load_obs_ref(uvfits_path, array_file)

            model_path = os.path.join(output['model_dir'],
                                       f"{model_name}.hdf5")
            uncal_dir = os.path.join(output['data_dir'], 'uncal')
            caltable_dir = config['caltable']['dir']

            if step in ('uncal', 'all'):
                print(f"\n[uncal] {model_name} / {band}")
                make_synthetic_dataset(
                    model_path=model_path,
                    obs_ref=obs_ref,
                    output_dir=uncal_dir,
                    caltable_dir=caltable_dir,
                    band=band,
                    tstart=time_config['tstart'],
                    tstop=time_config['tstop'],
                    tshift=time_config.get('tshift', 0),
                    apply_scat=corruption['apply_scattering'],
                    rngseed=seeds['scattering_screen'],
                    seed=seeds['observation_noise'],
                    add_th_noise=corruption['add_thermal_noise'],
                    gain=True,
                    dterm=corruption.get('dterm', False),
                    nproc_scatter=compute.get('nproc_scatter', 32))

            gc.collect()

    print("\n[OK] Batch processing complete.")
