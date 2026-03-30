=======================
Synthetic Data Pipeline
=======================

This page documents the synthetic data generation framework within ``ehteval``.
Use it to generate realistic synthetic observations from model movies for
use in method validation and benchmarking.

----

Quick Start
-----------

**Recommended entry point for new users:** open ``tutorials.ipynb`` in
Jupyter. It walks through the full workflow interactively.

.. code-block:: bash

   conda activate evaluation
   cd ehteval/
   jupyter notebook tutorials.ipynb

For bulk generation of the full model/data sets:

.. code-block:: bash

   # 1. Generate all model movies
   python generate_models.py --config models_config.yaml

   # 2. Generate all synthetic datasets (all steps)
   python generate_data.py --config data_config.yaml

   # 3. Run only a specific step for specific models
   python generate_data.py --config data_config.yaml --step preimcal --models mring+hsCW

----

Architecture
------------

.. code-block::

   models_config.yaml  ──►  generate_models.py  ──►  models/*.hdf5
   data_config.yaml    ──►  generate_data.py    ──►  synthetic_data/
                                 │
                       src/models.py          ← base model primitives
                       src/syntheticdata.py   ← core data generation engine
                       src/preimcal.py        ← pre-imaging calibration

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Responsibility
   * - ``src/models.py``
     - Creates ``ehtim.Movie`` objects from physical parameters. Never touches observations or noise.
   * - ``src/syntheticdata.py``
     - Simulates observations from movies (scattering, gains, caltables). Never knows specific model names.
   * - ``src/preimcal.py``
     - Applies post-observation calibration pipeline. No knowledge of model generation.
   * - ``generate_models.py``
     - Wires model names to builder functions via registry pattern.
   * - ``generate_data.py``
     - Orchestrates the full model → scatter → observe → preimcal pipeline.

----

Configuration Files
-------------------

models_config.yaml
^^^^^^^^^^^^^^^^^^

Controls model generation. Key sections:

.. code-block:: yaml

   seeds:
     model_generation: 42   # RNG seed — ensures reproducibility

   image:
     fov_uas: 200
     npix: 200

   polarization:
     mbreve_mean: 0.2       # fractional linear polarization
     vbreve_mean: 0.002     # fractional circular polarization

   models:
     generate:
       - mring+hsCW
       - crescent
       - grmhd1

     recipes:
       mring+hsCW:
         type: "mring_hs"   # registered type in generate_models.py
         total_flux: 2.7
         hs_flux: 0.3
         period_min: 80
         direction: "CW"
         # ... other params

**YAML anchors** are used to avoid parameter duplication across mring+hs variants:

.. code-block:: yaml

   _anchors:
     mring_base: &mring_base
       total_flux: 2.7
       PA: 120
       diameter_uas: 52
       # ... shared params

   recipes:
     mring+hsCW:
       <<: *mring_base       # merges all mring_base params
       type: "mring_hs"
       period_min: 80
       direction: "CW"

data_config.yaml
^^^^^^^^^^^^^^^^

Controls synthetic data generation. Key sections:

.. code-block:: yaml

   seeds:
     scattering_screen: 1   # seed for MakeEpsilonScreen
     observation_noise: 1   # seed for gain/noise simulation

   observation:
     uvfits_files:
       LO: "hops_3601_SGRA_LO_netcal_LMTcal_10s_ALMArot_JCcalDouble_dcal.uvfits"
       HI: "hops_3601_SGRA_HI_netcal_LMTcal_10s_ALMArot_JCcalDouble_dcal.uvfits"

   caltable:
     dir: "./caltable_april11/"
     generate_new: false   # true = generate new caltable from scratch

   preimcal:
     tint_options: [60, 120]   # time averaging options (seconds)
     variants:
       onsky:   {do_deblurr: false, ref_optype: null}
       deblur:  {do_deblurr: true,  ref_optype: null}
       dsct:    {do_deblurr: true,  ref_optype: "quarter1"}

----

Pipeline Steps
--------------

generate_models.py
^^^^^^^^^^^^^^^^^^

**Step 1:** Generate model movies from recipes in ``models_config.yaml``.

For each model, outputs:

- ``<model>.hdf5`` — movie frames
- ``<model>.fits`` — time-averaged frame
- ``<model>_params.yaml`` — generation parameters (provenance)
- ``avg/<model>.png`` — average frame image
- ``gifs/<model>_total.gif`` — total intensity animation
- ``gifs/<model>_lp.gif`` — linear polarization animation

CLI options:

.. code-block:: bash

   python generate_models.py --config models_config.yaml
   python generate_models.py --config models_config.yaml --models mring+hsCW point
   python generate_models.py --config models_config.yaml --no-plots    # skip GIFs
   python generate_models.py --config models_config.yaml --dry-run     # preview only

generate_data.py
^^^^^^^^^^^^^^^^

**Steps 2–4:** Generate synthetic uvfits from model movies.

.. code-block:: bash

   # Full pipeline (all steps)
   python generate_data.py --config data_config.yaml

   # Step-by-step
   python generate_data.py --config data_config.yaml --step caltable
   python generate_data.py --config data_config.yaml --step uncal --models mring+hsCW
   python generate_data.py --config data_config.yaml --step preimcal
   python generate_data.py --config data_config.yaml --step groundtruth

Output structure:

.. code-block::

   synthetic_data/april11_v4/
   ├── uncal/
   │   ├── {model}_{band}_{flags}.uvfits   ← raw synthetic observation (e.g. mring_LO_scat_gA_gP_th)
   │   ├── {model}_{band}_{flags}.hdf5     ← scattered model movie
   │   └── {model}_{band}_{flags}.fits     ← averaged frame
   ├── fulltime/
   │   ├── data/
   │   │   ├── netcal+tavg60s/
   │   │   │   ├── {model}_{band}_{flags}_netcal_tavg60_onsky.uvfits
   │   │   │   ├── {model}_{band}_{flags}_netcal_tavg60_deblur.uvfits
   │   │   │   └── {model}_{band}_{flags}_netcal_tavg60_dsct.uvfits
   │   │   └── netcal+tavg120s/
   │   └── groundtruth/
   │       ├── {model}_{band}_onsky_truth.hdf5
   │       ├── {model}_{band}_deblur_truth.hdf5
   │       └── {model}_{band}_dsct_truth.hdf5
   └── besttime/
       └── ... (same structure)

----

Pre-imaging Calibration Order
------------------------------

``src/preimcal.py`` implements the standard EHT pre-imaging pipeline.
The order is fixed and must not be changed:

.. list-table::
   :header-rows: 1
   :widths: 5 30 65

   * - #
     - Step
     - Notes
   * - 1
     - Network calibration (netcal)
     - LC-based (``do_lc_netcal``) or static ZBL (``do_static_netcal``)
   * - 2
     - Flux normalization
     - Divide visibilities by intra-site light curve
   * - 3
     - LMT calibration
     - Optional (``do_LMTcal``); use if LMT data not pre-calibrated
   * - 4
     - JCMT phase calibration
     - Optional (``do_JCMTcal``)
   * - 5
     - Time averaging
     - ``tint`` seconds (e.g. 60s or 120s)
   * - 6
     - Band merging
     - Optional (``do_mergebands``); for LO+HI combined products
   * - 7
     - Systematic errors
     - Fractional noise floor (``syserr``); skip if negative
   * - 8
     - Refractive noise floor
     - ``ref_optype``: ``"quarter1"``, ``"quarter2"``, ``"dime"``, or ``None``
   * - 9
     - PSD noise
     - Optional (``do_psd_noise``); variability budget
   * - 10
     - Deblurring
     - Optional (``do_deblurr``); for ``deblur``/``dsct`` variants
   * - 11
     - Flux rescaling
     - Multiply back by light curve
   * - 12
     - Time flagging
     - ``do_timeflag`` with ``tstart``/``tstop`` bounds

----

Adding Custom Models
--------------------

To add a new model type so it works with ``generate_models.py``:

**Step 1** — Implement your builder in ``src/models.py``:

.. code-block:: python

   def make_my_model_movie(params: dict, obs_ref) -> eh.Movie:
       \"\"\"Description of your custom model.\"\"\"
       times, Nframes = _get_frame_times(obs_ref)
       framelist = []
       for t in times:
           im = ...  # build your image
           im.time = t
           framelist.append(im)
       return eh.movie.merge_im_list(framelist, interp='linear', bounds_error=True)

**Step 2** — Register it in ``generate_models.py``:

.. code-block:: python

   from src.models import make_my_model_movie
   # In build_model_registry():
   registry['my_model'] = (make_my_model_movie, None)

**Step 3** — Add a recipe in ``models_config.yaml``:

.. code-block:: yaml

   my_model:
     type: "my_model"
     total_flux: 2.7
     # ... your model-specific parameters

**Step 4** — Add to the generate list and run:

.. code-block:: bash

   python generate_models.py --config models_config.yaml --models my_model

----

Reproducibility
---------------

.. important::

   Every stochastic operation receives an explicit seed.
   The combination of ``models_config.yaml`` (or ``data_config.yaml``) +
   per-model ``_params.yaml`` sidecar files constitutes a **complete provenance record**.

Seed control:

- ``seeds.model_generation`` (models_config.yaml) — seeds ``numpy.random`` before each model build
- ``seeds.scattering_screen`` (data_config.yaml) — seeds ``MakeEpsilonScreen``
- ``seeds.observation_noise`` (data_config.yaml) — seeds gain/noise simulation

The config YAML is automatically copied into every output directory.

----

Adapting for a New Epoch
------------------------

To generate data for a different EHT observation (e.g. April 6, 2017):

.. code-block:: yaml

   # data_config.yaml excerpt for April 6
   observation:
     epoch: "3598"
     mjd: 57849
     uvfits_files:
       LO: "ER6_SGRA_2017_096_lo_hops_netcal_LMTcal_10s_ALMArot_dtermcal.uvfits"
   time:
     tstart: 7.5
     tstop: 13.0
   gains:
     offset:
       AA: 0.029
       # ... April 6 specific station gains

No code changes are needed — only the YAML changes.
