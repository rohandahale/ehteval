# Evaluation and validation scripts for black hole video reconstructions

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://rohandahale.github.io/ehteval/)
[![PyPI version](https://img.shields.io/pypi/v/ehteval.svg)](https://pypi.org/project/ehteval/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified driver to automate the evaluation pipeline for Sgr A* Dynamics project submissions.

## Documentation

📚 **[Read the full documentation →](https://rohandahale.github.io/ehteval/)**

## Installation

### Conda Environment

Create and activate the conda environment using the provided `environment.yml`:

```bash
git clone https://github.com/rohandahale/ehteval.git
cd ehteval
conda env create -f environment.yml
conda activate evaluation
```

## Julia Installation (Optional - for VIDA polarimetric analysis)

Julia is only required if you plan to run `vida_pol.py` for polarimetric ring fitting. 
If you don't need this feature, you can skip this section.

Install Julia 1.10.9 from [juliaup](https://github.com/JuliaLang/juliaup):

```bash
curl -fsSL https://install.julialang.org | sh

source ~/.bashrc
juliaup add 1.10.9
juliaup default 1.10.9
```

Install the required Julia packages:

```bash
cd src
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This reads the `src/Project.toml` and `src/Manifest.toml` to install packages including:
- VIDA
- VLBISkyModels
- Comrade
- Other optimization and data handling packages

## Verify Installation

```bash
# Test Python environment
python -c "import ehtim; import numpy; import pandas; print('Python OK')"

# Test Julia packages (if installed)
julia --project=src -e 'using VIDA; using VLBISkyModels; println("Julia OK")'
```

## Configuration

The entire run is configured via `params.yml`. Key parameters include:

- **Models**: List of models to process
- **Run Settings**: `data_band`, `scattering`, and `pipeline` (single choice per run)
- **Mode**: Set `is_bayesian: True` for multiple-sample inputs (glob pattern) or `False` for single HDF5 inputs
- **Steps**: Toggle specific evaluation modules (e.g., `chisq`, `nxcorr`, `vida`) in the `run_steps` section

## Usage

Run the evaluation pipeline:

```bash
python evaluate.py params.yml
```

## Output

Results are saved to: `<results_dir>/<model>_<pipeline>/`

The script copies the configuration file used into the results directory for reproducibility.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.