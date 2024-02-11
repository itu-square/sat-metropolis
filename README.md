# MCMC+SAT

Python library using SAT/SMT samplers in MCMC algos

## Installation

1. Install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
2. Create mcmc_sat environment: `conda create -n mcmc_sat` (say yes to the question)
3. Set environment variable to avoid use system packages `conda env config vars set PYTHONNOUSERSITE=1` (note: the environment must be active)
4. Reactivate environment `conda deactivate` and `conda activate mcmc_sat`
5. Install jupyter lab in environment: conda install -c conda-forge jupyterlab (with mcmc_sat environment active)
6. Add environemnt as a kernel: `python -m ipykernel install --user --name=mcmc_sat`
7. Install Z3, numpy and arviz in environment, e.g., run `%pip install z3-solver numpy arviz` in a notebook cell running on kernel in the `mcmc_sat` environment
   - Z3 version: z3_solver-4.12.4.0-py2.py3-none-manylinux2014_x86_64.whl.metadata
   - Numpy version: numpy-1.26.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata
   - Arviz version: arviz-0.16.1-py3-none-any.whl.metadata
8. Download and install [SPUR](https://github.com/ZaydH/spur)
   /Steps for Ubuntu/
   1. `git clone https://github.com/ZaydH/spur.git`
   2. `cd spur/`
   3. `sudo snap install cmake --classic`
   4. `sudo apt install libgmp-dev `
   5. `./build.sh`
   6. `ln -s <path_to_spur_repo>/build/Release/spur ~/.local/bin/spur`
9. Download and install [MegaSampler](https://github.com/chaosite/MeGASampler)
   1. `sudo apt install git build-essential python3-minimal python3-dev libjsoncpp-dev python3-venv`
   2. `python -m venv venv --upgrade` (somehow did not create `venv/bin/activate` ...)
   3. `python -m venv venv` (this solved the problem above, and created `venv/bin/activate`)
   4. `git clone https://github.com/chaosite/MeGASampler.git`
   5. `git clone https://github.com/chaosite/z3.git` (patched z3 for SMTSampler coverage)
   6. `pushd z3`
   7. `python scripts/mk_make.py --python`
   8. `cd build`
   9. `sudo make install`
   10. `popd`
   11. `cd MeGASampler`
   12. `make`
   13. `conda env config vars set LD_LIBRARY_PATH=<path_to_dir_hosting_virtual_environment>/venv/lib` (this is an example to add the variable to a conda environment, it is also possible to simply add it to `.bashrc`)
   14. `ln -s <path_to_megasampler>/megasampler ~/.local/bin/megasampler`
   

