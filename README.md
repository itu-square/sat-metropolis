# SAT-Metropolis

Python implementation of the SAT-Metropolis algorithm presented in the accompanying paper: "SAT-Metropolis: Combining Markov Chain Monte Carlo with SAT/SMT sampling". SAT-Metropolis uses SAT/SMT samplers as proposal distributions to effectively sample in probabilistic inference problems with hard constraints.

**Code for paper experiments:** The notebook `experiments/experiments.ipynb` contains the code for all the experiments in the accompanying paper.

**Get started with SAT-Metropolis:** The notebooks in folders `experiments/sat` and `experiments/smt` contain multiple examples on using SAT-Metropolis with the different backends currently available, namely, SPUR, CMSGen and MegaSampler. Below we provide installation instructions for a conda environment for the library, and instructions to install each backend.

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
   5. Spur would not compile with newer versions of g++ because gcc does not include `stdint.h`. Thus, it is necessary to modify the flags in the `CMakeFile.txt` file as follows:
      ```
      set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wall -include stdint.h")

      set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -O3 -DNDEBUG -Wall -include stdint.h")

      set(CMAKE_CXX_FLAGS_PROFILING "${CMAKE_CXX_FLAGS_PROFILING} -std=c++11 -O3 -g
          -DNDEBUG -Wall -fno-omit-frame-pointer -include stdint.h")
      ```
   6. `./build.sh`
   7. `ln -s <path_to_spur_repo>/build/Release/spur ~/.local/bin/spur`
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
   
10. Download and install [CMSGen](https://github.com/meelgroup/cmsgen) (this is the same guide as in the repository)
    1. `sudo apt install zlib1g-dev help2man`
    2. `cd cmsgen`
    3. `mkdir build && cd build`
    4. `cmake ..`
    5. `make`
    6. `sudo make install`
    7. `sudo ldconfig`
