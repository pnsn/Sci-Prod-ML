# Sci-Prod-ML
PNSN Science Products team Machine Learning workflows/utilities development space

This repository acts as a development space / testing ground for machine-learning enhancement of seismic analysis workflows for the Pacific Northwest Seismic Network (PNSN). Content will largely remain in `feature` and `develop` branches due to the experimental nature of these code snippets. 

Realistically, code chunks developed in this space that have reasonable operational connections will be extracted as seeds for initial `develop` and `main` branches on more-focused repositories in the PNSN code space.

-Nate Stevens (Dec 2023)

# Installation

The current dependency set for this repository can be installed on an Apple M2 chipset with:  
`conda create -f environments/env_dev_apple_silicon.yml`  

Some modifications (or omissions) on channels and specific versioning may be necessary for installation on other operating systems and hardware.

# Other (older) installation recipes via YAML  

This repository provides uses a minimal installation of `SeisBench` that can be installed via `conda` by invoking: 

`conda create -f environments/seisbench_lite_env.yml`  

Testing on Apple M1 chipsets suggests the following modified invocation is needed for a successful installation in conjunction with a `miniforge` installation:  

`CONDA_SUBDIR=osx-arm64 conda env create -f environments/seisbench_lite_env.yml`


# Contents
### environments/
 - Contains YAML files and notes on environment installation

### examples/
 - Contains Jupyter notebooks documenting "toy" workflow elements.

# Contact Info
Supervisor: jrhartog@uw.edu  
Primary editor: ntsteven@uw.edu  

# Supporting Repositories
### SeisBench  - ML workflow benchmarking
https://github.com/seisbench/seisbench
### PNW-ML - utilities for PNW Store
https://github.com/niyiyu/PNW-ML
https://github.com/niyiyu/pnwstore
### libcomcat - COMCAT query tools
https://code.usgs.gov/ghsc/esi/libcomcat-python  

# References
#### SeisBench
Woollam, J., Münchmeyer, J., Tilmann, F., Rietbrock, A., Lange, D., Bornstein, T., Diehl, T.,
    Giunchi, C., Haslinger, F., Jozinović, D., Michelini, A., Saul, J., & Soto, H. (2022).
    SeisBench—A Toolbox for Machine Learning in Seismology. Seismological Research Letters,
    93(3): 1695–1709. doi: https://doi.org/10.1785/0220210324

#### PNW Store
Ni, Y., Hutko, A., Skene, F., Denolle, M., Malone, S., Bodin, P., Hartog, R., & Wright, A. 
    (2023). Curated Pacific Northwest AI-ready Seismic Dataset. Seismica, 2(1). 
    doi: https://doi.org/10.26443/seismica.v2i1.368