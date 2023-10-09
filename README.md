# Sci-Prod-ML
PNSN Science Products team Machine Learning workflows/utilities

# Installation
This repository currently uses a minimal installation of `SeisBench` that can be installed via `conda` by invoking: 

`conda create -f environments/seisbench_lite_env.yml`  

Testing on Apple M1/M2 chipsets suggests the following modified invocation is needed for a successful installation:  

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
Jack Woollam, Jannes Münchmeyer, Frederik Tilmann, Andreas Rietbrock, Dietrich Lange, Thomas Bornstein, Tobias Diehl, Carlo Giunchi, Florian Haslinger, Dario Jozinović, Alberto Michelini, Joachim Saul, Hugo Soto; SeisBench—A Toolbox for Machine Learning in Seismology. Seismological Research Letters 2022;; 93 (3): 1695–1709. doi: https://doi.org/10.1785/0220210324

#### PNW Store
Ni, Y., Hutko, A., Skene, F., Denolle, M., Malone, S., Bodin, P., Hartog, R., & Wright, A. (2023). Curated Pacific Northwest AI-ready Seismic Dataset. Seismica, 2(1). https://doi.org/10.26443/seismica.v2i1.368