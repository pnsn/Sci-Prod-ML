**Purpose**: brief instructions on how to install TensorFlow (and most modern ML modules) on Apple M1/M2 chipsets.

**Note**: This uses an adapted version of the environment.yml from ESS 469 (`environments/environment_AppleM1.yml`) that has been tested on a MacBook Air with M1 (2020) and a light-weight install using the environment proposed in Reference (1) (`environments/environment_TF_Apple.yml`).

## Method
Follow macOS-side installation instructions from references (2) & (3)  
    1. Install xcode command line tools  
        `xcode-select --install`  
    2. Install Miniforge - follow instructions in reference (4)  
    3. Then turn off the default base environment (ref 2)  
        `conda config --set auto_activate_base false`  
    
Run the following version of `conda env create` from the comments and gist in references (1) & (4), respectively  
`CONDA_SUBDIR=osx-arm64 conda env create -f environment_AppleM1.yml`

## Known issues
Conducting step 3. appears to require using the full path to conda environments to invoke:  
`conda activate <my_env>`  
e.g.,  
`conda activate /Users/nates/miniconda3/envs/seisbench`  

These paths can be found using  
`conda env list`  

## Compiler Contact Information
Nathan Stevens:
ntsteven@uw.edu

## References
environment_AppleM1.yml install instructions based on: 
(1) https://gist.github.com/corajr/a10af2f893c4b79275281f6b6fd915d3  
(2) https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706  
(3) https://github.com/conda-forge/miniforge  
(4) https://developer.apple.com/forums/thread/711792  
