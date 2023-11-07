# ml_prediction module  

:auth: Nathan T. Stevens  
:email: ntsteven at uw.edu  
:org: Pacific Northwest Seismic Network  
:license: MIT (2023)  
:purpose:  
This module and submodules provide methods for conducting machine-learning (ml) based analyses of seismic data on arbitrarily scaled and sequenced packets of near-real-time seismic data using the PyTorch API, supported in small part by the SeisBench API, in a computationally efficient manner for this particular data ingestion/flow modality  

:attribution:  
Core sections of this code are based on source-code written by Yiyu Ni (niyiyu at uw.edu) for the ELEP project, and conversations between N. Stevens, Y. Ni, R. Hartog (PNSN) and M. Denolle (UW).

The specific Jupyter notebook can be found here:  
        https://github.com/congcy/ELEP/blob/main/docs/tutorials/example_BB_continuous_data_PB_B204.ipynb  


## :: Structure ::  

**drivers**
   - *driver_local.py        wrapper for running processing pipeline on data saved  
                            to local disk (IN PLANNING)  
   - *driver_ew.py           wrapper for running processing pipeline in a continuous  
                            EarthWorm instance (IN PLANNING)  
   - \__init__.py             Dunder-init  
  
**core**
- preprocessing.py       
    - for converting obspy.core.stream formatted data into windowed, preprocessed, numpy arrays for input to a PyTorch model (v1.0.0 ready for debug/tests)  
- prediction.py  
    - for conducing batched processing of windowed data with core prediction execution via PyTorch 
    - (v1.0.0 ready for debug/tests)  
- postprocessing.py
    - for converting output windowed predictions back into data modalitices of source obspy.stream data
    - (v1.0.0 ready for debug/tests)  
- *feature_extraction.py  
    - for conducting pick/detection classification and estimating pick PDFs with scaled normal distributions
    - (UNDER DEVELOPMENT)
- *classes.py 
    - contains definitions of classes specific to this module that merge utilities of `obspy`, `libcomcat-python` and `pyrocko/snuffler`. These classes are primarily used in data exploration & parameter tuning documented in `ml_prediction.notebooks` (see below)
- *util.py                
    - supporting subroutines (UNDER DEVELOPMENT)  
- \__init__.py
    - Dunder-init  


**notebooks**  
- Jupyter Notebooks documenting parameter selection and processing pipeline tests leading to the default structure/settings in the `drivers/` scripts  
*preprocessing*  
- instrument_response_removal.ipynb  
    - Tests investigating the effects  
- gappy_data_processing.ipynb  


*process scaling*  
- batch_scaling.ipynb
    - Tests data throughput rate against compute resource and batch scling



**data**
- Holder directory for outputs of `ml_prediction.data.fetch_testdata`  
- fetch_testdata.py  
    - Fetch example data for the following scenarios:  
        1. Bremerton_6C: A few days of 6-C station (broadband + accelerometer) data during the 2017 EQ sequence near Bremerton, WA. Used to test the performance of pre-processing steps on both instrument types with source-receiver distances ranging
        2. Full_Array: A few minutes of data from the full PNSN array including the M4.3 near Port Townsend, WA on 2023-10-08T19:21 (Local Time)
        3. Gappy_Data: 20 minutes of data from PNSN stations that have abundant gaps
        4. Analogue_EQs: (IN DEVELOPMENT) a collection of analogus EQ records intended to approximate instrument types and source-receiver distances for the following PNW scenarios:  
            A. Cascadia megathrust M9+ (using Tohoku, 2011; Maule 1980)  
            B. Seattle Fault Zone M7+ (using Ridgecrest; others?)
            C. 