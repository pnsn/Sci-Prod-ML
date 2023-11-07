"""
:module: ml_prediction.prediction
:auth: Nathan T. Stevens
:email: ntsteven at uw.edu
:org: Pacific Northwest Seismic Network
:license: MIT License (2023)
:purpose: Methods wrapping model prediction using the PyTorch
          and SeisBench APIs focused on multi-station inputs
          of (semi)arbitrary length and computational expedience,
          as opposed to the default prediction modality offered by
          SeisBench that introduces large batches of windows from
          single stations to PyTorch predictions

          Some inputs to methods in this module are described in 
          the following module:
          ml_prediction.preprocessing

:attribution:
          Approaches herein build on code developed by Yiyu Ni
          for the ELEP project and conversations with Yiyu:
          https://github.com/congcy/ELEP/blob/main/docs/tutorials/example_BB_continuous_data_PB_B204.ipynb
"""
import numpy as np
from tqdm import tqdm
from obspy import UTCDateTime
import torch
import seisbench.models as sbm


####################################
# MODEL LOADING CONVENIENCE METHOD #
####################################


def initialize_EQT_model(sbm_model=sbm.EQTransformer.from_pretrained('pnw'),
                         nstep=1800, nlb=500, nrb=500,
                         device=torch.device('cpu'),
                         filter_args_overwrite=False,
                         filter_kwargs_overwrite=False):
    """
    Convenience method documenting a "standard" model for PNSN
    detection/phase picking prediction using EQTransformer

    :: INPUTS ::
    :param sbm_model: [seisbench.model.<model_subclass>]
            specified PyTorch model built with the SeisBench
            WaveformModel abstraction and weights loaded
    :param nstep: [int]
            Number of datapoints to advance an annotation window by for each
            prediction
    :param nlb: [int]
            Number of datapoints to blind on the left side of prediction output
            vectors
    :param nrb: [int]
            Number of datapoints to blind on the right side of predction output
            vectors
    :param filter_args_overwrite: [bool], [None], or [dict]
            Arguments for the optional prefilter built into the sbm_model class
            methods.
            Default `False` preserves the default settings associated with
            the pretrained model arguments
            `None` disables any pre-specified filter (also have to do this to
            filter_kwargs_overwrite)
            A dict can overwrite pre-specified filter arguments. See
            obspy.core.trace.Trace.filter() for formatting.
    :param filter_kwargs_overwrite: [bool], [None], or [dict]
            See documentation for `filter_args_overwrite`

    :: OUTPUT ::
    :return model: initialized seisbench.models.<model_subclass> object placed
                into evaluate mode.

    ** References **
    EQTransformer algorithm -               Mousavi et al. (2020)
    SeisBench/PyTorch EQT implementation -  Woollam et al. (2022)
                                            Münchmeyer et al. (2022)
    'pnw' EQT model weights -               Ni et al. (2023)
    """
    # Define local variable
    model = sbm_model
    # Proivide options for overwriting filter args/kwargs
    if filter_args_overwrite:
        model.filter_args = filter_args_overwrite
    if filter_kwargs_overwrite:
        model.filter_kwargs = filter_kwargs_overwrite
    # Assign what hardware the model is running predictions on
    model.to(device)
    # Append annotate() arguments to model
    model._annotate_args['overlap'] = ('Overlap between prediction windows in\
                                        samples (only for window prediction \
                                        models)', nstep)
    model._annotate_args['blinding'] = ('Number of prediction samples to \
                                        discard on each side of aeach \
                                        window prediction', (nlb, nrb))
    # Place model into evaluate mode
    model.eval()
    return model, device


######################
# Prediction Methods #
######################

def run_prediction(windows_tt, model, device):
    """
    Run predictions on windowed data and return a prediction
    vector of the same scale
    :: INPUTS ::
    :param windows_tt: [(m, 3, n) torch.Tensor]
                m preprocessed data windows of 3-channel
                data with n data for each channel
    :param model: [seisbench.models.<model_subclass>]
                model object with which to conduct prediction
    :param device: [torch.device]
                hardware specification on which to conduct prediction

    :: OUTPUT ::
    :return preds: [(m, p, n) numpy.ndarray]
                array of p predicted model parameters corresponding
                to m windows comprised of n data.
                Meaning of the p^{th} parameter will depend on the model.
                E.g.,
                    EQTransformer
                        p == 0 --> Detection
                        p == 1 --> P pick probability
                        p == 2 --> S pick probability
                    PhaseNet
                        p == 0 --> P pick probability
                        p == 1 --> S pick probability
                        p == 2 --> Noise segment probability                
    """ 
    # Preallocate space for predictions
    preds = np.zeros(shape=windows_tt.shape, dtype=np.float32)
    # Do prediction
    _torch_preds = model(windows_tt.to(device))
    # Unpack predictions onto CPU-hosted memory as numpy arrays
    for _i, _p in enumerate(_torch_preds):
        preds[:, _i, :] = _p.detach().cpu().numpy()

    return preds


def run_batched_prediction(windows, model, device, batch_size=4):
    """
    Run prediction with a specified batch_size (number of windows)
    passed to a given call of run_prediction(). batch_size should
    be approximately #cpu * 2 for best performance

    :: INPUTS ::
    :param windows: [(nwin, cchan, mdata) numpy.ndarray]
                array of preprocessed, windowed waveform data
                ready to convert into a torch.Tensor
    :param model: [seisbench.models.<model_subclass>]
                model object with which to conduct prediction
    :param device: [torch.device]
                hardware specification on which to conduct prediction
    :param batch_size: [int]
                number of windows to include per batch
    
    :: OUTPUT ::
    :return pred: [(nwin, ppred, mdata) numpy.ndarray]
                predicted parameter values. See run_prediction()
                for further information
    """
    # Preallocate space for predictions in memory
    pred = np.zeros(shape=windows.shape, dtype=np.float32)
    # Alias batch_size and ensure type == int
    _bs = int(batch_size)
    # Get the number of full batches
    n_fb = pred.shape[0]//_bs
    # Iterate across full windows
    for _i in tqdm(range(n_fb - 1)):
        # Get subset of windows and convert to torch.Tensor
        _wtt_batch = torch.Tensor(windows[_i*_bs:(_i + 1)*_bs])
        # Run prediction on batch
        _pred = run_prediction(_wtt_batch, model, device)
        # merge batch prediction into output
        pred[_i*_bs:(_i + 1)*_bs, :, :] = _pred
    # run last batch even if # of windows < _bs
    _wtt_batch = torch.Tensor(windows[n_fb*_bs:, :, :])
    # directly write prediction result from last batch into preds
    pred[n_fb*_bs:, :, :] = run_prediction(_wtt_batch, model, device)
    # return result
    return pred


def run_seisbench_pred(stream, model):
    """
    Convenience method for conducting a prediction workflow as facilitated by
    classes and methods included in SeisBench
    """
    annotation_stream = model.annotate(stream)
    return annotation_stream
