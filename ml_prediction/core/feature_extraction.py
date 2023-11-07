"""
:module: ml_prediction.core.feature_extraction
:auth: Nathan T Stevens
:email: ntsteven at uw.edu
:org: Pacific Northwest Seismic Network
:license: MIT (2023)
:purpose: Provide extension of obspy.signal.trigger module
          methods for pick onset probability predictions from
          ML models (e.g., EQTransformer; Mousavi et al., 2020)
          assuming peaks generally take the form of an normal
          distribution
"""
import numpy as np
from obspy import UTCDateTime
from obspy.signal.trigger import trigger_onset
from pandas import Timestamp, DataFrame
from pyrocko.gui.marker import Marker


def format_timestamp(pick_object):
    """
    Extract an epoch timestamp from a variety of pick object formats
    :: INPUT ::
    :param pick_object: Currently handles:
                        obspy.core.utcdatetime.UTCDateTime
                        pandas._libs.tslibs.timestamps.Timestamp
                        pyrocko.gui.markers.Marker (and child-classes)
                        timestamp.timestamp
    :: OUTPUT ::
    :return time: [float] epoch time
    """
    if isinstance(pick_object,UTCDateTime):
        time = pick_object.timestamp
    elif isinstance(pick_object,Timestamp):
        time = pick_object.timestamp()
    elif isinstance(pick_object,Marker):
        time1 = pick_object.get_tmin()
        time2 = pick_object.get_tmax()
        if time1 == time2:
            time = time1
        else:
            time = (time1 + time2)/2
    elif isinstance(pick_object, datetime):
        time = datetime
    else:
        print('Input object of type %s not handled by this method'%(str(type(pick_object))))
        time = False
    return time



def normal_pdf_error(p, x, y_obs):
    """
    Calculate the misfit between an offset normal distribution (Gaussian)
    with parameters:
    p[0] = A       - Amplitude of the distribution
    p[1] = mu      - Mean of the distribution
    p[2] = sigma   - Standard deviation of the distribution

    and X, y_obs data that may 

    :: INPUTS ::
    :param p: [array-like] see above
    :param x: [array-like] independent variable, sample locations
    :param y_obs: [array-like] dependent variable at sample locations

    :: OUTPUT ::
    :return y_err: [array-like] misfit calculated as y_obs - y_cal
    """
    # Calculate the modeled y-values given positions x and parameters p[:]
    y_cal = p[0]*np.exp(-0.5*((x - p[1])/p[2])**2)
    y_err = (y_obs - y_cal)
    return y_err


def fit_probability_peak(prediction_trace, fit_thr_coef=0.1, mindata=30, p0=None):
    """
    Fit a normal distribution to an isolated peak in a phase arrival prediction trace from
    a phase picking/detection ML prediction in SeisBench formats.

    Fitting of the normal distribution is conducted using scipy.optimize.leastsq()
    and the supporting method `normal_pdf_error()` included in this module.

    :: INPUTS ::
    :param prediction_trace: [obspy.core.trace.Trace]
                                Trace object windowed to contain a single prediction peak
                                with relevant metadata
    :param obs_utcdatetime:  [datetime.datetime] or [None]
                                Optional reference datetime to compare maximum probability
                                timing for calculating delta_t. This generally is used
                                for an analyst pick time.
    :param treshold_coef:    [float]
                                Threshold scaling value for the maximum data value used to
                                isolating samples for fitting the normal distribution
    :param mindata:           [int]
                                Minimum number of data requred for extracting features
    :param p0:                [array-like]
                                Initial normal distribution fit values
                                Default is None, which assumes
                                - amplitude = nanmax(data), 
                                - mean = mean(epoch_times where data >= threshold)
                                - std = 0.25*domain(epoch_times where data >= threshold)

    :: OUTPUTS ::
    :return amp:            [float] amplitude of the model distribution
                                IF ndata < mindata --> this is the maximum value observed
    :return mean:           [float] mean of the model distribution in epoch time
                                IF ndata < mindata --> this is the timestamp of the maximum observed value
    :return std:            [float] standard deviation of the model distribution in seconds
                                IF ndata < mindata --> np.nan
    :return cov:            [numpy.ndarray] 3,3 covariance matrix for <amp, mean, std>
                                IF ndata < mindata --> np.ones(3,3)*np.nan
    :return err:            [float] L-2 norm of data-model residuals
                                IF ndata < mindata --> np.nan
    :return ndata:          [int] number of data used for model fitting
                                IF ndata < mindata --> ndata
    """
    # Get data
    data = prediction_trace.data
    # Get thresholded index
    ind = data >= fit_thr_coef * np.nanmax(data)
    # Get epoch times of data
    d_epoch = prediction_trace.times(type='timestamp')
    # Ensure there are enough data for normal distribution fitting
    if sum(ind) >= mindata:
        x_vals = d_epoch[ind]
        y_vals = data[ind]
        # If no initial parameter values are provided by user, use default formula
        if p0 is None:
            p0 = [np.nanmax(y_vals), np.nanmean(x_vals), 0.25*(np.nanmax(x_vals) - np.nanmin(x_vals))]
        outs = leastsq(normal_pdf_error, p0,
                       args=(x_vals, y_vals),
                       full_output=True)
        amp, mean, std = outs[0]
        cov = outs[1]
        err = np.linalg.norm(normal_pdf_error(outs[0], x_vals, y_vals))
    
        return amp, mean, std, cov, err, sum(ind)
    
    else:
        try:
            return np.nanmax(data), float(d_epoch[np.argwhere(data = np.nanmax(data))]), np.nan, np.ones((3,3))*np.nan, np.nan, sum(ind)
        except:
            breakpoint()


def process_predictions(prediction_trace, et_obs=None, thr_on=0.1,
                        thr_off=0.1, fit_pad_sec=0.1, fit_thr_coef=0.1,
                        ndata_bounds=[30, 9e99]):
    """
    Extract statistical fits of normal distributions to prediction peaks from 
    ML prediction traces that trigger above a specified threshold.

    :: INPUTS ::
    :param prediction_trace:    [obspy.core.trace.Trace]
        Trace containing phase onset prediction probability timeseries data
    :param et_obs:              [None or list of epoch times]
        Observed pick times in epoch time (timestamps) associated with the
        station/phase-type for `prediction_trace`
    :param thr_on:              [float] trigger-ON threshold value
    :param thr_off:             [float] trigger-OFF threshold value
    :param fit_pad_sec:         [float] 
        amount of padding on either side of data bounded by trigger ON/OFF
        times for calculating Gaussian fits to the probability peak(s)
    :param fit_thr_coef:[float] Gaussian fit data 
    :param ndata_bounds [2-tuple of int] 
        minimum & maximum count of data for each trigger window

    :: OUTPUT ::
    :return df_out:     [pandas.dataframe.DataFrame]
        DataFrame containing the following metrics for each trigger
        and observed pick:
        'et_on'     - Trigger onset time [epoch]
        'et_off'    - Trigger termination time [epoch]
        'p_scale'   - Probability scale from Gaussian fit model \in [0,1]
        'et_mean'   - Expectation peak time from Gaussian fit model [epoch]
        'et_max'    - timestamp of the maximum probability [epoch]
        'det_obs_prob' - delta time [seconds] of observed et_obs[i] - et_max
                            Note: this will be np.nan if there are no picks in
                                  the trigger window
        'et_std'    - Standard deviation of Gaussian fit model [seconds]
        'L2 res'    - L2 norm of data - model residuals for Gaussian fit
        'ndata'     - number of data considered in the Gaussian model fit
        'C_pp'      - variance of model fit for p_scale
        'C_uu'      - variance of model fit for expectation peak time
        'C_oo'      - variance of model fit for standard deviation
        'C_pu'      - covariance of model fit for p & u
        'C_po'      - covariance of model fit for p & o
        'C_uo'      - covariance of model fit for u & o
    """
    # Define output column names
    cols = ['et_on', 'et_off', 'p_scale', 'et_mean', 'et_max',
            'det_obs_prob', 'et_std', 'L2 res', 'ndata',
            'C_pp', 'C_uu', 'C_oo', 'C_pu', 'C_po', 'C_uo']
    # Get pick indices with Obspy builtin method
    triggers = trigger_onset(prediction_trace.data,
                             thr_on, thr_off,
                             max_len=ndata_bounds[1],
                             max_len_delete=True)
    times = prediction_trace.times(type='timestamp')
    # Iterate across triggers:
    feature_holder = []
    for _trigger in triggers:
        _t0 = times[_trigger[0]]
        _t1 = times[_trigger[1]]
        # If there are observed time picks provided, search for picks 
        wind_obs = []
        if isinstance(et_obs, list):
            for _obs in et_obs:
                if _t0 <= _obs <= _t1:
                    wind_obs.append(_obs)
        _tr = prediction_trace.copy().trim(starttime=UTCDateTime(_t0) - fit_pad_sec,
                                           endtime=UTCDateTime(_t1) + fit_pad_sec)
        # Conduct gaussian fit
        outs = fit_probability_peak(_tr, fit_thr_coef=fit_thr_coef,
                                    mindata=ndata_bounds[0])
        # Get timestamp of maximum observed data
        et_max = _tr.times(type='timestamp')[np.argmax(_tr.data)]
        
        # Iterate across observed times, if provided
        # First handle the null
        if len(wind_obs) == 0:
            _det_obs_prob = np.nan
            feature_line = [_t0, _t1, outs[0], outs[1], et_max,
                            _det_obs_prob, outs[2], outs[4], outs[5],
                            outs[3][0, 0], outs[3][1, 1], outs[3][2, 2],
                            outs[3][0, 1], outs[3][0, 2], outs[3][1, 2]]
            feature_holder.append(feature_line)
        # Otherwise produce one line with each delta time calculation
        elif len(wind_obs) > 1:
            for _wo in wind_obs:
                _det_obs_prob = _wo - et_max
                feature_line = [_t0, _t1, outs[0], outs[1], et_max,
                                _det_obs_prob, outs[2], outs[4], outs[5],
                                outs[3][0, 0], outs[3][1, 1], outs[3][2, 2],
                                outs[3][0, 1], outs[3][0, 2], outs[3][1, 2]]
                feature_holder.append(feature_line)

    df_out = DataFrame(feature_holder, columns=cols)
    return df_out


def _pick_quality_mapping(X, grade_max=(0.02, 0.03, 0.04, 0.05, np.inf), dtype=np.int32):
    """
    Provide a mapping function between a continuous parameter X
    and a discrete set of grade bins defined by their upper bounds

    :: INPUTS ::
    :param X: [float] input parameter
    :param grade_max: [5-tuple] monotonitcally increasing set of
                    values that provide upper bounding values for
                    a set of bins
    :param dtype: [type-assignment method] default nunpy.int32
                    type formatting to assign to output
    
    :: OUTPUT ::
    :return grade: [dtype value] grade assigned to value
    
    """
    # Provide sanity check that INF is included as the last upper bound
    if grade_max[-1] < np.inf:
        grade_max.append(np.inf)
    for _i, _g in enumerate(grade_max):
        if X <= _g:
            grade = dtype(_i)
            break
        else:
            grade = np.nan
    return grade
    


def ml_prob_models_to_PICK2K_msg(feature_dataframe, pick_metric='et_mean',
                                 std_metric='et_std', qual_metric='',
                                 m_type=255, mod_id=123, org_id=1,
                                 seq_no=0,):
    """
    Convert ml pick probability models into Earthworm PICK2K formatted
    message strings
    -- FIELDS -- 
    1.  INT Message Type (1-255)
    2.  INT Module ID that produced this message: codes 1-255 signifying, e.g., pick_ew, PhaseWorm
    3.  INT Originating installation ID (1-255)
    4.  Intentional 1 blank (i.e., ‘ ‘)
    5.  INT Sequence # assigned by picker (0-9999). Key to associate with coda info.
    6.  Intentional 1 blank
    7.  STR Site code (left justified)
    8.  STR Network code (left justified)
    9.  STR Component code (left justified)
    10. STR Polarity of first break
    11. INT Assigned pick quality (0-4)
    12. 2 blanks OR space for phase assignment (i.e., default is ‘  ‘)
    13. INT Year
    14. INT Month
    15. INT Day
    16. INT Hour
    17. INT Minute
    18. INT Second
    19. INT Millisecond 
    20. INT Amplitude of 1st peak after arrival (counts?)
    21. INT Amplitude of 2nd peak after arrival
    22. INT Amplitude of 3rd peak after arrival
    23. Newline character

    """
    msg_list = []
    for _i in range(len(feature_dataframe)):
        _f_series = feature_dataframe.iloc[_i,:]
        grade = _pick_quality_mapping(_f_series[qual_metric])
        # Fields 1, 2, 3, 4
        fstring =  f'{m_type:3d}{mod_id:3d}{org_id:3d} '
        # Fields 5 - 8
        fstring += f'{seq_no:4d} {_f_series.sta:-5s}{_f_series.net:-2s}'
        fstring += f'{_f_series.cha:-3s} '
        # Fields 10 - 
        # fstring += f' {}
        msg_list.append(fstring)
    
    return msg_list
