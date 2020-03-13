# MIT License
#
# Copyright (c) 2020, Andrew Warrington.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import pickle
from matplotlib.patches import Ellipse
from types import SimpleNamespace
from pprint import pprint
from tqdm import tqdm
from copy import deepcopy as dc

# Generate tensor aliases depending on what hardware is available.
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

try:
    import src.Util.particleFilter as pf
except:
    import Util.particleFilter as pf

# Define some 'nice' colours for us to use throughout.
from colormap import hex2rgb
muted_colours = np.asarray(["#4878D0", "#D65F5F", "#EE854A", "#6ACC64", "#956CB4",
                            "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"] * 5)
muted_colours = np.asarray([hex2rgb(_c) for _c in muted_colours]) / 256


""" -------------------------------------------------------------------- SUBROUTINES DOING SMC EXPERIMENTS EN MASSE. """


def do_smc_experiments(_model):
    """
    AW - do_smc_experiments - another nasty function. This is designed to be a wrapper for conducting a multitude of
    experiments and hence does not implement any `core' functionality, and is more an example of how to call functions
    and generate some of the plots we use in the main paper.

    This function automates the process of doing many SMC experiments, and saves the output to a
    (reasonably) well ordered pickle for analysis afterwards. It is currently set up for also sweeping over the number
    of particles used in the sweep for evaluating the efficiency gain from using q. This can be `disabled' by setting
    smc_particles = [_model.smc_particles] to use the number of particles specified in the config.
    :param _model: object of type ModelBase (or child classes) that provides the iterate, score, prior distributions.
    :return: None: Results are written out to a pickle.
    """

    echo_to_file(_model.report_name, '\n[Util       ]: Doing SMC.')

    _initial_state = None
    _deterministic = True
    _supersample = 10
    _data_initial_state = None

    # Force eval mode.
    _model.iter_proxy.eval()

    p_evidences, q_evidences = [], []
    p_bot, q_bot = [], []
    p_time, q_time = [], []
    p_ess, q_ess = [], []
    legend = True

    smc_particles = [_model.smc_particles]  # [100, 75, 50, 25, 10]  # {_model.smc_particles, [list: int]}
    echo_to_file(_model.report_name, '[Util       ]: Testing using: {} particles.'.format(smc_particles))

    for _d in range(_model.smc_data_to_generate):

        data = _model.simulate_super_sampled_trajectory(_initial_state=_data_initial_state,
                                                        _deterministic=_deterministic,
                                                        _supersample=_supersample)

        for _s in range(len(smc_particles)):

            _p_evidences, _q_evidences = [], []
            _p_bot, _q_bot = [], []
            _p_time, _q_time = [], []
            _p_ess, _q_ess = [], []

            _model.smc_particles = smc_particles[_s]

            with tqdm(total=_model.smc_experiments, ncols=120, smoothing=0.0) as pbar:

                for _i in range(_model.smc_experiments):

                    # Only run the p model with the first number of particles we are comparing to.
                    p_results = _model.do_sweep(data,  _add_legend=legend, _initial_state=_initial_state)

                    # Run the model using the learned surrogate.
                    q_results = _model.do_sweep(data, _model.proxy_iter_model_sample, _add_legend=legend,
                                                _initial_state=_initial_state)

                    # Append all of the results to the results files.
                    _p_evidences.append(p_results['log_evidence'])
                    _q_evidences.append(q_results['log_evidence'])
                    _p_bot.append(np.nansum(np.asarray(p_results['b_history'])))
                    _q_bot.append(np.nansum(np.asarray(q_results['b_history'])))
                    _p_time.append(p_results['time'])
                    _q_time.append(q_results['time'])
                    _p_ess.append(p_results['ess_history'])
                    _q_ess.append(q_results['ess_history'])

                    # Update the progress bar.
                    _string = 'particles: {0}; \t p: {1:.5f}, q: {2:.5f}'.format(_model.smc_particles,
                                                                                 p_results['log_evidence'],
                                                                                 q_results['log_evidence'], )
                    pbar.set_postfix(loss=_string)
                    pbar.update()
                    pbar.refresh()
                    legend = False  # Suppress legend for this plot.

            # # < PLOT SOME STUFF. >
            # plt.figure(1)
            # plt.plot(p_results['e_history'], c='r', label='Likelihood, p' if _d == 0 else None)
            # plt.plot(q_results['e_history'], c='g', label='Likelihood, q' if _d == 0 else None)
            # plt.title('Likelihood.')
            # plt.legend()
            # plt.savefig('./{}/images/{}_smc_likelihood.pdf'.format(_model.folder_save_name, _model.experiment_name))
            #
            # plt.figure(2)
            # plt.plot(p_results['b_history'], c='r', label='Bot rate, p' if _d == 0 else None)
            # plt.plot(q_results['b_history'], c='g', label='Bot rate, q' if _d == 0 else None)
            # plt.title('Bottom rate.')
            # plt.legend()
            # plt.savefig('./{}/images/{}_smc_bottom.pdf'.format(_model.folder_save_name, _model.experiment_name))
            #
            # plt.figure(3)
            # plt.plot(np.cumsum(np.asarray(p_results['e_history'], dtype=np.float)), c='r', label='Evidence, p' if _d == 0 else None)
            # plt.plot(np.cumsum(np.asarray(q_results['e_history'], dtype=np.float)), c='g', label='Evidence, q' if _d == 0 else None)
            # plt.title('Cumulative Evidence.')
            # plt.legend()
            # plt.savefig('./{}/images/{}_smc_evidence.pdf'.format(_model.folder_save_name, _model.experiment_name))
            #
            # try:
            #     plt.figure(4)
            #     plt.scatter(np.mean(p_results['x_history'][:, :, 0, 0], axis=1),
            #                 np.mean(p_results['x_history'][:, :, 0, 1], axis=1), label='p', c='r')
            #     plt.scatter(np.mean(q_results['x_history'][:, :, 0, 0], axis=1),
            #                 np.mean(q_results['x_history'][:, :, 0, 1], axis=1), label='q', c='g')
            #     plt.scatter(data[_d]['states'][:, 0], data[_d]['states'][:, 1], label='True', c='b')
            #     plt.axis('equal')
            #     plt.grid(True)
            #     plt.savefig('./{}/images/{}_smc_reconstruction_{}.pdf'.format(_model.folder_save_name,
            #                                                                   _model.experiment_name, _d))
            #     plt.close(4)
            # except:
            #     pass
            # # <\ PLOT SOME STUFF. >

            # Print some output.
            echo_to_file(_model.report_name, '\n[Util       ]: '
                                             '{0}/{1}: '
                                             'Particles: {2}, '
                                             'p mean: {3:.2f} \\pm {4:.2f}, '
                                             'q_mean: {5:.2f} \\pm {6:.2f}. '.
                         format(_d, _model.smc_data_to_generate,
                                _model.smc_particles,
                                np.nanmean(_p_evidences), np.nanstd(_p_evidences),
                                np.nanmean(_q_evidences), np.nanstd(_q_evidences)))

            # Append to results files and upload to Sacred.
            p_evidences.append(_p_evidences)
            q_evidences.append(_q_evidences)
            p_bot.append(_p_bot)
            q_bot.append(_q_bot)
            p_time.append(_p_time)
            q_time.append(_q_time)
            p_ess.append(_p_ess)
            q_ess.append(_q_ess)
            log_scalar('p_evidence_mu', np.nanmean(_p_evidences), _d, _model._run)
            log_scalar('q_evidence_mu', np.nanmean(_q_evidences), _d, _model._run)
            log_scalar('p_evidence_sd', np.nanstd(_p_evidences), _d, _model._run)
            log_scalar('q_evidence_sd', np.nanstd(_q_evidences), _d, _model._run)
            log_scalar('p_failures', np.nansum(np.isnan(_p_evidences)), _d, _model._run)
            log_scalar('q_failures', np.nansum(np.isnan(_q_evidences)), _d, _model._run)
            log_scalar('p_bot', np.nanmean(np.asarray(_p_bot)), _d, _model._run)
            log_scalar('q_bot', np.nanmean(np.asarray(_q_bot)), _d, _model._run)
            log_scalar('p_time', np.nanmean(_p_time), _d, _model._run)
            log_scalar('q_time', np.nanmean(_q_time), _d, _model._run)

        # We are just going to dump this to a pickle for inspection later.
        # This will overwrite the pickle each time, but not much gets written out so its reasonably fast.
        _artefact_name = _model.folder_save_name + '/smc_results.p'
        with open(_artefact_name, 'wb') as f:
            _dict = {'p': {'evidences': p_evidences,
                           'bot': p_bot,
                           'time': p_time},
                     'q': {'evidences': q_evidences,
                           'bot': q_bot,
                           'time': q_time}}
            pickle.dump(_dict, f)

    # Print out some output to close the experiment.
    echo_to_file(_model.report_name, '\n')
    for _i in range(len(p_evidences)):
        echo_to_file(_model.report_name, '[Util       ]: Mean p evidence: {0: .05f} pm {1:.05f} \t '
                                         'Mean q evidence: {2: .05f} pm {3:.05f} \t '
                                         'Q better? mu: {4}, sd: {5}'.
                     format(np.nanmean(p_evidences[_i]),
                            np.nanstd(p_evidences[_i]),
                            np.nanmean(q_evidences[_i]),
                            np.nanstd(q_evidences[_i]),
                            int(np.nanmean(p_evidences[_i]) < np.nanmean(q_evidences[_i])),
                            int(np.nanstd(p_evidences[_i]) > np.nanstd(q_evidences[_i]))))
    echo_to_file(_model.report_name, '\n')

    # Upload this to Sacred as well to make sure we dont lose it.
    echo_to_file(_model.report_name, '\n[Util       ]: Uploading smc_results artefact to Sacred.')
    _artefact_name = _model.folder_save_name + '/smc_results.p'
    with open(_artefact_name, 'wb') as f:
        _dict = {'p': {'evidences': p_evidences,
                       'bot': p_bot,
                       'time': p_time},
                 'q': {'evidences': q_evidences,
                       'bot': q_bot,
                       'time': q_time}}
        pickle.dump(_dict, f)
        _model._run.add_artifact(_artefact_name)

    # Slightly messy little bit of code.
    # This tries to upload whatever model we used to Sacred to retain it for posterity.
    try:
        echo_to_file(_model.report_name, '[Util       ]: Uploading model artefact to Sacred.')
        echo_to_file(_model.report_name, '[Util       ]: Warning -- saving model will destroy current verson.')
        try:
            delattr(_model, 'sim')  # Need to remove any sim.
        except:
            pass
        _artefact_name = _model.folder_save_name + '/smc_model.p'
        with open(_artefact_name, 'wb') as f:
            _dict = {'model': _model}
            pickle.dump(_dict, f)
            _model._run.add_artifact(_artefact_name)
    except:
        echo_to_file(_model.report_name, '[Util       ]: Saving model failed, no real biggie.')
        pass

    echo_to_file(_model.report_name, '[Util       ]: Finished SMC.\n')
    return None


""" ------------------------------------------- FUNCTIONS FOR EVALUATING STANDARD DENSITIES AND STABLE EXPECTATIONS. """


def log_normal_pdf(_mu, _sd, _x):
    if _x.ndim == 1:
        a = 0
    else:
        a = 1
    d = _x.shape[-1]

    return ((-1.0 / (2.0 * np.square(_sd))) * np.sum(np.square(_x - _mu), axis=a)) \
           - (np.float(d) * np.log(_sd)) \
           - ((np.float(d) / 2.0) * np.log(2.0*np.pi))


def normal_pdf(_mu, _sd, _x):
    return np.exp(log_normal_pdf(_mu, _sd, _x))


def torch_log_normal_pdf(_mu, _sd, _x):
    d = _x.shape[-1]
    return ((-1.0 / (2.0 * torch.pow(_sd, 2))) * torch.sum(torch.pow(_x - _mu, 2), dim=(_x.dim()-1))) \
            - (d * torch.log(_sd)) \
            - ((d / 2.0) * torch.log(FloatTensor([2.0*np.pi])))


def torch_normal_pdf(_mu, _sd, _x):
    return torch.exp(log_normal_pdf(_mu, _sd, _x))


def torch_log_imultinormal_pdf(_mu, _sd, _x):
    if _sd.numel() == 1:
        # If there is only one standard deviation, use the standard distribution.
        return torch_log_normal_pdf(_mu, _sd, _x)
    else:
        d = _x.shape[-1]
        _density = (torch.sum((-1.0 / (2.0 * torch.pow(_sd, 2))) * torch.pow(_x - _mu, 2), dim=(_x.dim() - 1))) \
                   - (torch.sum(torch.log(_sd), dim=(_x.dim() - 1))) \
                   - ((d / 2.0) * torch.log(FloatTensor([2.0 * np.pi])))
        return _density


def stable_mean(_p):
    if np.alen(np.shape(_p)) > 1:
        return np.asarray([stable_mean(__p) for __p in _p])

    stabilizer = np.nanmax(_p)
    mean_likelihood = stabilizer + \
                      np.log(np.nansum(np.exp(_p - stabilizer))) - \
                      np.log(np.sum(np.logical_not(np.isnan(_p))))
    return mean_likelihood


# def stable_log_mean(_p):
#     dim = 0                                         # Always doing the mean of the first dimension.
#     if len(_p.shape) > 1: _p = _p.transpose(0, 1)   # Flip if we are providing rows.
#     _stabilizer = torch.max(_p, dim=dim).values     # Find the stabilizer.
#     _a = _p - _stabilizer                           # Stabilize the weights.
#     _b = torch.exp(_a)                              # Turn log probs into probs.
#     _c = torch.mean(_b, dim=dim)                    # Take the mean.
#     _d = torch.log(_c)                              # Take the logarithm of the mean.
#     _e = _d + _stabilizer                           # Restore the scale.
#     if len(_e.shape) > 1: _e = _e.transpose(0, 1)   # Give back the right shape.
#     return _e                                       # Return the log mean.


""" ------------------------------------------------------------------------------ FUNCTIONS FOR SACRED AND LOGGING. """


def log_scalar(name, scalar, step=None, _run=None):
    assert np.isscalar(scalar)  # Tensors, numpy arrays, etc wont work
    try:
        if step is not None:
            _run.log_scalar(name, scalar, step)
        else:
            _run.log_scalar(name, scalar)
    except:
        print('Upload failed.')
        pass


def saved_experiment(_seed, _config, _run):
    # Convert and seed experiment.
    pprint(_config)
    _config = SimpleNamespace(**_config)
    _seed = _config.seed

    os.makedirs(_config.folder_save_name)
    os.makedirs(_config.folder_save_name + '/images')

    with open(_config.config_save_name, 'wb') as f:
        pickle.dump(_config, f)
    _run.add_artifact(_config.config_save_name)
    return _config, _seed


def echo_to_file(_f, _str, _dont_display=False):
    """
    AW - echo_to_file - instead of using `print', use this function, that
    wraps the call to print with a file write, for writing reports that are
    accessible during execution. If the file does not exist, the file is created.
    :param _f: string containing the relative or absolute file path to the file.
    :param _str: the string to be written to the file.
    :param _dont_display: do not display to stdout, just echo to file.
    :return: None
    """
    with open(_f, mode='a+') as fid:
        fid.write(_str + '\n')
    if not _dont_display:
        print(_str)


""" ---------------------------------------------------------------------------------------------- REALLY MISC CRAP. """


def draw_ball(ax, pos, r, **kwargs):
    """
    AW - draw_ball - wrap drawing a sphere/circle on an axis.
    :param ax:      Matplotlib axis object where we wish to draw the object.
    :param pos:     Coordinates of the centre of the ball.
    :param r:       Radius of the ball.
    :param kwargs:  additional arguments to Ellipse.
    :return:        Handle to drawn ellipse.
    """
    e = Ellipse(xy=pos, width=r * 2, height=r * 2, **kwargs)
    ax.add_artist(e)
    return e


def sbe():
    """
    AW - sbe - switch back end to something for local plotting.
    Can be handy for if you are operating locally, dont want all plots to appear, but then want a particular plot
    to appear for visual inspection at the end of the experiment before exiting.
    :return: None
    """
    try:
        plt.switch_backend('qt5agg')
    except:
        pass


def chunkIt(seq, num):
    """
    AW - chunkIt - separate 
    :param seq:     iterable:   sequence to be chunked into num chunks.
    :param num:     int:        the number of chunks desired.
    :return:        iterable:   iterable of the chunks created.
    """
    if num == 1:
        return [seq]

    if torch.is_tensor(seq):
        idx = np.linspace(0, len(seq), num+1).astype(np.int)
        out = [seq[_i:_j, ] for _i, _j in zip(idx[:-1], idx[1:])]
    else:
        out = np.array_split(np.array(seq), num)

    return out

