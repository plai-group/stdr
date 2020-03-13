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

from sys import platform
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import copy
import Util.util as util
from Util.util import FloatTensor, LongTensor, cuda
from ModelBase import ModelBase

# Set up Sacred experiment to record experiments.
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred import SETTINGS
SETTINGS.CAPTURE_MODE = 'sys'
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment()
ex.observers.append(FileStorageObserver.create("my_runs"))
ex.captured_out_filter = apply_backspaces_and_linefeeds

# # Configure MatPlotLib to use LaTeX fonts.
# from matplotlib import rc
# rc('text', usetex=True)
# rc('font', family='serif', size=8)
# rc('text.latex', preamble='\\usepackage{amsmath},\\usepackage{amssymb}')


@ex.config
def experiment():
    """
    AW - experiment - this function is where all of the experimental settings are configured. By wrapping it in the
    Sacred @ex.config wrapper Sacred will recognise it as the experimental configuration and automatically generate
    the experimental log.
    :return: None (although creates Sacred experiment as a side-effect).
    """

    # We use Macs as our personal machines, and linux clusters for larger jobs.
    # This switch (signified by the `cluster' flag) allows for changing the
    # size of experiments without direct modification of the source code.
    # Other than that, very little is different.
    if str(platform).find('linux') != -1:
        cluster = True
    else:
        cluster = False

    TRAINSURROGATE = False
    DOSMC = True
    USE_PREMADE_VAL_DATASET = None  # './WormSimNFValData.p'  # {None, $path}
    USE_PREMADE_TRAIN_DATASET = None  # './WormSimNFTrainData.p'  # {None, $path}
    USE_PRETRAINED_NETWORK = './Results/RNF_local_2020_02_21_16_06_37/RNF_best_model.pt'  # './Results/Success/RNF_cluster_2019_09_26_12_27_27/RNF_best_model.pt'  # {None, $path}

    # Define any global settings that are frequently different based on whether we are training on a local
    # machine or on a larger cluster.
    if not cluster:
        # CNF training and architectural parameters,
        # See MADE.__init__ for more details on many of these.
        nf_batch_size = 100             # Batch size for training.
        nf_steps_to_take = 100000       # Optimization steps to take.
        nf_hidden_size = 128            # Size of hidden layers
        nf_n_hidden = 1                 # Number of hidden layers per block.
        nf_n_blocks = 5                 # Number of MADE blocks.
        nf_lr = 1e-04                   # Learning rate for surrogate.
        nf_epochs = 20                  # Number of times to evaluate the evidence per experiment.
        nf_max_batch_size = 1000000     # Largest batch size that can be handled in the NF.
        nf_n_validate = 10000           # The size of the validation batch to use.
        nf_n_train = None               # {None, int} None signifies training data generated on the fly.

        # Settings for when the CNF is deployed in an SMC scenario.
        smc_experiments = 50            # Number of independent SMC sweeps to perform (to calculate variances).
        smc_particles = 100             # Default number of particles to use in the SMC sweeps.
        smc_data_to_generate = 5        # Number of independent datasets to generate.
        smc_tMax = 100                  # Length of the datasets to generate.

    else:

        # CNF training and architectural parameters,
        # See MADE.__init__ for more details on many of these.
        nf_batch_size = 1000            # Batch size for training.
        nf_steps_to_take = 500000       # Optimization steps to take.
        nf_hidden_size = 128            # Size of hidden layers
        nf_n_hidden = 1                 # Number of hidden layers per block.
        nf_n_blocks = 5                 # Number of MADE blocks.
        nf_lr = 1e-04                   # Learning rate for surrogate.
        nf_epochs = 200                 # Number of times to evaluate the evidence per experiment.
        nf_max_batch_size = 1000000     # Largest batch size that can be handled in the NF.
        nf_n_validate = 10000           # The size of the validation batch to use.
        nf_n_train = None               # {None, int} None signifies training data generated on the fly.

        # Settings for when the CNF is deployed in an SMC scenario.
        smc_experiments = 100           # Number of independent SMC sweeps to perform (to calculate variances).
        smc_particles = 100             # Default number of particles to use in the SMC sweeps.
        smc_data_to_generate = 100      # Number of independent datasets to generate.
        smc_tMax = 100                  # Length of the datasets to generate.

    # Define parameters of the simulator.
    noise_states = LongTensor([0, 1, 2, 3])  # These are the states that will have noise added to them under the model.
    observed_states = LongTensor([0, 1])  # Which states are observed.
    nBalls = 1
    observation_frequency = 1
    true_ring_width = 0.03
    nDim = len(noise_states)
    dt = 1.0

    # Define the default parameters of the graphical model. These define the standard deviation of the additive noise
    # in the forward plant model and the observation model.
    default_parameters = {'noise_sd': torch.log(FloatTensor([0.1, 0.1, 0.1, 0.1])),
                          'observation_noise_sd': torch.log(FloatTensor([0.1] * observed_states.shape[0]))}

    # Standard nf settings, these probably dont need changing much...
    nf_model = 'maf'                # CNF architecture family, normally MAF.
    nf_cond_size = nDim * nBalls    # Number of states we are conditioning on.
    nf_input_order = 'sequential'   # Order states are presented to the CNF layers.
    nf_no_batch_norm = False        # Batch norm in layers?
    nf_activation_fn = 'relu'       # Activation between layers.
    nf_input_size = noise_states.shape[0] * nBalls  # Number of inputs to the flow.

    # Inscribe some default stuff.
    reject_invalid = True  # Use a different script if you dont want rejection...
    timenow = time.strftime("%Y_%m_%d") + "_" + time.strftime("%H_%M_%S")
    experiment_name = 'RNF'
    _s = 'cluster' if cluster else 'local'
    folder_save_name = './Results/' + experiment_name + '_' + _s + '_' + timenow
    report_name = folder_save_name + '/report.txt'
    config_save_name = folder_save_name + '/{}_experiment.p'.format(experiment_name)
    model_save_name = folder_save_name + '/{}_best_model.pt'.format(experiment_name)
    LOSS_BOUND = 100  # The maximum loss we will report to sacred
    _true_param_the = copy.deepcopy(default_parameters)
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Record the device we are operating on.
    if cuda:
        device = torch.cuda.current_device()
    else:
        device = 'cpu'


class Model(ModelBase):
    """
    AW - Model - inherits from ModelBase - adapt the base class to be applicable for RNF.
    """

    def init_particle(self, _n=1):
        """
        AW - init_particle - Initialize _n particles from the prior. Note that this does not pay attention to the
        validity constraint imposed by bot, so when this is used it is often called in a safe initialization loop that
        re-initializes until the required number of _valid_ particles has been reached.
        :param _n:  int:    number of particles to initialize.
        :return:    tensor: the initialized states.
        """
        state_x1 = torch.distributions.Normal(0.0, 1.0).sample([_n, self.nBalls, 1])  # x coordinate.
        state_x2 = torch.distributions.Normal(0.0, 1.0).sample([_n, self.nBalls, 1])  # y coordinate.
        state_v1 = torch.distributions.Normal(0.0, 0.1).sample([_n, self.nBalls, 1])  # x velocity.
        state_v2 = torch.distributions.Normal(0.0, 0.1).sample([_n, self.nBalls, 1])  # y velocity.
        state = torch.cat((state_x1, state_x2, state_v1, state_v2), dim=-1).to(self.device)
        return state

    def test_bot(self, _state_in, _radius_in=None):
        """
        AW - test_bot - test whether or not the states, defined in _state_in are valid under the constraint, or whether
        it should return bot. In this case, validity is determined via iteration, and hence iterate must be called.
        :param _state_in:   tensor: tensor of input states to test validity.
        :param _radius_in:  tensor(float): radius of the point before iteration. If this radius changed more than a
                                    fixed  amount (specified by true_ring_width) during a single iteration, the particle
                                    is  invalid and hence should be allocated bot. It is sometimes useful to test for a
                                    bot for states that have already been iterated. In this case, set radius_in to the
                                    radius prior to iteration, and then _state_in represents the iterated particles.
        :return:            tensor: tensor containing whether or not each particle should be considered valid or not.
        """
        _x_in = _state_in[:, :, 0:2].clone()        # Cloen to states to make sure there are no side effects.
        _radius = torch.norm(_x_in, p=2, dim=2)     # Get the radius of the states.

        # If the particles have not already been iterated...
        if _radius_in is None:
            # Get the input radius.
            _radius_in = torch.norm(_x_in[:, :, 0:2], p=2, dim=2)

            # Iterate the particles deterministically, this will return NaN if the particle fails (due to bot).
            _x_it, _, _, _ = self.iterate(_state_in, _deterministic=True, _max_attempts=1)

            # Search for NaNs to indicate bot.
            _bot = torch.any(torch.isnan(_x_it.view(_x_it.shape[0], -1)), dim=1)

        else:
            # If they have been iterated, compate the _radius_in to the current radius.
            _bot = torch.sum((_radius > (_radius_in + self.true_ring_width)) +
                             (_radius < (_radius_in - self.true_ring_width)), dim=1)
            _bot = _bot > 0.0  # > 0.0 if either inside or outside the ring.

        return _bot

    def obs_model(self, _input_states, _param):
        """
        AW - obs_model - model for generated observations from state. Those dimensions that are observed are specified
        in the config by using self.observed_states. It is assumed that all nBalls are observed in the same way.
        Assumes observations are zero mean, gaussian distorted observations of a subset of the state.
        :param _input_states:   tensor: the states that are being observed.
        :param _param:          dict:   parameters of noise model.
        :return:                tensor: tensor of floats containing
        """
        _z = torch.randn(_input_states[:, :, self.observed_states].shape, device=self.device)
        _obs = _z * torch.exp(_param['observation_noise_sd']) + _input_states[:, :, self.observed_states]
        return _obs

    def _iterate(self, _input_states, _param, _true_model=False):
        """
        AW - _iterate - iterate the states specific to the annulus example. If true model is specified, the state will
        curve in a circular orbit, otherwise, the state continues on in a straight line.
        :param _input_states:   tensor:     the states we wish to iterate through the plant model.
        :param _param:          dict:       parameters of the observation model for generating observations.
        :param _true_model:     bool:       if true, we are using the 'true model' and hence we explicitly curve
                                            the state so that it follows a roughly circular orbit. (The radius will
                                            gradually increase due to the Euler integration scheme, but this doesn't
                                            matter too much here.)
        :return: tuple: _output_states:     tensor:     the iterated states. always has value, but if the value is
                                                        invalid bot will be true for that entry.
                        _bot:               tensor(bool): tensor of which particles failed.
                        _obs:               tensor:     the observation associated with each particle.
        """

        # Clone state.
        _output_states = _input_states.clone()

        # Get the input radius and _speed_.
        _r_in = torch.norm(_input_states[:, :, 0:2], p=2, dim=2)
        _s_in = torch.norm(_input_states[:, :, 2:4], p=2, dim=2)

        # Iterate.
        _output_states[:, :, 0:2] += self.dt * _output_states[:, :, 2:4]

        # Catch any new bots.
        _bot = self.test_bot(_output_states, _r_in)

        # If we are using the true model, curve the ball, otherwise, let it go straight by not changing the velocity.
        if _true_model:
            # Update the velocity to be perpendicular.
            _theta = torch.atan2(_output_states[:, :, 1], _output_states[:, :, 0])
            _output_states[:, :, 2] = - torch.sin(_theta) * _s_in
            _output_states[:, :, 3] = torch.cos(_theta) * _s_in

        # Generate an observation.
        _obs = self.obs_model(_output_states, _param)

        return _output_states, _bot, _obs

    """ --------------------------------------------------------------------------------------- DEFINE MENIAL STUFF. """

    def plot_density(self, _model, _data, _ax, _ranges, _range_total=None):
        """
        AW - this function plots the conditional density for a number of different states.

        :param _model:  the model object of type ModelBase. must have a logprob function for evaluating the
                        conditional density at point xy, conditioned on the values in _data.
        :param _data:   The x-y-dot{x}-dot{y} list containing the state that the normalizing flow will be conditioned
                        on. This `simulates' the latent state held by the particle in an SMC sweep.
        :param _ax:     the axis onto which the densities will be plotted.
        :param _ranges: data range to plot densities for above and below the value in _data.
        :param _range_total: Force the axis limits to take particular values irrespective of the conditioning state.
        :return:
        """
        data = _data.clone()
        if data.shape[0] > 1:
            raise RuntimeError  # Can only condition on a single point.

        (xmin, xmax), (ymin, ymax) = _ranges

        idx_to_vary = [0, 1]  # The coordinates of what we are varying in the figure-level grid.

        # sample uniform grid
        n = 100
        xx1 = torch.linspace(xmin, xmax, n)
        xx2 = torch.linspace(ymin, ymax, n)
        xx, yy = torch.meshgrid(xx1, xx2)
        xy = torch.stack((xx.flatten(), yy.flatten()), dim=-1).squeeze().to(_model.device)

        # Repeat the sampling points for each input point.
        # Add zeros for velocity.
        data = torch.repeat_interleave(data, xy.shape[0], dim=0).to(_model.device)
        xy = xy.unsqueeze(1)
        xy = torch.cat((torch.zeros((xy.shape[0], 1, self.noise_states.shape[0] - 2)).to(_model.device), xy), dim=-1)
        xy = xy.to(_model.device)

        ball_kwargs = {'facecolor': 'None', 'edgecolor': 'k', 'linestyle': '--'}
        util.draw_ball(_ax, [0, 0], np.linalg.norm(data[0, 0, 0:2]) + self.true_ring_width, **ball_kwargs)
        util.draw_ball(_ax, [0, 0], np.linalg.norm(data[0, 0, 0:2]) - self.true_ring_width, **ball_kwargs)

        # --------- BASE DISTRIBUTION ---------
        p_density = self.noise_density(data,
                                       xy,
                                       self.default_parameters).exp()
        _ax.contour(xx.to('cpu').detach() + data[0, 0, idx_to_vary[0]].to('cpu').detach(),
                   yy.to('cpu').detach() + data[0, 0, idx_to_vary[1]].to('cpu').detach(),
                   p_density.view(n, n).data.to('cpu').detach().numpy(), cmap='Reds')

        # --------- LEARNED MODEL ---------
        q_density = self.proxy_iter_model_prob(xy, data).exp().to('cpu')
        _ax.contour(xx.to('cpu').detach().numpy() + data[0, 0, idx_to_vary[0]].to('cpu').detach().numpy(),
                   yy.to('cpu').detach().numpy() + data[0, 0, idx_to_vary[1]].to('cpu').detach().numpy(),
                   q_density.view(n, n).data.to('cpu').detach().numpy(), cmap='Greens')

        # Set the limits.
        if _range_total is None:
            _ax.set_xlim(xmin + data[0, 0, idx_to_vary[0]].to('cpu').detach().numpy(),
                         xmax + data[0, 0, idx_to_vary[0]].to('cpu').detach().numpy())
            _ax.set_ylim(ymin + data[0, 0, idx_to_vary[1]].to('cpu').detach().numpy(),
                         ymax + data[0, 0, idx_to_vary[1]].to('cpu').detach().numpy())
        else:
            _ax.set_xlim(_range_total[0], _range_total[1])
            _ax.set_ylim(_range_total[0], _range_total[1])

        _ax.grid(True)
        _ax.set_axisbelow(True)
        _ax.set_aspect('equal', 'datalim')
        return None

    def plot_sample_and_density(self):
        """
        AW - plot_sample_and_density - large, slightly nasty function for generating the plots used in the paper.
        For a series of inputs, specified by data_all, where here input corresponds to the latent state on which the
        normalizing flow is conditioned; test the density over perturbations as defined by either the apriori specified
        noise model, or the learned perturbation model.

        As part of this, we also test the number of bots for the validation data, to ensure that the learned q is
        reducing the number of bots generated.
        :return:
        """
        self.iter_proxy.eval()

        # The range around the datapoint to plot the density for.
        # Here we use +/- 3 standard deviations as specified by the model.
        ranges_density = [[-3.0 * np.exp(self.param_the['noise_sd'][0].detach().to('cpu').numpy()),
                           3.0 * np.exp(self.param_the['noise_sd'][1].detach().to('cpu').numpy())],
                          [-3.0 * np.exp(self.param_the['noise_sd'][0].detach().to('cpu').numpy()),
                           3.0 * np.exp(self.param_the['noise_sd'][1].detach().to('cpu').numpy())]]

        # Look at a stationary point at the 9 verticies of the four unit squares.
        _vel_x = 0.0
        _vel_y = 0.0
        z = 1.0
        data_all = [[[[-z, z, 0.0, 0.0]],       [[0.0, z * 2,  0.0, 0.0]],      [[z, z, 0.0, 0.0]]],
                    [[[-z * 2, 0.0, 0.0, 0.0]], [[0.0, 0.0,  0.0, 0.0]],        [[z * 2, 0.0, 0.0, 0.0]]],
                    [[[-z, -z, 0.0, 0.0]],      [[0.0, -z * 2, 0.0, 0.0]],      [[z, -z, 0.0, 0.0]]]]

        # Define the axis.
        fig, axs = plt.subplots(np.alen(data_all), np.alen(data_all[0]),
                                figsize=(4, 4), tight_layout=True, squeeze=False)

        for _i in range(len(data_all)):
            for _j in range(len(data_all[_i])):
                # Evaluate the log density defined by the NF at each point.
                if data_all[_i][_j] is not None:
                    data = FloatTensor([data_all[_i][_j]])
                    self.plot_density(self.iter_proxy, data, axs[_i, _j], ranges_density)
                else:
                    # If data is set to be none, turn the axis off.
                    axs[_i, _j].axis('off')

        # Save the image.
        fig_save_name = './{}/images/{}_'.format(self.folder_save_name, self.experiment_name)
        if self.cluster: fig_save_name += 'cluster_'
        fig_save_name += 'density_array_{0}balls_{1:02}_{2:07}.pdf'.format(self.nBalls, self.exp_num, self.i)
        plt.savefig(fig_save_name)
        plt.close('all')

        # Now test the bot rate.
        _bots_q, _bots_p = 0.0, 0.0
        _the_detach = self.detach_params(self.param_the)
        _particles_init = self.validation_x.clone().to(self.device)
        _, _, _bots_p, _ = self.iterate(_particles_init, _the_detach, _max_attempts=1)
        util.log_scalar('nf-p-bots', _bots_p.to('cpu').item() / len(_particles_init), self.i.item(), self._run)
        _, _, _bots_q, _ = self.iterate(_particles_init, _the_detach, _max_attempts=1, _q=self.proxy_iter_model_sample)
        util.log_scalar('nf-q-bots', _bots_q.to('cpu').item() / len(_particles_init), self.i.item(), self._run)
        util.echo_to_file(self.report_name, '[RNF        ]: For {} sampled: bots using p: {}, bots using q: {}'.
                          format(len(_particles_init), _bots_p, _bots_q))

        # Return the bots, can be handy to have these back to track the progress of the algorithm.
        return _bots_p, _bots_q


@ex.automain
def main(_seed, _config, _run):

    # Grab the settings.
    settings, _seed = util.saved_experiment(_seed, _config, _run)
    util.echo_to_file(settings.report_name, '\n\n[RNF        ]: Starting experiment.')

    # Initialize empty model.
    # This will be an instance of the class defined above, but inherits many of its characteristics and implementations
    # from the base class defined in ModelBase.
    model = Model(settings, _run=_run)

    # # n_train indicates that we are using an a-priori generated corpus of training data, as opposed to generating
    # # the data online during the training process. Generating data online means that you are accessing an unlimited
    # # pool of training data, but at the expense of computational cost. For more expensive simulators, generating
    # # data upfront makes most sense, otherwise, generate on the fly.
    # if settings.nf_n_train is not None: model.generate_training_data()

    # If we are training the surrogate from scratch, call the training routine.
    # We might be pre-loading a trained surrogate for deployment and so we dont want to re-train everything.
    if settings.TRAINSURROGATE: model.learn_surrogate()

    # If we are testing the surrogate on an SMC sweep example, call the SMC routine.
    # Set determinsitc to be true as we want the ground-truth data to be specified from the un-perturbed model.
    # Supersampling means that we generate data at a higher temporal fidelity and then downsample to the fidelity
    # used in inference. This means that we do not run into problems of the true data being invalid.
    if settings.DOSMC: util.do_smc_experiments(model)

    # Goodbye!
    util.echo_to_file(settings.report_name, '[Util       ]: Experiment complete. \n')

    # Wait for Sacred to sync.
    time.sleep(30)
