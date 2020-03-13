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

import numpy as np
import torch
import timeit
import pickle
import time
from tqdm import tqdm
from copy import deepcopy as dc

import Util.util as util
import Util.particleFilter as pf
from Util.util import FloatTensor
from maf import dispatch_model


class ModelBase:

    def __init__(self, _settings, _ex_num=0, _run=None):
        """
        AW - ModelBase - Base class for the object that holds the normalizing flow, as well as the code for interfacing
        with the simulator, initializing particles and performing smc sweeps. This class is an abstract base class
        and hence must be inherited from to define a valid object. This is kind of obvious as you have to insert
        simulation code, probability densities etc to make this work...
        :param _settings:   SimpleNamespace:    object containing all the required experimental settings.
        :param _ex_num:     Int:                indexing the experiment, not used very frequently. Allows save files to
                                                be uniquely named etc.
        :param _run:        Sacred Run:         run object as created by Sacred for uploading things to db.
        """

        util.echo_to_file(_settings.report_name, '[ModelBase  ]: Constructing base model.')

        # This is hideous. :(
        # Define a load of stuff to shut the linter up. These will be overwritten.
        self.folder_save_name, self.experiment_name, self._run, self.report_name, self.nBalls, self.default_parameters,\
           self.USE_PRETRAINED_NETWORK, self.model_save_name, self.nf_lr, self.USE_PREMADE_VAL_DATASET,\
           self.TRAINSURROGATE, self.USE_PREMADE_TRAIN_DATASET, self.nDim, self.dt, self.noise_states, \
           self.observed_states, self.nf_max_batch_size, self.smc_tMax, self.observation_frequency, self.sim_viewer,\
           self.smc_particles, self.device, self.cluster, self.nf_steps_to_take, self.nf_epochs, self.LOSS_BOUND,\
           self.nf_batch_size = [None] * 27

        # Automatically inscribe all of the settings from the config.
        _s = {_k: getattr(_settings, _k) for _k in dir(_settings) if _k[0:2] != '__'}
        for _k in _s:
            if 'ReadOnlyDict' in str(type(_s[_k])):
                # Strange new behaviour in Sacred that pins config dicts to be read only...
                setattr(self, _k, dict(_s[_k]))
            else:
                setattr(self, _k, _s[_k])

        # Learning generative model.
        # The SD of the noise kernel in the forward model.
        # The = theta as in the parameters of a generative model.
        # When we do parameter estimation, this allows us to vary param_the, while retaining the default parameters.
        self.param_the = self.default_parameters

        # Define the proxies / CNF for sampling from the state dependent noise model.
        self.iter_proxy = dispatch_model(self)

        # If we are using a pretrained network, we load it here and overwrite the parameters of the network loaded
        # above. As a result, one has to be careful that the code bases `line up' (i.e. layer widths etc are the same)
        # otherwise Torch will give `variable not found' errors, and this throw and exception.
        # There is additional behaviour in that this will also dump the model used into the folder being used for this
        # experiment as well (the savefile is duplicated). This is simply for portability, i.e. you always have the
        # artifact handy. It also uploads it to sacred (again). This might be undesirable behaviour if storage space
        # on the sacred database is limited.
        if self.USE_PRETRAINED_NETWORK is not None:
            # Try to load the network from the prescribed location.
            # If the network cannot be loaded, throw and unhandled exception.
            try:
                util.echo_to_file(self.report_name, '[ModelBase  ]: Loading network from: ' +
                                  str(self.USE_PRETRAINED_NETWORK))
                self.iter_proxy.load_state_dict(torch.load(self.USE_PRETRAINED_NETWORK, map_location='cpu'))
            except Exception as e:
                util.echo_to_file(self.report_name, '[ModelBase  ]: Failed to load network: ' + str(e))
                raise RuntimeError  # Couldn't find savefile.

            # Duplicate the model for portability.
            util.echo_to_file(self.report_name, '[ModelBase  ]: Duplicating model.')
            torch.save(self.iter_proxy.state_dict(), self.model_save_name)

            # Try uploading the model to sacred again. Failing this step is non-critical, although it might be something
            # you want to look into as it might indicate sacred is not recording.
            try:
                util.echo_to_file(self.report_name, '[ModelBase  ]: Uploading loaded PyTorch model artefact to Sacred.')
                self._run.add_artifact(self.model_save_name)
            except Exception as e:
                util.echo_to_file(self.report_name, '[ModelBase  ]: Couldnt upload to Sacred: ' + str(e))
                pass

            # Put it into eval mode since we are not
            self.iter_proxy.eval()

            # Set some stuff to None that isn't defined here.
            self.optimizer_iter_proxy, self.lr_scheduler = None, None

        else:
            # Since we are learning a new normalizing flow, define the optimizer and the scheduler.
            self.optimizer_iter_proxy = torch.optim.Adam(self.iter_proxy.parameters(), lr=self.nf_lr)
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_iter_proxy,
                                                                           patience=10,
                                                                           min_lr=1.0e-07,
                                                                           factor=0.8,
                                                                           verbose=True, )

        # Define some other stuff that will be overwritten in due course..
        self.exp_num = _ex_num              # Experiment number.
        self.best_loss = float('inf')       # Best loss obtained by the CNF on the validation set.
        self.i = torch.scalar_tensor(0)     # Counter for optimization steps.
        self._run = _run                    # Inscribe the Sacred experiment.

        # If we are training the surrogate, and we are using a fixed validation set (advised) then construct that here.
        # This will either be loaded from another location if that dataset has already been generated, or, it will
        # be built from scratch here and saved here.
        if self.TRAINSURROGATE:
            if self.USE_PREMADE_VAL_DATASET is not None:
                try:
                    # Use a premade dataset that is loaded from a different location.
                    util.echo_to_file(self.report_name, '[ModelBase  ]: Loading validation data at {}'.
                                      format(self.USE_PREMADE_VAL_DATASET))
                    with open(self.USE_PREMADE_VAL_DATASET, 'rb') as f:
                        data = pickle.load(f)
                    self.validation_x = data['x'].to(self.device)       # x = state at t.
                    self.validation_n = data['n'].to(self.device)       # n = (valid) perturbation applied at t.
                    self.validation_y = data['y'].to(self.device)       # y = observation at t+1.
                    self.validation_xi = data['xi'].to(self.device)     # xi = (iterated) state at t+1.
                    self.nf_n_validate = len(self.validation_x)
                    util.echo_to_file(self.report_name, '[ModelBase  ]: Using {} validation examples.'.
                                      format(self.nf_n_validate))
                except Exception as e:
                    # If loading the dataset fails, fall back to generating data.
                    # This might be undesired behaviour, in which case, just comment this code out.
                    util.echo_to_file(self.report_name, '[ModelBase  ]: ' + str(e))
                    util.echo_to_file(self.report_name, '[ModelBase  ]: Loading validation dataset failed... '
                                                        'Falling back to re-generation.')
                    self.generate_validation_data()
            else:
                # Call the subroutine for generating and inscribing the validation data.
                self.generate_validation_data()
            util.echo_to_file(self.report_name, '[ModelBase  ]: Done preparing validation set.\n')

            if self.nf_n_train is not None:
                # Same as above, except for training data.
                if self.USE_PREMADE_TRAIN_DATASET is not None:
                    try:
                        util.echo_to_file(self.report_name, '[ModelBase  ]: Loading training data at {}'.
                                          format(self.USE_PREMADE_TRAIN_DATASET))
                        with open(self.USE_PREMADE_TRAIN_DATASET, 'rb') as f:
                            data = pickle.load(f)
                        self.train_x = data['x'].to(self.device)        # x = state at t.
                        self.train_n = data['n'].to(self.device)        # n = (valid) perturbation applied at t.
                        self.train_y = data['y'].to(self.device)        # y = observation at t+1.
                        self.train_xi = data['xi'].to(self.device)      # xi = (iterated) state at t+1.
                        self.nf_n_train = len(self.train_x)
                        util.echo_to_file(self.report_name, '[ModelBase  ]: Using {} training examples.'.
                                          format(self.nf_n_train))
                    except Exception as e:
                        util.echo_to_file(self.report_name, '[ModelBase  ]: Loading training dataset failed... '
                                                            'Falling back to re-generation.')
                        util.echo_to_file(self.report_name, '[ModelBase  ]: ' + str(e))
                        self.generate_training_data()
                else:
                    self.generate_training_data()
                util.echo_to_file(self.report_name, '[ModelBase  ]: Done preparing training set.')

        time.sleep(1)  # Let the output flush.
        util.echo_to_file(self.report_name, '[ModelBase  ]: Done initializing the base model.\n')

    ''' ----------------------------------- DEFINE THE ABSTRACT FUNCTIONS THAT MUST BE OVERWRITTEN IN CHILD CLASSES. '''

    def init_particle(self, _n=1):
        """
        AW - init_particle - abstract function that must be overridden in child classes.
        :param _n: number of particles to initialize.
        :return: initialized particles.
        """
        raise TypeError  # This has to be overridden in child class.

    def test_bot(self, _state_in):
        """
        AW - test_bot - abstract function that must be overwritten in child class. Tests if the state is valid or not.
        :param _state_in:   tensor:         states to test.
        :return:            tensor(bool):   true if the state is invalid (==bot)
        """
        raise TypeError  # This has to be overridden in child class.

    def obs_model(self, _input_states, _param):
        """
        AW - obs_model - abstract class defining the observation model.
        :param _input_states:   tensor:     tensor containing states to generate observations for.
        :param _param:          dict:       parameters of the observation function.
        :return:                tensor:     observations.
        """
        raise TypeError  # This has to be overridden in child class.

    def _iterate(self, _input_states, _param, _true_model):
        """
        AW - _iterate - abstract function for iterating states.
        :param _input_states:   tensor:     tensor containing states to be iterated.
        :param _param:          tensor:     parameters of the observation or plant model.
        :param _true_model:     bool:       allow different functionality whether or not we are generating data or
                                            performing inference using a misspecified model.
        :return: tuple:         _output_states:     tensor:     iterated states.
                                _bot:               tensor(bool): tensor of which states failed.
                                _obs:               tensor:     tensor of observations generated.
        """
        raise TypeError  # This has to be overridden in child class.

    ''' ------------------------------------------------------------------------ DEFINE LOOP FOR LEARNING SURROGATE. '''

    def learn_surrogate(self):
        """
        AW - learn_surrogate - Wrapper for optimizing the surrogate / proxy conditional normalizing flow.
        This will run the selected optimizer for the desired number of steps, dumping the model every epoch if the
        model has improved. Upload some stats to Sacred as well. Pretty standard stuff.
        :param None.
        """
        util.echo_to_file(self.report_name, '\n[Util       ]: Learning surrogate.')

        try:
            self.train_x.to('cpu')
            self.train_n.to('cpu')
            self.train_y.to('cpu')
            self.train_xi.to('cpu')
            self.validation_x.to('cpu')
            self.validation_n.to('cpu')
            self.validation_y.to('cpu')
            self.validation_xi.to('cpu')
        except:
            pass

        # Do the training loop.
        with tqdm(total=self.nf_steps_to_take, ncols=100, smoothing=0.1) as pbar:
            for self.i in torch.arange(self.nf_steps_to_take):

                # < DIRTY-ASS CODE>
                if ('BBNF' in self.experiment_name) and (self.i > 0.1 * self.nf_steps_to_take):
                    # NOTE - BBNF seems to have some weird issue with big batches at first, so
                    # we start with a smaller batch, and then increase it....
                    if 'increased' not in locals().keys():
                        util.echo_to_file(self.report_name, '\n[Util       ]: \n\nWARNING: increasing batch size\n\n')
                        self.nf_batch_size *= 10
                        increased = True
                # < \ DIRTY-ASS CODE>

                loss = self.take_proxy_step().detach().to('cpu').item()

                # Validate the network and generate plots if we are at the end of an `epoch'.
                if ((self.i + 1) % (self.nf_steps_to_take / self.nf_epochs) == 0) or (self.i == 0):

                    # Validate the network on the validation set, and upload the result to sacred.
                    self.iter_proxy.eval()
                    q_prob_benchmark = - self.proxy_iter_model_prob(self.validation_n, self.validation_x,
                                                                    _detach=True).mean(0).item()
                    if float(self.i.item()) / self.nf_steps_to_take > 0.05 and (q_prob_benchmark < self.LOSS_BOUND):
                        util.log_scalar('Validation Loss', q_prob_benchmark, self.i.item(), self._run)
                    pbar.set_postfix(loss='{:.3f}'.format(q_prob_benchmark))
                    pbar.update()
                    pbar.refresh()

                    # If the network has improved, save the new improved model.
                    if q_prob_benchmark < self.best_loss:
                        torch.save(self.iter_proxy.state_dict(), self.model_save_name)
                        improved = True
                        self.best_loss = dc(q_prob_benchmark)
                    else:
                        improved = False

                    # Try to make some plots, but this function might not exist or fail, and so leave this behaviour.
                    if hasattr(self, 'plot_sample_and_density'):
                        try:
                            self.plot_sample_and_density()
                        except Exception as e:
                            # This function may fail / not exist for some models, in which case, this not critical.
                            # It just means no plots will be generated.
                            util.echo_to_file(self.report_name, '\n[Util       ]: Warning: util.learn_surrogate 1: ')
                            util.echo_to_file(self.report_name, '[Util       ]: ' + str(e))
                            pass

                    # Do some remedial stuff.
                    self.lr_scheduler.step(q_prob_benchmark)
                    self.iter_proxy.train()
                    self.optimizer_iter_proxy.zero_grad()
                    util.echo_to_file(self.report_name, '\n[Util       ]: Done benchmarking at step %d. Improved %s.' %
                                      (self.i.item(), str(improved)))

                else:
                    pbar.set_postfix(loss='{:.3f}'.format(loss))
                    pbar.update()
                    pbar.refresh()

        util.echo_to_file(self.report_name, '\n[Util       ]: Done learning surrogate.\n')

        try:
            util.echo_to_file(self.report_name, '[Util       ]: Uploading best PyTorch model artefact to Sacred.')
            self._run.add_artifact(self.model_save_name)
        except RuntimeWarning as e:
            pass

        # Put us back into eval mode to be safe.
        self.iter_proxy.eval()

    def take_proxy_step(self):
        """
        AW - take_proxy_step - Take an individual optimization step in the flow. This is done by calculating the
        log probability for the batch as defined by the flow, and then using Torchs autograd capability to automate
        the backprob through the model to reduce the loss. The step is also taken here.
        :return: float: the loss (the conditional density in this case) calculated.
        """

        # Put us in train mode, zero the optimizer and detach any parameters we will use.
        self.iter_proxy.train()
        self.optimizer_iter_proxy.zero_grad()
        _the_detached = self.detach_params(self.param_the)

        # To be redefined.
        _q_prob_exp = None
        _particles_init = None
        _noise_terms = None

        # Samples from p_inf(x), and then calculates the probability q_{phi}(z | x) as defined by the flow.
        try:
            if self.nf_n_train is None:
                # If we are generating data on the fly, generate some and find the iterated states.
                # This can be quite slow to run, as iterate will try _max_attempts times (currentl 2000).
                _particles_init = self.init_valid_particle(self.nf_batch_size)
                _iterated_particles, _noise_terms, _, _ = self.iterate(_particles_init.to(self.device), _the_detached)
            else:
                # If we are not generating data, we can just grab some from the premade training set.
                _idx = np.random.randint(0, len(self.train_x), self.nf_batch_size)
                _particles_init = self.train_x[_idx].clone().to(self.device)
                _noise_terms = self.train_n[_idx].clone().to(self.device)
        except Exception as e:
            util.echo_to_file(self.report_name, '\n[Util       ]: Error in util.take_proxy_step (1): ' + str(e))

        try:
            # Get the probability from the recognition flow.
            _q_prob_exp = - self.proxy_iter_model_prob(_noise_terms.to(self.device),
                                                       _particles_init.to(self.device)).mean(0)
        except Exception as e:
            util.echo_to_file(self.report_name, '\n[Util       ]: Error in util.take_proxy_step (2): ' + str(e))

        # Check to make sure that the gradient calculated is valid.
        if torch.isnan(_q_prob_exp) or torch.isinf(_q_prob_exp):
            util.echo_to_file(self.report_name, '\n[Util       ]: Warning: NaN or Inf, not applying update.')
        else:
            # Take gradient step.
            _q_prob_exp.backward()
            self.optimizer_iter_proxy.step()

        return _q_prob_exp


    ''' ------------------------------------------------------------------------------------ DEFINE BASE BEHAVIOURS. '''

    def init_valid_particle(self, _n=1):
        """
        AW - init_valid_particle - wraps calls to init_particle until _n valid particles have been initialized.
        :param _n:  int:        number of particles to initialize.
        :return:    tensor:     tensor of initialized particles.
        """
        state = torch.zeros((_n, self.nBalls, self.nDim), device=self.device, requires_grad=False)
        _n_initial = int(_n)
        _n_valid = 0
        _n_sim = 0
        while True:
            _n_sim += _n_initial
            _test_particles = self.init_particle(_n_initial)
            _test_bot = self.test_bot(_test_particles)
            _test_valid = _test_particles[(_test_bot == False).type(torch.bool), :]
            if len(_test_valid) > 0:
                if _n_valid + _test_valid.shape[0] >= _n:
                    state[_n_valid:] = _test_valid[:_n-_n_valid]
                    break
                else:
                    state[_n_valid:_n_valid + _test_valid.shape[0]] = _test_valid
                    _n_valid = _n_valid + _test_valid.shape[0]
                _n_initial = int(1.2 * (_n_sim / _n_valid) * (_n-_n_valid))
                _n_initial = np.min((np.max((_n-_n_valid, _n_initial)), _n))
                _n_initial = np.max((_n_initial, 1))
        return state

    @staticmethod
    def evaluate_log_prior(_s):
        """
        AW - evaluate_log_prior - evaluate the logarithm of the prior probability of the state, _s
        :param _s:  the state we wish to compute the log density of.
        :return:    log probability/density under the prior.
        """
        # Uniform prior over state as default. Bot defines regions that are not allowed under the prior.
        return torch.scalar_tensor(0.0)

    def noise_prior(self, _s, _param):
        """
        AW - noise_prior - sample from the apriori specified noise distribution.
        Currently fixed to be a zero-mean Gaussian distribution, independent of state, _s, with variance as defined
        in the argument _param.
        :param _s:      the state we are conditioning the noise on (normally independent).
        :param _param:  parameters of the noise distribution. Dictionary containing 'noise_sd'.
        :return:        the perturbation to be applied to the state.
        """
        _noise_mean = 0.0  # We assume a zero-mean noise distribution...
        _n_to_generate = _s.shape[0]
        _noise = torch.randn((_n_to_generate, self.nBalls, _param['noise_sd'].shape[0]), device=self.device)
        _noise_scaled = _noise * torch.exp(_param['noise_sd']) + _noise_mean
        return _noise_scaled

    def noise_density(self, _s, _n, _param):
        """
        AW - noise_density - Evaluate the density of a particular perturbation under the apriori specified model.
        :param _s:      the state on which the noise is being conditioned (normally independent).
        :param _n:      the value of the perturbation.
        :param _param:  the parameters of the noise distribution.
        :return:        density of the perturbation.
        """
        return util.torch_log_imultinormal_pdf(FloatTensor([[0.0] * _param['noise_sd'].shape[0]]),
                                               torch.exp(_param['noise_sd']).unsqueeze(0).unsqueeze(1),
                                               _n[:, :, self.noise_states])

    def get_noise(self, _s, _param, _q=None):
        """
        AW - get_noise - get the perturbation to state, using either the apriori specified noise distribution, or,
        the learned proposal if such as object is specified (though q).

        :param _s:      The state on which we are conditioning the noise. This is often disregarded if we are using
                        the apriori specified noise model.
        :param _param:  The parameters to use in the perturbation model (normally contains the noise variance).
        :param _q:      If _q is not None, use _q to sample the perturbations from the learned noise model.
        :return:        Sample from the noise distribution, either the apriori specified distribution or as generated
                        by a learned artifact.
        """

        if _q is None:
            _noise = self.noise_prior(_s, _param)
        else:
            _noise = _q(_s)
        return _noise

    def iterate(self, _state, _param=None, _deterministic=False, _q=None, _max_attempts=2000, _true_model=False):
        """
        AW - iterate - probably the most irritating complex function in this whole codebase...
        Iterate here is a wrapper for iterating the simulator, that takes into consideration how it should be called.
        Ultimately, _state is the simulator states that we wish to step forward. This stepping is done using the
        parameters held in the attribute self.param_the, unless _param is specified in which case the states can be
        iterated with a set of parameters of your choosing. One can choose to supress noise in the generativ model
        by setting deterministic to be true. This can be handy for generating `clean' data to perform SMC sweeps on.
        The origin of the perturbation that is applied to the state (if deterministic if False) is determined by _q.
        If this is not set, the apriori specified model is used, otherwise, _q must be a CNF of type ModelBase that can
        generate conditional samples.

        Here comes the magic sauce. This function will loop at most _max_attempts times to iterate the state.
        Therefore, by leaving _max_attempts unchanged, the rejection sampler will run for some time (i found 2,000 to be
        a reasonable ceiling) until _every_ particle has been successfully iterated. However, we are considering the
        case where we can only afford / want to simulate each particle only once. Therefore, at SMC time, this
        _max_attempts is set to one. This means the loop exits after the first attempt, regardless of how the noise was
        generated or if the particle failed. If the particle fails, is behaviour is defined by the implementation
        specified by the user (i set to NaNs often), but it is indicated by _bot being True for that sample.

        As soon as a particle has been iterated without failure, _still_bot will be set to false and that particle
        will not be iterated again. This is just to save computational resources when the simulator is expensive.

        _true_model is a lesser-used flag that essentially allows the behaviour of the simulator to be changed when
        generating data, to facilitate model misspecification at runtime. This flag must be used in the _iterate
        function of the user-specified simulator.

        :param _state:          tensor:     the latent states that we wish to implement through the simulator.
        :param _param:          tensor:     overwrite the parameters of the generative model.
        :param _deterministic:  bool:       set to true to suppress stochasticity in the generative model.
        :param _q:              {None, Model}: Leave as None for the apriori specified noise distribution to be used,
                                            otherwise set it to an instance of iter_proxy (CNF) for that to be used
                                            in the generative model instead.
        :param _max_attempts:   int:        limit the number of samples the rejection sampler can be used.
        :param _true_model:     bool:       set to True to switch the behaviour of the generative model to induce model
                                            misspecification.
        :return: tuple:     _iterated_states:       tensor:   tensor of the iterated states.
                            _noise:                 tensor:   tensor of the noise used to iterate _state(s).
                            _total_bots:            int:      the number of bots incurred for iterating all states.
                            _obs:                   tensor:   holds the observation if one was generated.
        """

        # If we are deterministic, the loop will be the same each time, so exit after one iteration.
        if _deterministic:
            _max_attempts = 1

        # Grab the parameters if we aren't overwriting them.
        if _param is None:
            _param = self.param_the

        # Define some holders.
        _state = _state.clone()  # Avoid side-effects.
        _n = _state.shape[0]  # Number of particles.
        _total_bots = 0  # Track the number of failed iterations.
        _still_bot = torch.arange(_n).to(self.device)  # Those particles that still require iteration.
        _attempts = 0  # Count up to a max of _max_attempts...

        # Define holders for the accepted noise, the noise we will test, the iterated state and the observation.
        _noise = torch.zeros((_n, self.nBalls, self.noise_states.shape[0]), device=self.device, dtype=torch.float)
        _noise_temp = torch.zeros((_n, self.nBalls, self.noise_states.shape[0]), device=self.device, dtype=torch.float)
        _iterated_states = float('nan') * FloatTensor(torch.ones_like(_state, device=self.device))
        _obs = float('nan') * FloatTensor(torch.zeros((_n, self.nBalls, self.observed_states.shape[0]),
                                                      device=self.device, dtype=torch.float))

        # Loop while there are still particles to iterate and we have not maxed out our attempts.
        while (len(_still_bot) > 0) and (_attempts < _max_attempts):

            # Sample the noise if we are not deterministic, otherwise, just preserve the state.
            if not _deterministic:
                _noise_temp = self.get_noise(_state[_still_bot], _param, _q)
                _noised_states = _state[_still_bot].clone()
                _noised_states[:, :, self.noise_states] += _noise_temp
            else:
                _noised_states = _state[_still_bot]

            # Iterate the (now) deterministic model.
            _is, _bot_new, _obs_new = self._iterate(_noised_states, _param, _true_model=_true_model)
            _total_bots += torch.sum(_bot_new.type(torch.uint8))

            # For any particles that passed, pull them out into the store and remove that particle from those
            # requiring iteration.
            if torch.any(_bot_new==False):
                _iterated_states[_still_bot[(_bot_new==False).type(torch.bool)]] = _is[(_bot_new==False).type(torch.bool)]
                _noise[_still_bot[(_bot_new==False).type(torch.bool)]] = _noise_temp[(_bot_new==False).type(torch.bool)]
                _obs[_still_bot[(_bot_new==False).type(torch.bool)]] = _obs_new[(_bot_new==False).type(torch.bool)]
                _still_bot = _still_bot[_bot_new.nonzero()][:, 0]

            # Increment the attempt counter and exit if we have maxed out.
            # Maxing out doesn't happen very often, and if it does, it normally signifies the limit is too low, or
            # something else is wrong. This isn't a critical failure, as bots can be handled downstream, and the
            # iterated state is set to NaN which causes the state to be rejected, but it is a bit untidy...
            _attempts += 1
            if _attempts == _max_attempts and _max_attempts > 1:
                util.echo_to_file(self.report_name, '[ModelBase  ]: WARNING - '
                                                    'max rejections reached, undefined behaviour incoming...')
                break

        return _iterated_states, _noise, _total_bots, _obs

    """ ---------------------------------------------------------------------------- DEFINE METHODS FOR NOISE PROXY. """

    def proxy_iter_model_sample(self, _x):
        """
        AW - proxy_iter_model_sample - sample from the CNF to get a valid perturbation (with high probability at least)
        conditional on the current state, _x.
        :param _x:  tensor:     state vector we wish to condition on.
        :return:    tensor:     perturbations sampled from the CNF.
        """
        _s = _x.shape  # Get and preserve the shape of the state.
        _n_ex = _s[0]  # Number of particles we are generating perturbations for.
        _xf = _x.reshape([-1, np.prod(_s[1:])])  # Flatten the state vector for input into the CNF.

        # We are going to chunk the input states into batches, where the max size of a batch is defined
        # by nf_max_batch_size. For large networks, state-space dimensionalities, or smaller GPUs i found that the CNF
        # code will regularly eat more memory than the GPU allows. this is just a hacky way of limiting the memory use.
        _idx = util.chunkIt(torch.arange(len(_xf)), np.ceil(float(len(_xf)) / self.nf_max_batch_size))
        _perturbations = []  # Going to append the sampled perturbations in here.
        for _i in _idx:
            _perturbations.append(self.iter_proxy.generate(_xf[_i].to(self.device)).detach())  # Sample from the flow.

        # Flatten the perturbations and reshape to be the same size as in the input state.
        return torch.cat(_perturbations, dim=0).reshape([_n_ex, self.nBalls, -1])

    def proxy_iter_model_prob(self, _noise_terms, _particles, _observations=None, _detach=False):
        """
        AW - proxy_iter_model_prob - for a particular perturbation-state pair, get the probability of the perturbation
        under the learned model. There is experimental functionality (but i havent gone through with it yet) to allow
        the CNF to also be conditioned on the observation (as per amortized inference). In this case, the observation
        is provided, flattened, and eppended to the state vector. As i say though, this is not really functional yet.
        :param _noise_terms:    tensor:     tensor of the noise terms we are evaluating the conditional density of.
        :param _particles:      tensor:     tensor of the conditioning states.
        :param _observations:   tensor:     NOT USED.
        :param _detach:         bool:       Can be handy to force the loss to be detached here, especially for long
                                            traces where you might accidentally let the computation graph build up.
        :return:                tensor:     the log conditional density (nats) of the perturbation.
        """
        _lp = []

        # Allow batching of the input states to limit the ceiling of the GPU memory used.
        _n_batch = int(np.ceil(float(len(_noise_terms)) / self.nf_max_batch_size))
        if _n_batch > 1:
            _particles_chunked = util.chunkIt(_particles, _n_batch)
            _noise_terms_chunked = util.chunkIt(_noise_terms, _n_batch)
            _state_flat = [torch.flatten(_p.clone(), start_dim=1) for _p in _particles_chunked]
            _noise_terms_flat = [torch.flatten(_p.clone(), start_dim=1) for _p in _noise_terms_chunked]
        else:
            _state_flat = [torch.flatten(_particles.clone(), start_dim=1)]
            _noise_terms_flat = [torch.flatten(_noise_terms.clone(), start_dim=1)]

        # # Currently commented out as i haven't proven the efficacy of using observations yet.
        # if _observations is not None:
        #     _observations_chunked = util.chunkIt(_observations, _n_batch)
        # else:
        #     _observations_chunked = None  # This None value should never get used...
        #                 _observations_flat = torch.flatten(_observations_chunked[_i], start_dim=1)
        #                 _state_flat = torch.cat((_state_flat, _observations_flat), dim=-1)

        # Loop over all of the batches, appending the log prob to the list as we go.
        for _i in range(len(_state_flat)):

            # Calculate the log prob.
            _loss = self.iter_proxy.log_prob(_noise_terms_flat[_i].to(self.device), _state_flat[_i].to(self.device))

            # If we are detaching, detach the loss.
            if _detach: _loss = _loss.detach().to('cpu')

            # Append it to the vector.
            _lp.append(_loss)

        # Flatten out the tensor and return!
        return torch.cat(_lp)

    """ ------------------------------------------------------------------------- WRAPPERS FOR GENERATING BULK DATA. """

    def simulate_trajectory(self, _i=None, _param=None, __plot=False, _initial_state=None, _deterministic=False,
                            _true_model=True, _rotate=False, _return_partial=False, _max_attempts=None, _q=None):
        """

        :param _i:
        :param _param:
        :param __plot:
        :param _initial_state:
        :param _deterministic:
        :param _true_model:
        :param _rotate:
        :param _return_partial:
        :param _max_attempts:
        :param _q:
        :return:
        """
        if _param is None:
            _param = dc(self.default_parameters)

        # Define the initial state.
        if _initial_state is not None:
            state = _initial_state.clone()
        else:
            state = self.init_valid_particle(_n=1)
        s_h = [state.clone().to('cpu').detach().numpy()]
        b_h = []
        n_h = []
        o_h = [self.obs_model(state, _param).clone().to('cpu').detach().numpy()]
        t_h = np.arange(0, self.smc_tMax+1)
        obs_bins = [0]
        for _t in t_h[1:]:

            _state_old = state.clone()
            if _max_attempts is None:
                state, noise, bots, obs = self.iterate(state, _param, _deterministic=_deterministic,
                                                       _true_model=_true_model, _q=_q)
            else:
                state, noise, bots, obs = self.iterate(state, _param, _deterministic=_deterministic,
                                                       _true_model=_true_model, _max_attempts=_max_attempts, _q=_q)

            if torch.any(torch.isnan(state)):
                # This indicates unilateral failure.
                if _return_partial:
                    break
                else:
                    return None

            b_h.append(bots.to('cpu').detach().numpy())
            n_h.append(noise.to('cpu').detach().numpy())
            s_h.append(state.to('cpu').detach().numpy())

            if _t % self.observation_frequency == 0:
                o_h.append(obs.clone().to('cpu').detach().numpy())
                obs_bins.append(_t)

            try:
                if not self.cluster:
                    for _ in range(10):
                        self.sim_viewer.render()
            except:
                pass

        # Convert to NP and save.
        s_h = np.squeeze(np.asarray(s_h))
        o_h = np.squeeze(np.asarray(o_h))
        n_h = np.squeeze(np.asarray(n_h))
        return {'states': s_h,
                'obs': o_h,
                'times': t_h,
                'obs_bins': np.asarray(obs_bins),
                'bots': b_h,
                'noise': n_h}

    def simulate_super_sampled_trajectory(self, _supersample=10, _initial_state=None, _deterministic=True):
        """
        AW - simulate_super_sampled_trajectory - for the purposes of generating data, having the simulator crash is
        quite annoying. It also means that the true data would never incur a crash which limits the data we will
        present to the algorithm. Therefore, this script allows one to generate a supersampled trajectory, whereby
        supersampled we mean the integration time is much smaller than usual (maybe a factor of ten), which normally
        erridates the failures. The generated data is then sub-sampled to give the illusion of failure-free true data.
        This wraps a call to simulate_trajectory while correctly modifying the model to implement the supersampling.
        :param _supersample:    float:      Factor by which to reduce the integration timestep.
        :param _initial_state:  tensor:     Tensor of the initial state if one wishes to force the initial state.
        :param _deterministic:  bool:       Use noise in the data generation.
        :return:                dict:       Dictionary of data, as formatted in simulate_trajectory.
        """
        # First we are going to modify the simulator such that it works at a smaller integration timestep.
        _tMax = self.smc_tMax
        self.dt /= _supersample
        self.smc_tMax *= _supersample
        self.observation_frequency *= _supersample
        self.param_the['noise_sd'] -= 0.5 * np.log(_supersample)

        # Try `attempts' times to generate the required data.
        data = None
        attempts = 10
        while data is None:
            data = self.simulate_trajectory(_initial_state=_initial_state, _deterministic=_deterministic)
            attempts -= 1
            if attempts == 0:
                raise RuntimeError  # Cant generate for some reason...

        # Undo the above modifications to the simulator to return it to 'normal'.
        self.dt *= _supersample
        self.smc_tMax /= int(_supersample)
        self.observation_frequency /= _supersample
        self.param_the['noise_sd'] += 0.5 * np.log(_supersample)

        # Now subsample the generated supersampled states.
        _n_state = len(data['states']) - 1
        _n_obs = len(data['obs']) - 1
        data['states'] = data['states'][np.linspace(0, _n_state, _tMax + 1).astype(np.int), :]
        data['times'] = data['times'][np.linspace(0, _n_state, _tMax + 1).astype(np.int)]
        data['obs_bins'] = (data['obs_bins'] / _supersample).astype(np.int)

        return data

    def generate_training_data(self):
        """
        AW - generate_training_data - function for generating a fixed training corpus but repeatedly calling
        iterate. The data is then inscribed to self, and also saved to a pickle for experimental control
        and also to be uses elsewhere if data generation is slow.
        :return: None (but data is returned as side effects and a pickle).
        """
        util.echo_to_file(self.report_name, '[ModelBase  ]: Generating training data.')
        st = timeit.default_timer()
        _the_detached = self.detach_params(self.param_the)

        # We can generate training data in batches to keep the memory footprint low.
        _n_per = util.chunkIt(range(self.nf_n_train), np.ceil(float(self.nf_n_train) / self.nf_max_batch_size))
        _x = []     # The state we are conditioning on.
        _n = []     # Perturbation noise.
        _y = []     # Observations (if these are generated).
        _xi = []    # Iterated state.
        bots = 0    # The number of bots we incur.
        try:
            for _i in _n_per:
                _particles_init = self.init_valid_particle(len(_i))
                _iterated_particles, _noise_terms, _bots, _obs = self.iterate(_particles_init, _the_detached)
                _x.append(_particles_init.detach().to('cpu'))
                _n.append(_noise_terms.detach().to('cpu'))
                _y.append(_obs.detach().to('cpu'))
                _xi.append(_iterated_particles.detach().to('cpu'))
                bots += _bots
            self.train_x = torch.cat(_x, dim=0)
            self.train_n = torch.cat(_n, dim=0)
            self.train_y = torch.cat(_y, dim=0)
            self.train_xi = torch.cat(_xi, dim=0)
        except Exception as e:
            util.echo_to_file(self.report_name, '[ModelBase  ]: Generation failed, using batched generation, '
                                                'using {} batches'.format(len(_n_per)))
            util.echo_to_file(self.report_name, '[ModelBase  ]: ' + str(e))

        elapsed = timeit.default_timer() - st
        util.echo_to_file(self.report_name, '[ModelBase  ]: Done generating training data. '
                                            '{} bots incurred for {} samples. {} seconds.'.
                          format(bots, self.nf_n_train, elapsed))

        if torch.any(torch.isnan(self.train_x)):
            util.echo_to_file(self.report_name, '\n[ModelBase  ]: ERROR Failure in training generation code, '
                                                'NaN incurred.')
            raise RuntimeError  # Generation error!

        # Now dump the dataset to a pickle.
        with open('./{}/{}TrainData.p'.format(self.folder_save_name, self.experiment_name), 'wb') as f:
            pickle.dump({'x': self.train_x.detach().to('cpu'),
                         'n': self.train_n.detach().to('cpu'),
                         'y': self.train_y.detach().to('cpu'),
                         'xi': self.train_xi.detach().to('cpu')}, f)

    def generate_validation_data(self):
        """
        AW - generate_validation_data - basically the same as generating training data, except we dont batch, since the
        validation dataset is often much smaller than the training dataset.
        :return: None (but data is returned as side effects and a pickle).
        """
        _the_detached = self.detach_params(self.param_the)
        _particles_init = self.init_valid_particle(self.nf_n_validate)
        _iterated_particles, _noise_terms, _, _obs = self.iterate(_particles_init, _the_detached)
        self.validation_x = _particles_init.detach().to('cpu')
        self.validation_n = _noise_terms.detach().to('cpu')
        self.validation_xi = _iterated_particles.detach().to('cpu')
        self.validation_y = _obs.detach().to('cpu')

        if torch.any(torch.isnan(self.validation_x)):
            util.echo_to_file(self.report_name, '\n[ModelBase  ]: ERROR Failure in validation generation code, '
                                                'NaN incurred.')
            raise RuntimeError  # Generation error!

        # Now dump the dataset.
        with open('./{}/{}ValData.p'.format(self.folder_save_name, self.experiment_name), 'wb') as f:
            pickle.dump({'x': self.validation_x.detach().to('cpu'),
                         'n': self.validation_n.detach().to('cpu'),
                         'y': self.validation_y.detach().to('cpu'),
                         'xi': self.validation_xi.detach().to('cpu')}, f)

    """ ----------------------------------------------------------------------------------------- DEFINE SMC METHOD. """

    def do_sweep(self, _data, _q=None, _add_legend=False, _initial_state=None):
        """
        AW - do_sweep - another slightly grizzly function...

        This function performs a single SMC sweep, generating the filtering distribution over state conditioned on the
        data provided in _data. _q allows the user to chose whether the default, apriori specified noise distribution
        is used, or, the learned q distribution is substituted in.

        :param _data:           dict:   data we wish to condition on, contains the observations ('obs'), the times
                                        at which those observations were recorded at ('obs_bins'), and the vector
                                        of timesteps that the simulator should be iterated ('times'). This is directly
                                        the format generated by simulate_(super_sampled)_trajectory.
        :param _q:
        :param _add_legend:     bool:   add a legend to the plots. only add a legend on the first iteration.
        :param _initial_state:  tensor: force the inital state of sweep.
        :return: dict:          'x_history':    the filtering distribution over state.
                                'b_history':    the time series of how many failures were incurred per obs.
                                'e_history':    time series of the likelihood (or the evidence over time).
                                'log_evidence': the total log evidence across data.
                                'time':         elapsed time in smc sweep.
                                'e_not_nan':    evidence of those particles that did not fail.
        """
        st = timeit.default_timer()

        # Grab the data we are going to condition on.
        y = FloatTensor(_data['obs'])

        # Initialize the particles with an importance sampling step.
        if _initial_state is None:
            x_0 = self.init_valid_particle(self.smc_particles * 10)
        else:
            x_0 = _initial_state

        # Perform the resampling step.
        pfr = pf.iterate(self,
                         x_0[:, :, self.observed_states],
                         y[0],
                         torch.exp(self.param_the['observation_noise_sd']),
                         _n_resamp=self.smc_particles)
        x_t = x_0[pfr['resampled_indices']].clone()

        # Define some holders for stuff.
        x_history = [x_t.to('cpu').detach().numpy()]
        e_history = []
        b_history = []
        l_history = []
        ess_history = []
        e_not_nan = 0.0
        next_obs = 1
        bots_per_bin = 0
        log_evidence = pfr['log_mean_weight']
        t = _data['times']

        # Iterate over the whole observed dataset, iterating using the desired noise model, but with only a single
        # call to the plant model. If there is an observation, we perform resampling and record the likelihood for
        # evidence computation.
        for _t in range(1, len(t)):
            x_t, _, _bot, _ = self.iterate(x_t, _q=_q, _max_attempts=1)  # Iterate the model using a SINGLE attempt.
            bots_per_bin += _bot  # Tot up the number of bots used.

            # If we have an observation at this time point, we can perform resampling.
            if _t in _data['obs_bins']:
                pfr = pf.iterate(self, x_t[:, :, self.observed_states], y[next_obs],
                                 torch.exp(self.param_the['observation_noise_sd']))

                # If the particle filter fails, it returns None, in which case we should return none from here.
                if pfr is None:
                    if not self.cluster: util.echo_to_file(self.report_name, '[ModelBase  ]: PF failing...!')
                    return {'x_history': float('nan'),
                            'b_history': float('nan'),
                            'e_history': float('nan'),
                            'l_history': float('nan'),
                            'ess_history': float('nan'),
                            'log_evidence': float('nan'),
                            'time': float('nan'),
                            'e_not_nan': float('nan')}

                # Do the resampling and append the likelihoods to the evidence computation.
                x_t = x_t[pfr['resampled_indices']].clone()
                log_evidence += pfr['log_mean_weight']
                e_history.append(pfr['log_mean_weight'].detach().to('cpu').numpy())
                e_not_nan += util.stable_mean(pfr['log_weights'][torch.isfinite(pfr['log_weights']).type(torch.bool)].
                                              detach().to('cpu').numpy())

                # Compute the normalized weights.
                _lw = np.asarray(pfr['log_weights'])
                _nw = np.exp(_lw) / np.sum(np.exp(_lw))
                l_history.append(_nw)
                ess_history.append(1.0 / np.sum(np.square(_nw)))

                # We need to move to the next observation.
                next_obs += 1

                # Zero the bots per observed bin.
                b_history.append(bots_per_bin.to('cpu').detach().numpy())
                bots_per_bin = 0

            # Append the iterated and/or resampled state to the history.
            x_history.append(x_t.to('cpu').detach().numpy())

        return {'x_history': np.asarray(x_history),
                'b_history': np.asarray(b_history),
                'e_history': np.asarray(e_history),
                'l_history': np.asarray(l_history),
                'ess_history': np.asarray(ess_history),
                'log_evidence': log_evidence.detach().to('cpu').item(),
                'time': timeit.default_timer() - st,
                'e_not_nan': e_not_nan}

    """ --------------------------------------------------------------------------------------- DEFINE MENIAL STUFF. """

    @staticmethod
    def detach_params(_params):
        """
        AW - detach_params - wrap calling detach for each parameter.
        :param _params: dictionary of parameters to be detached.
        :return: dictionary of detached parameters.
        """
        return {_k: _params[_k].detach() for _k in _params.keys()}
