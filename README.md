# Coping with simulators that don't always return.

Source code for the paper Warrington, A., Naderiparizi, S., & Wood, F. (2019). Coping with simulators that don't always return. In _The 23nd International Conference on Artificial Intelligence and Statistics_ (_to appear_).

**Code to follow soon.**

## TL;DR
This work tackles the scenario where a simulator does not return a value (crashes) for particular inputs, the impact of this on pseudomarginal methods and how this can be mitigated using a conditional normalizing flow.

## Paper Abstract
Deterministic models are approximations of reality that are easy to interpret and often easier to build than stochastic alternatives. Unfortunately, as nature is capricious, observational data can never be fully explained by deterministic models in practice. Observation and process noise need to be added to adapt deterministic models to behave stochastically, such that they are capable of explaining and extrapolating from noisy data. We investigate and address computational inefficiencies that arise from adding process noise to deterministic simulators that fail to return for certain inputs – a property we describe as “brittle.” This paper is a step towards a novel Bayesian methodology for performing inference in such models. We show performing inference in this space can be viewed as rejection sampling and train a conditional normalizing flow that is capable of proposing noise values such that the sample is accepted with high probability, increasing computational efficiency.
 
## General Gist
Without going into too much detail here (see XXX for more detailed infomation), the purpose of our work here is to mitigate against the case when a simulator might fail, an eventually we denote using ⊥ (\bot in LaTeX, read 'bot' or 'bottom'). We consider specifically the case where an otherwise deterministic time series simulator (or state-space model) is converted to be a stochastic model by simply perturbing the state at each time point such that one can use it in a probabilistic inference tool such as sequential Monte Carlo (SMC).

<img src="https://github.com/plai-group/stdr/blob/master/docs/figures/rs_p.jpg" width="300">

The above figure is a simple description of the process. The deterministic simulator is denoted as `f` and is applied as a deterministic function to the state `x_{t-1}`. However, it is additively perturbed by an amount, denoted `z_t`, drawn from some user-specified distribution `p(z_t | x_{t-1})`. While we provision this being conditioned on current state, it often is not, and is chosen to be Gaussian distributed noise with heuristically determined variance. The perturbed state is then passed through `f`. If the simulator fails, the loop is repeated with a new `z_t` drawn. If the simulator does not fail, the process exits. In this manner, this resembles a rejection sampler with hard rejections. 

The rejected samples represent wasted computational resources, especially in domains or operating regimes where the rejection rate is high. We may also be operating with a simulator that is expensive and hence we do not want to waste computation, and may only be able to afford a _single_ sample of `z_t`. If the sample fails, then that particle (in the SMC/particle filter sweep) is simply removed. This reduces the effective sample size and increases the variance of any summary statistic computed from the resulting distributions.

<img src="https://github.com/plai-group/stdr/blob/master/docs/figures/rs_q.jpg" width="300">

Therefore, we modify the original algorithm by replacing `p(z_t | x_{t-1})` with a learned object `q_{\phi}(z_t | x_{t-1})`, where this distribution is trained such that no rejections are incurred. To do this, we use evidence maximization of the learned density on perturbation-state pairs that we know to be successfully integrable. As such, we eliminate the rejection sampling loop, resulting in much higher effective sample sizes and lower wasted computation. (Obviously, `q` will not be perfect, and so this rejection sampling loop is still provisioned for, but for illustrations sake this is what we are striving towards.)

We parameterize `q_{\phi}` using an masked autoregressive flow [G. Papamakarios et al., 2017], such that the resulting density remains evaluable. This is desirable as one might wish to interrogate the accept-reject characteristics of the simulator, a task that is much easier exploiting the favourable characteristics of the autoregressive flow. 

This work is about the implementation of this principal, and analysis and demonstration of it on several examples. 

## Repo Layout

### `src`
Contains all the source code files for running experiments and reproducing figures.

### `docs` 
Contains the LaTeX source code and figures for the most recent revision of the paper.   Also contains the documentation and help files for running experiments and exploring the code base.

### `misc` 
Contains a number of scripts for configuring the repository and cloning items not included in the repo (data, logs etc). 


