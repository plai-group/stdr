# MIT License
#
# Copyright (c) 2017, Andrew Warrington.
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

"""
particleFilter.py
AW

TL;DR -- This module contains the necessary functions for adding particle-filter
like behaviours to the SimpleCElegans script.

This script contains the basic functionality for performing (sequential) importance
sampling. The core function is the `iterate' function. This function takes a
vector of particles, an observation, and the standard deviation of this observation
(under the observation model) and resamples the particles according to their
likelihood. This function, in conjunction with a plant model provided outside of
this script, allows you to write a particle filter.

The key `flaw' in this script is that it assumes the observation is zero mean error
about the true state. If the observation function is more complex, then this will
need to be updated. This assumption was made to make the code easier to use.
The permutation matrix must also be provided, that maps states onto observations.
"""

# Import modules.
import numpy as np
import torch
import time
from copy import deepcopy as dc
import math

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def iterate(_settings, _particles, _observation, _observation_sd, _n_resamp=None):
	'''
	particleFilter.iterate:
	Function takes in the current particles as an NxM length vector (where N is
	the number of particles and M is the dimensionality) of the state and a
	single observation with dimensionality Hx1 (where H is the dimensionality
	of the observation.
	Assumes the observations are normally distributed about the true value.
	
	:param _particles: NxM length vector of particles
	:param _observation: single observation.
	:param _observation_sd: positive float containing the standard deviation of the observation.
	:param _new_particle_count:  Default: None, how many particles to resample.
	:return: Dictionary:
				{
					'log_weights':          the log weight of each of the input
											N particles.
					'resampled_particles':  the vector of newParticleCount
											resampled particle indicies.
				}
	'''

	# Retrieve the number of particles, dimensionality of state and dimensionality
	# of the observation.
	N = _particles.shape[0]
	
	# Are we subsampling?
	if _n_resamp is None:
		_n_resamp = N
		
	# Calculate the log probability of each particle under a Gaussian observation
	# model.
	_log_weights = torch.sum(norm_log_pdf(_particles, _observation, _observation_sd), dim=-1)

	# Remove any nans, replace them with -inf.
	_log_weights[torch.isnan(_log_weights)] = -math.inf

	# Make the weights zero-mean to improve the numerical stability.
	zeroed_weights = _log_weights - torch.max(_log_weights)
	zeroed_weights = torch.exp(zeroed_weights)
	zeroed_weights_sum = torch.sum(zeroed_weights)
	zeroed_weights = zeroed_weights / zeroed_weights_sum
	
	# If we are resampling the same number of particles, we can use TuanAnhLes
	# fast systematic resampling code.
	uniforms = torch.rand([1]).to(_settings.device) / _n_resamp + torch.linspace(0, _n_resamp-1, steps=_n_resamp, dtype=torch.float).to(_settings.device) / float(_n_resamp)
	resampled_indexes = np.digitize(uniforms.to('cpu').numpy(), bins=np.cumsum(zeroed_weights.to('cpu').numpy()))

	# The resample will return the final element if the final two are too similar, so
	# if that happens use a multinomial resampling.
	if np.max(resampled_indexes) == np.alen(zeroed_weights):
		# print('Warning -- Bailing to choice resampling.')
		# Make the weights zero-mean to improve the numerical stability.
		zeroed_weights = zeroed_weights - torch.max(zeroed_weights)
		zeroed_weights = torch.exp(zeroed_weights)
		zeroed_weights_sum = torch.sum(zeroed_weights)
		zeroed_weights = zeroed_weights / zeroed_weights_sum
		zeroed_weights = zeroed_weights.to('cpu').detach().numpy()
		zeroed_weights[np.isnan(zeroed_weights)] = 0.0
		try:
			resampled_indexes = np.random.choice(N, N, p=zeroed_weights, replace=True)
		except:
			# print('resampling failed.')
			return None

	# AW - stabiliser is correctl.
	stabilizer = torch.max(_log_weights)
	log_mean_weight = stabilizer + torch.log(torch.sum(torch.exp(_log_weights - stabilizer))) - np.log(N)

	# Check if the particle filter failed.
	if torch.isnan(log_mean_weight):
		failed = True
	else:
		failed = False

	return {'log_weights': _log_weights,
			'resampled_indices': resampled_indexes,
			'log_mean_weight': log_mean_weight,
			'failed': failed}
	

def norm_log_pdf(x, loc=0, sd=1):
	'''
	particleFilter.normpdf:
	Calculate the probability density for a set of particles, given the
	normal distribution.

	:param x: Input particles.
	:param loc: Mean of normal distribution.
	:param sd: Standard deviation of normal distribution.
	:return: Vector of log-probabilities.
	'''

	d = x.shape[1]
	ll = - ((d/2.0) * np.log(2*math.pi)) \
		 - (d * torch.log(sd)) \
		 - (torch.sum((1 / (2.0 * (sd**2))) * (x - loc)**2, dim=1))
	return ll


def smooth(filtering, ancestors):
	'''
	particleFilter.smooth:
	Traces back through the particle history to create the
	smoothing distribution from the filtering distribution
	and ancestry.
	:param filtering:
	:param ancestors:
	:return:
	'''

	obs_i = np.alen(ancestors)
	nParticles = np.alen(filtering[0, :, 0])
	smoothing = np.zeros_like(filtering)

	for k in np.arange(nParticles):
		next_ancestor = k
		smoothing[obs_i - 1, k, :] = dc(filtering[obs_i - 1, k, :])

		for j in np.arange(np.alen(ancestors)):
			next_ancestor = int(ancestors[obs_i - j - 1, next_ancestor])
			smoothing[(obs_i - j - 1 - 1):(obs_i - j - 1), k, :] = dc(filtering[(obs_i - j - 1 - 1):(obs_i - j - 1), next_ancestor, :])

	return smoothing


if __name__ == "__main__":
	'''
	Define a main function for demonstrative purposes.
	'''
	print('Particle filter demonstration.')
	start = time.time()
	
	steps = 100
	observations = np.zeros((steps, 1))
	states = np.zeros((steps, 2))
	states[0, 0] = np.random.normal(0, 1)
	states[0, 1] = np.random.normal(1, 0.1)

	for i in range(steps):
		if i > 1:
			velocity = np.random.normal(states[0, 1], 0.1)
			states[i, 0] = states[i-1, 0] + velocity
		observations[i] = np.random.normal(states[i, 0], 0.5)
		
	particles = np.random.rand(500, 2)
	
	state_estimate = np.zeros((steps, 2))
	
	for i in range(0, steps):
		# Iterate the plant model.
		velocities = np.random.normal(particles[:, 1], 0.1)
		particles[:, 1] = velocities
		particles[:, 0] = particles[:, 0] + velocities
		p = 0
		
		# Do the re-sampling step.
		it = iterate(np.expand_dims(particles[:, 0], axis=1), observations[i], 0.5)
		particles = particles[it['resampled_indices'], :]
		log_weights = it['log_weights']
		state_estimate[i, :] = np.mean(particles, 0)
	
	end = time.time()
	print(end - start)
		
	# # Plot some stuff.
	# plt.plot(state_estimate[:, 0])
	# plt.plot(observations)
	# plt.pause(0.001)
	#
	print('test complete.')
