"""
Convolving predicted neural time course with the hemodynamic response function
Getting predicted BOLD signals

Notes:
Since the duration of the task is 3 seconds and the onsets are not equally spaced, we have to
mannually calcualte convolution instead of using builtin np.convolve

"""
from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.stats import gamma
from stimuli import events2neural

def hrf(times):
	""" Return values for canonical HRF at given times 
	
	Parameter:
	---------
	times: array
		an array of times points

	Return:
	------
	an array (len(times),)
		hemodynamic response
	"""
	# Gamma pdf for the peak
	peak_values = gamma.pdf(times, 6)
	# Gamma pdf for the undershoot
	undershoot_values = gamma.pdf(times, 12)
	# Combine them
	values = peak_values - 0.35 * undershoot_values
	# Scale max to 0.6
	return values / np.max(values) * 0.6

def convolve(neural, TR, n_vols, duration):
	"""return convolved BOLD signal 

	Parameter:
	---------
	neural: neural prediction returned by events2neural
	TR: Time to repetition
	n_vols: number of volumns
	duration: original duration / TR

	Return:
	-------
	convolved: an array (n_vols,)
		convolution

	Note:
	----
	The hemodynamic response usually lasts for 30 seconds. However, since our duration is 3 seconds, 
	we need to conside the effect when hemodynamic responses overlap.

	"""
	times = np.arange(n_vols)
	on_time = times[neural!=0]
	tr_times = np.arange(0, 30, TR)
	N = n_vols
	M = len(tr_times)
	convolved = np.zeros(N+M-1)
	dur = np.arange(duration,dtype=int)
	for i in range(len(on_time)):
		for j in dur:
			convolved[on_time[i]+j:(on_time[i]+j+len(tr_times))] += hrf(tr_times) * neural[on_time[i]]
	n_to_remove = M-1
	convolved = convolved[:-n_to_remove]
	return convolved
