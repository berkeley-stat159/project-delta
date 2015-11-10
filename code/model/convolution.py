"""
Convolving predicted neural time course with the hemodynamic response function
Getting predicted BOLD signals

"""
from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.stats import gamma
from stimuli import events2neural

def hrf(times):
	""" Return values for HRF at given times """
	# Gamma pdf for the peak
	peak_values = gamma.pdf(times, 6)
	# Gamma pdf for the undershoot
	undershoot_values = gamma.pdf(times, 12)
	# Combine them
	values = peak_values - 0.35 * undershoot_values
	# Scale max to 0.6
	return values / np.max(values) * 0.6

def predict_bold_signal(filename, duration, by, TR, n_vols):
	"""return neural prediction and predicted BOLD signal """
	tr_times = np.arange(0, duration, by)
	hrf_at_trs = hrf(tr_times)
	neural_prediction = events2neural(filename, TR, n_vols)
	convolved = np.convolve(neural_prediction, hrf_at_trs)
	assert len(neural_prediction)+len(hrf_at_trs)-1==len(convolved)
	n_to_remove=len(hrf_at_trs)-1
	convolved = convolved[:-n_to_remove]
	return neural_prediction, convolved

