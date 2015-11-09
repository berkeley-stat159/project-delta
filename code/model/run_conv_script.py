import numpy as np
import matplotlib.pyplot as plt
from convolution import predict_bold_signal

all_tr_times = np.arange(240)*2
neural_prediction, convolved = predict_bold_signal("./temp_data_for_testing/cond001.txt",3, 0.3, 2,240)
plt.plot(all_tr_times, convolved)
plt.plot(all_tr_times, neural_prediction)
