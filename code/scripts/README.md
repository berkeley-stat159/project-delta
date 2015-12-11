This directory contains scripts that automate statistical analyses, as well as
production of processed data and graphical figures. Scripts include:
- `convolution`: Contains code to compute convolved hemodynamic response
  function predictions for three conditions given in the original data:
  *parametric gain*, *parametric loss*, and *distance from indifference*.
  Generates one figure of six plots and one plaintext file for each condition.
- `diagnosis`: Contains code to find, inspect, and remove outliers with respect
  to volume standard deviation.
  that highlight areas of task-dependent activation in the brain.
- `glm`: Contains code to fit and assess the use of a generalized linear model
  on the smoothed data. This follows the `convolution` and `diagnosis`, as one
  can see from the choice to skip dropping outliers.
- `logistic`: Contains code to fit logistic regression models to predict subject
  response using three regressors: *parametric gain*, *parametric loss*, and the
  *euclidean distance* of the gain/loss combination from the diagonal of the
  gain/loss matrix.
- `smoothing`: Contains code to apply smoothing with a Gaussian kernel in bulk
  to the raw and filtered BOLD signal data. Generates four figures, two each for
  the raw and filtered data, one of the original image and one of the smoothed
  image.