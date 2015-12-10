This directory contains scripts that automate statistical analyses, as well as
production of processed data and graphical figures. Scripts include:
- `convolution`: Contains code to compute convolved hemodynamic response
  function predictions for three conditions given in the original data:
  *parametric gain*, *parametric loss*, and *distance from indifference*.
  Generates one figure of six plots and one plaintext file for each condition.
- `diagnosis`: Contains code to fit gseneralized linear model on the fMRI data,
  remove outliers with respect to volume standard deviation, and produce plots
  that highlight areas of task-dependent activation in the brain.
- `linear_analysis`: Contains code to perform enhanced diagnoses of the data,
  using smoothed data, skipping the dropping of outliers, and adding linear and
  quadratic drift components to the generalized linear model approach.
- `logistic`: Contains code to fit logistic regression models to predict subject
  response using three regressors: *parametric gain*, *parametric loss*, and the
  *euclidean distance* of the gain/loss combination from the diagonal of the
  gain/loss matrix.
- `smoothing`: Contains code to apply smoothing with a Gaussian kernel in bulk
  to the raw and filtered BOLD signal data. Generates four figures, two each for
  the raw and filtered data, one of the original image and one of the smoothed
  image.