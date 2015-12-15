This directory contains prewritten functions that can be imported into and used
in subsequent scripts. These utilities include:
- `stat_utils`: Contains functions that are useful in many common statistical
  analyses.
- `diagnostics`: Contains a collection of utility functions to perform
  diagnostics on fMRI data.
- `hrf`: Contains a function that computes the canonical hemodynamic response
  function signals at specified time points, giving the most basic estimate of
  the neural time course.
- `hypothesis`: Contains code to assess the statistical significance of results
  returned from generation of predictive models.
- `make_class`: Contains code to set up two classes that perform all the grunt
  work necessary to set up the Python environment necessary for quick, easy, and
  efficient data analysis of fMRI data.
- `make_test_data`: Contains code to create a complete set of dummy data to be
  used to assess the integrity of other utilities. Said data is saved to the
  `data/ds005/subtest/` directory.
- `plot_tool`: Contains code that helps to facilitate and standardize the
  formulation of graphical figures.
