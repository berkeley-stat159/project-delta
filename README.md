# Project Delta
## UC Berkeley Stat 159/259 Reproducible and Collaborative Statistical Data Science 
### Fall 2015 Project Delta

[![Build
Status](https://travis-ci.org/berkeley-stat159/project-delta.svg?branch=master)](https://travis-ci.org/berkeley-stat159/project-delta?branch=master)
[![Coverage
Status](https://coveralls.io/repos/berkeley-stat159/project-delta/badge.svg?branch=master)](https://coveralls.io/r/berkeley-stat159/project-delta?branch=master)

_**Topic:**_ The Neural and Behavioral Analysis of Loss Aversion in Decision-Making of Under Risk 

_**Group members:**_ 
- Victor Kong (`VictorKong94`), 
- Ce Li (`karenli`), 
- Anna Liu (`liuanna`), 
- Yunfei Xia (`yfxia`), 
- Weidong Qin (`j170382276`)

This project builds upon Sabrina M. Tom, Craig R. Fox, Christopher Trepel,
Russell A. Poldrack's published work of the same name in Science, Volume 315
from 26 January 2007. The aim is to reproduce the fMRI brain analysis results
detailed in this study, verify the behavioral findings in the 50-50 chance
gamble, and perform conjunction analysis on neural and behavioral data.

## Installation Guidelines

###1. Projection File Directory 

All utilities can be used from the main project directory (right here!) by
calling a set of commands from your terminal or command prompt.  

Contents:

- `code`: A complete set of functionals for analysis.

- `data`: Components of the repository that have to do with the dataset of
  interest.

- `paper`: Components of our written report.

- `slides`: Components of slides for presentation.

###2. Data Acquisition

The following commends can be performed to obtain the data set used for analysis:

- `make dataset`: Download and upzip ds005 file from openFMRI website. Be sure
  to have at least 17 gigabytes of storage space on your hard drive.

- `make validate-data`：Verify the dataset is correct and complete.
 

###3. Statistical Analysis 

**Import Notice**: Be sure you have installed all necessary modules listed in
the requirements plaintext file. Do this by calling `pip install module_name`.  
Because some of these analyses were done in parallel, they can be run either
altogether or individually. The commands to run them are:  

- `make analyses`: Performs the complete set of neural, behavioral, and conjunction analyses as we did them.

- `make diagnosis`: Perform model diagnostics for neural fMRI analysis.

- `make smoothing`: Smooth the retrieved data with a Gaussian kernel.

- `make convolution`: Produce convolved time courses.

- `make glm` (Prerequisite: `convolution`): Fit generalized linear models.

- `make visualization` (Prerequisite: `glm`): Generate neuroimages on fMRI runs
  using the `nilearn` Python module.

- `make logistic`: Perform logistic regression on behavioral data.

- `make conjunction` (Prerequisites: `glm`, `logistic`): Perform conjunction
  analyses on neural and behavioral data.

- `make rm-results`: Remove *all* analysis results.

###4. Create Written Report


- `make report`: Produces a written copy of our findings, and saves it as a pdf
  in the paper/ directory.

- `make rm-report`: Delete the report.

###5. Other Utilites

- `make test-data`: Generate a set of dummy data that can be used to check the
  build's integrity.

- `make test`: Test the integrity of functions contained in the code/utils
  directory.

- `make rm-test-data`: Delete the dummy data.

- `make clean`: Remove cache, preference, and other unnecessary files.


## Acknowledgements

We would like to give big thanks to Jarrod Millman, Matthew Brett, JB
Poline, and Ross Barnowski for their extraordinary support and advice throughout
the process of this project.
