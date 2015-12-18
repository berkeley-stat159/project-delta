# Project Delta
## UC Berkeley Stat 159/259 Reproducible and Collaborative Statistical Data Science 
### Fall 2015 Project Delta

[![Build
Status](https://travis-ci.org/berkeley-stat159/project-delta.svg?branch=master)](https://travis-ci.org/berkeley-stat159/project-delta?branch=master)
[![Coverage
Status](https://coveralls.io/repos/berkeley-stat159/project-delta/badge.svg?branch=master)](https://coveralls.io/r/berkeley-stat159/project-delta?branch=master)

_**Topic:**_ The Neural and Behaviroal Analysis of Loss Aversion in Decision-Making of Under Risk 
_**Group members:**_ 
Victor Kong (`VictorKong94`), 
Celi(`karenli`), 
Anna Liu(`liuanna`), 
Yunfei Xia(`yfxia`), 
Weidong Qin(`j170382276`)

This project is built upon Sabrina M. Tom, Craig R. Fox, Christopher Trepel, Russell A. Poldrack's published work on SCIENCE, VOL 315 26 JANUARY 2007. 
The aim is to reproduce the fMRI brain analysis results, verify the behavioral assumptions in the 50-50 chance gain/loss gamble and perform conjunction analysis on neural and behavioral studies. 

## Installation Guideline

###1. Projection File Directory 

All utilities can be used from the main project directory (right here!) by
calling a set of commands from your terminal or command prompt.  

- `code`: Contain a complete set of functionalities for analysis 

- `data`: Contain mainly components of the reposity that have to do with the dataset of interest

- `paper`: Contain functionalities for written report

- `slides`: Contain functionalities of slides for presentation

###2. Data Acquisition
- `make dataset`: Download and upzip ds005 file from openfMRI website. It at least 17 gigabytes of storage (~2 hrs if Internet Connection good) 

- `make validate-data`：Verify ds005 file is correct and complete
 

###3. Statistical Analysis 

**Import Notice**: You may need to `pip install nilearn` package for the following command 
- `make analyses`: Perform a complete set of preprosessed data with neural, behavioral and conjunction analysis for all 16 subjects at once. (~2 hrs for completion). 

- `make smoothing`: Smooth the retrieved data with a Gaussian Kernel

- `make convolution`: Perform convolution method on 16 subject

- `make glm`(Prerequisite: Convolution): Perform generalized Linear Model 

- `make diagnosis`: Perform model diagnostics for neural fMRI analysis

- `make visualization` (Prerequisite: GLM): Generate neuroimages on fMRI run using nilearn package

- `make logistic`: Perform logistic regression on behavioral data 

- `make conjunction` (Prerequisites: GLM, Logistic): Perform conjunction anaysis of neural and behavioral data

- `make rm-results`: Remove *all* analysis results

###4. Create Written Report

- `make report`: Produce a copy of our findings, will be saved as a pdf file in the paper/ directory

- `make rm-report`: Delete the report

-`make clean`: Remove cache, preference, and other unnecessary files.

###5. Other Utilites

- `make test-data`: Generate a set of dummy data that can be used to check the build's integrity

- `make test`: perform the above checks

- `make rm-test-data`: Delete the dummy data


## Acknowledgements

We would like to give big thanks Jarrod Millman, Matthew Brett, J-B Poline, and Ross Barnowski for their extraordinary support and advice throughout the completion process of this project. 


