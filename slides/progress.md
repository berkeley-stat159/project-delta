% Project Delta Progress Report
% Victor Kong, Ce Li, Anna Liu, Weidong Qin, Yunfei Xia
% November 12, 2015


# BACKGROUND

## The Paper

- From OpenFMRI.org (ds005)
- "The Neural Basis of Loss Aversion in Decision-Making Under Risk"
  - by Sabrina M. Tom et al. (2007) in Science

## The Data

- 16 subjects, 1 task per subject, 3 runs per task
- Examination of the neural systems that process decision utility with fMRI data
- Task:
  - Subjects offered 50/50 wager
  - Varying potential gains/losses
  - Prompted for decision to accept or decline

# COMPLETING/IN PROGRESS

## Data Fetching and Preprocessing 

- Download from OpenFRMI.org and decompress
- Plot to explore potentially useful information
- Drawing summary statistics from the plotted data
- Smoothing periodic noise

## Initial Analysis

- Hypothesis Testing
- Convolution
- Logistic Regression
- Linear Regression
  - Multiple and single regression with stimulus

# OUR PLAN

## Goal

- To reproduce methods as well as adding our own thoughts into it
- Using other methods that may or may not come to the same conclusion

## Methods and Analysis to Perform

- Hypothesis tests
- Linear regression, Logistic regression, Correlation analysis
- Robust regression analysis, Principle component analysis

# OUR PLAN

## Methods and Analysis to Perform(cont'd)

- Support vector machines
  - Process: draw boundaries between clusters
  - Classify parts of the brain
    - What parts (de)activate most when making decisions?
    - What parts are active given a good/bad/obvious/etc. wager?
    - Are these parts the same or different?
- Decision trees
  - Process: analyze inputs consecutively
    - Models human decision-making well
  - MANY questions:
    - What results from combinations of parts activating?
    - What results from combinations of gains/losses?
    - What parts activate given combinations of gains/losses?

# Our PLAN

- Simplification Steps
- Issues We have Discussed
- Methods of validating models
  - t-tests
  - RSS
  - Cross-validation


# OUR PROCESS

## Most Difficult Parts of the Project

- Size of data
  - Spent much time deciphering format
  - What we need and don't need to look at
- Writing tests for functions
  - Lack of small piece of data that we know all about
  - Can improvise for simple functions only

## Issues Working as a Team

- Difficult for all to meet together
- Different styles of coding and documenting
- Difficult to communicate what we want to do
  - Don't tell each other what we plan to do
- Organizing GitHub repository

# OUR PROCESS

## Most Useful Parts of Class

- Linear modelling
- Correlation per voxel

## Least Helpful Parts of Class

- Comparison to R
- Mathematical Writing

## What We Need to Accomplish in the Project

# Potential Topics to Cover in Future

- More linear regression, ANOVA, Principle component analysis
- Machine learning (classification, prediction, cross-validation)
- Permutation tests (bootstrap)
- Software tools (Git, Python)
