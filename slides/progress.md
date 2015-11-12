% Project Delta Progress Report
% Victor Kong, Ce Li, Anna Liu, Weidong Qin, Yunfei Xia
% November 9, 2015


# Background

## The Paper

- from OpenFMRI.org (ds005)
- "The Neural Basis of Loss Aversion in Decision-Making Under Risk"
  - by Sabrina M. Tom et al.

## The Data

- 16 subjects, 1 task per subject, 3 runs per task
- Examination of the neural systems that process decision utility with fMRI data
- Task:
  - Subjects offered wager with 50/50 chance of winnning
  - Varying potential gains/losses
  - Decision to accept or decline

# Completing/In Progress

## Data Fetching and Preprocessing 

- Downloading data from OpenFRMI.org and decompressing it
- Plotting data to explore potential useful information for our project
- Drawing summary statistics from the plotted data

## Initial Analysis

- Convolution
- Smoothing
- Linear Regression
  - Multiple and single regression with stimulus
- Hypothesis Testing
  - General t-tests
- Time Series
- PCA

# Our Plan

## Goal

- Trying to reproduce methods as well as adding our own thoughts into it
  - Using other methods that may or may not come to the same conclusion

## Analysis to Perform

- Logistic regression
- Time series 
- Hypothesis tests
- Correlation analysis
- Robust regression analysis
- PCA
- Support vector machines
  - Machine learning algorithm
    - Process: draw boundaries between clusters of data
  - Plan to use to classify parts of the brain
    - What parts (de)activate most when making decisions?
    - What parts are more active when given a good/bad/obvious/etc. wager?
    - Are these parts the same, or different?
- Decision trees
  - Machine learning algorithm
    - Analyze input variables consecutively
    - Models human decision-making very well
  - Plan to analyze brain activity and decision-making multiple ways
    - What conclusions result from combinations of these parts activating?
    - What conclusions result from combinations of parametric gains/losses?
    - What parts activate given combinations of parametric gains/losses?

## Simplification Steps

## Issues We have Discussed

## Methods of validating models

- t-tests
- RSS
- Cross-validation


# Our Process

## Hardest Parts of the Project

- Size of data
  - Spent a lot of time trying to figure out format
  - What we need and don't need to look at
- Writing tests for functions
  - Lack of small piece of data that we know all about
  - Can improvise for simple functions only

## Issues With Working as a Team

- Difficult for all to meet together
- Different styles for writing and documenting code
- Difficult to communicate what we want to do
  - Don't tell each other what we're planning to do
    - Merge conflicts all day, er'day
- Organizing GitHub repository

## Most Useful Parts of Class

- Linear Modelling
- Correlation per voxel

## Least Helpful Parts of Class

- Comparison to R
- Mathematical Writing

# Our Process (cont'd)

## What We Need to Accomplish in the Project

## Difficulty in Making Work Reproducible

# Potential Topics to Cover in Future

- More linear regression (ANOVA), PCA
- Machine learning (classification, prediction, cross-validation)
- Permutation tests (bootstrap)
- Software tools (Git, Python)
