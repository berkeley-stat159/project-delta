# project-template
[![Build
Status](https://travis-ci.org/berkeley-stat159/project-delta.svg?branch=master)](https://travis-ci.org/berkeley-stat159/project-delta?branch=master)
[![Coverage
Status](https://coveralls.io/repos/berkeley-stat159/project-delta/badge.svg?branch=master)](https://coveralls.io/r/berkeley-stat159/project-delta?branch=master)

Fall 2015 group project delta

Team Members:
Victor Kong,
Ce Li,
Anna Liu,
Weidong Qin, and
Yunfei Xia

All utilities can be used from the main project directory (right here!) by
calling a set of commands from your terminal or command prompt. Before any
analyses can be done, you must first download the dataset. Do this by calling
`make dataset`. Be sure to have at least 17 gigabytes of storage space available
on your hard drive before doing this. Needless to say, if you are on a slower
connection, you should also be prepared to wait a while. Once the process has
been completed, you can verify that you have the correct and complete data set
by calling `make validate-data`.

Once you have the data on your hard drive, you are ready to reproduce our
experimental process. Because some of these statistical analyses were done in
parallel, they can either be run separately or altogether. Be wary however that
some have prerequisites. Commands pertaining to analyses are listed here:  
- Perform *all* analyses: `make analyses` (Our process)
- Conjunction: `make conjunction` (Prerequisites: GLM, Logistic)
- Convolution: `make convolution`
- Diagnosis: `make diagnosis`
- Generalized Linear Model: `make glm` (Prerequisite: Convolution)
- Logistic Regression: `make logistic`
- Smoothing with a Gaussian Kernel: `make smoothing`
- Remove *all* analysis results: `make rm-results`

You can also produce a copy of our findings by calling `make report`, which will
be saved as a pdf file in the paper/ directory.

Included with this package are also some general purpose utilities that may come
in handy. Calling `make test-data` generates a set of dummy data that can be
used to check the build's integrity. Once this dummy data has been generated,
call `make test` to perform these checks. When you are happy with the results,
call `make rm-test-data` to delete the dummy data. Lastly, if you ever find the
project directory to be more cluttered than you would like, call `make clean` to
remove cache, preference, and other unnecessary files.
