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
`make dataset`. If you are on a slower connection, you should be prepared to
wait. You can verify that you have the correct and complete data set by calling
`make validate-data`.

The analyses can eitherbe run altogether or separately. To perform all analyses
written, call `make analyses`. The commands to perform individual analyses is
listed below:  
- Convolution: `make convolution`
- Logistic Regression: `make logistic`
- Smoothing with a Gaussian Kernel: `make smoothing`  
The analysis results can also be removed by calling `make rm-results`.

Included with this package are also some general purpose utilities that may come
in handy. Calling `make test-data` generates a set of dummy data that can be
used to check the build's integrity. Once this dummy data has been generated,
call `make test` to perform these checks. When you are happy with the results,
call `make rm-test-data` to delete the dummy data. Lastly, if you ever find the
project directory to be more cluttered than you would like, call `make clean` to
remove cache, preference, and other unnecessary files.
