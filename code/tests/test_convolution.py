"""
"""
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import fileinput, os, sys


def test_convolution():

    #
    f = open("code/scripts/convolution.py", "r")
    lines = f.readlines()
    f.close()

    #
    lines_test = lines
    lines_test[25] = "subjects = ['test']\n"
    f = open("code/scripts/convolution.py", "w")
    lines_test = "".join(lines_test)
    f.write(lines_test)
    f.close()

    #
    os.system("python code/scripts/convolution.py")

    #
    files = ["results/subtest_run001/convolution.png",
             "results/subtest_run001/conv_gain.txt",
             "results/subtest_run001/conv_loss.txt",
             "results/subtest_run001/conv_dist2indiff.txt",
             "results/subtest_run002/convolution.png",
             "results/subtest_run002/conv_gain.txt",
             "results/subtest_run002/conv_loss.txt",
             "results/subtest_run002/conv_dist2indiff.txt",
             "results/subtest_run003/convolution.png",
             "results/subtest_run003/conv_gain.txt",
             "results/subtest_run003/conv_loss.txt",
             "results/subtest_run003/conv_dist2indiff.txt"]
    for file in files: assert os.path.isfile(file)

    #
    f = open("code/scripts/convolution.py", "w")
    for line in lines: f.write(line)
    f.close()
