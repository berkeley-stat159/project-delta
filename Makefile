#####################
# GENERAL UTILITIES #
#####################

.PHONY: all clean coverage test

all: clean

clean:
	find . -name "*.pyx.md5" -o -name ".DS_Store" -o -name "__pycache__" \
	-o -name "._hashes.json" -o -name "ds107_sub001_highres.nii" \
	-o -name "*.pyc" -o -name ".ipynb_checkpoints" | xargs rm -rf


#############
# VARIABLES #
#############

raw="http://openfmri.s3.amazonaws.com/tarballs/ds005_raw.tgz"
filtered="http://nipy.bic.berkeley.edu/rcsds/ds005_mnifunc.tar"
color="http://www.jarrodmillman.com/rcsds/_downloads/actc.txt"

##########
# SET-UP #
##########

dataset:
	wget -N $(raw) --directory-prefix=data
	wget -N $(filtered) --directory-prefix=data
	wget -N $(color) --directory-prefix=code/scripts
	tar -xvzf data/ds005_raw.tgz -C data/
	tar -xvzf data/ds005_mnifunc.tar -C data/
	rm data/ds005_raw.tgz data/ds005_mnifunc.tar

test-data:
	python code/utils/make_test_data.py

validate-data:
	python code/data.py

test:
	nosetests code/tests data/test_data.py

rm-test-data:
	find . -name "subtest" | xargs rm -rf

verbose:
	nosetests -v code/tests data/tests


############
# COVERAGE #
############

coverage:
	make test-data
	nosetests code/tests data/test_data.py --with-coverage \
	--cover-package=code/utils,data/data.py


#############################
# RUN DATA ANALYSIS SCRIPTS #
#############################

convolution:
	python code/scripts/convolution.py

logistic:
	python code/scripts/logistic.py

smoothing:
	python code/scripts/smoothing.py

diagnosis:
	python code/scripts/diagnosis.py

linear-analysis:
	python code/scripts/linear_analysis.py

analyses:
	make convolution
	make logistic
	make smoothing
	make diagnosis
	make linear-analysis

rm-results:
	rm -rf results
	
