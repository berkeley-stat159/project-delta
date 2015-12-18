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
	tar -xvf data/ds005_mnifunc.tar -C data/
	rm data/ds005_raw.tgz data/ds005_mnifunc.tar

test-data:
	python code/utils/make_test_data.py

validate-data:
	python data/data.py

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

diagnosis:
	python code/scripts/diagnosis.py

smoothing:
	python code/scripts/smoothing.py

convolution:
	python code/scripts/convolution.py

glm:
	python code/scripts/glm.py

visualization:
	python code/scripts/visualization.py

logistic:
	python code/scripts/logistic.py

conjunction:
	python code/scripts/conjunction.py

analyses:
	make diagnosis
	make smoothing
	make convolution
	make glm
	make visualization
	make logistic
	make conjunction

rm-results:
	rm -rf results
	
##################
# Project Report #
##################
	
report:
	cd paper && make all && make clean
	
rm-report:
	rm -f paper/report.pdf

