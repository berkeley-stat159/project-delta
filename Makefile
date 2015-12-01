filtered_data="http://nipy.bic.berkeley.edu/rcsds/ds005_mnifunc.tar"
raw_data="http://openfmri.s3.amazonaws.com/tarballs/ds005_raw.tgz"

.PHONY: all clean coverage test

all: clean

clean:
	find . -name "*.pyx.md5" -o -name ".DS_Store" -o -name "__pycache__"\
	-o -name "._hashes.json" -o -name "ds107_sub001_highres.nii"\
	-o -name "*.pyc" -o -name ".ipynb_checkpoints" | xargs rm -rf

coverage:
	make test_data
	nosetests code/tests data/test_data.py --with-coverage\
	--cover-package=code/model --cover-package=code/utils\
	--cover-package=data/data.py	

data:
	wget $raw_data --directory-prefix=data
	wget $filtered_data --directory-prefix=data
	tar -xvzf data/ds005_raw.tgz -C data/
	tar -xvzf data/ds005_mnifunc.tar -C data/
	rm data/ds005_raw.tgz data/ds005_mnifunc.tar

remove-test-data:
	find . -name "subtest" | xargs rm -rf

test:
	nosetests code/tests data/test_data.py

test-data:
	python code/utils/make_test_data.py

verbose:
	nosetests -v code/tests data/tests

validate-data:
	python code/data.py