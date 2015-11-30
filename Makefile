.PHONY: all clean coverage test

all: clean

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" -o -name "__pycache__" -o -name "._hashes.json" -o -name ".DS_Store" | xargs rm -rf

coverage:
	nosetests code/tests data/tests --with-coverage --cover-package=code/model --cover-package=code/utils --cover-package=data/data.py

test:
	nosetests code/tests data/tests

test-data:
	#wget http://openfmri.s3.amazonaws.com/tarballs/ds005_raw.tgz --directory-prefix=data
	#wget http://nipy.bic.berkeley.edu/rcsds/ds005_mnifunc.tar --directory-prefix=data
	#tar -xvzf data/ds005_raw.tgz -C data/
	#tar -xvzf data/ds005_mnifunc.tar -C data/
	#rm data/ds005_raw.tgz data/ds005_mnifunc.tar

verbose:
	nosetests -v code/tests data/tests
