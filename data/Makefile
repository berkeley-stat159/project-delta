data:
	wget http://openfmri.s3.amazonaws.com/tarballs/ds005_raw.tgz
	wget http://nipy.bic.berkeley.edu/rcsds/ds005_mnifunc.tar
	tar -xvzf ds005_raw.tgz
	tar -xvzf ds005_mnifunc.tar
	rm ds005_raw.tgz ds005_mnifunc.tar

validate:
	python data.py
