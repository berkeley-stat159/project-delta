.PHONY: all clean coverage test

all: clean

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" -o -name "*pycache*" -o -name "._hashes.json" | xargs rm -rf

coverage:
	nosetests code/utils/tests code/model/tests data/tests --with-coverage --cover-package=code/model --cover-package=code/utils --cover-package=data/data.py

test:
	nosetests code/model/tests code/utils/tests data/tests

verbose:
	nosetests -v code/model/tests code/utils/tests data/tests
