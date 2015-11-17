.PHONY: all clean coverage test

all: clean

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

coverage:
	nosetests data --with-coverage --cover-package=data

test:
	nosetests code/utils data

verbose:
	nosetests -v code/utils data
