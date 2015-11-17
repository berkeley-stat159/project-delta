.PHONY: all clean coverage test

all: clean

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

coverage:
	wget https://www.dropbox.com/s/qrf2jqgr1pyr7bi/behavdata.txt
	nosetests data --with-coverage --cover-package=data
	nosetests code/utils --with-coverage --cover-package=utils
	rm -f behavdata.txt

test:
	nosetests code/utils data

verbose:
	nosetests -v code/utils data
