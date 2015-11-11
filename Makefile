<<<<<<< HEAD
TITLE="report"

all:
	pdflatex $(TITLE).tex
	bibtex $(TITLE)
	pdflatex $(TITLE).tex
	pdflatex $(TITLE).tex


clean:
	rm -f *.{aux,log,bbl,lof,lot,blg,out}
=======
.PHONY: all clean coverage test

all: clean

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

coverage:
	nosetests code/utils data --with-coverage --cover-package=data  --cover-package=utils

test:
	nosetests code/utils data

verbose:
	nosetests -v code/utils data
>>>>>>> e1ecf7f096378429537bb75ac48522f4d28200b6
