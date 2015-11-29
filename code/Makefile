test_data:
	python utils/make_test_data.py

test:
	nosetests tests

coverage:
	make test_data
	nosetests tests --with-coverage --cover-package=model --cover-package=utils
	make clean

clean:
	rm -f *.{nii.gz,txt}
