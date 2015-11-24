test_data:
	python utils/make_test_data.py

test:
	nosetests utils
	nosetests model

coverage:
	make test_data
	nosetests utils --with-coverage --cover-package=utils
	nosetests model --with-coverage --cover-package=model
	make clean

clean:
	rm -f *.{nii.gz,txt}
