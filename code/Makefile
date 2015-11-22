test_data:
	wget https://www.dropbox.com/s/vkauzyy9aog958h/bold.nii.gz
	wget https://www.dropbox.com/s/qrf2jqgr1pyr7bi/behavdata.txt
	#Proposed change:
	#python make_test_data.py

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
