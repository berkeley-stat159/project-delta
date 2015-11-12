test_data:
	wget https://www.dropbox.com/s/vkauzyy9aog958h/bold.nii.gz
	wget https://www.dropbox.com/s/qrf2jqgr1pyr7bi/behavdata.txt
	wget https://www.dropbox.com/s/7rdn76wty0qq0ml/cond001.txt
	wget https://www.dropbox.com/s/ezcx9mz68mn2k68/cond002.txt
	wget https://www.dropbox.com/s/98rzlspocvdwgx1/cond003.txt
	wget https://www.dropbox.com/s/xc89fyeosqpt6i8/cond004.txt

test:
	nosetests utils
	nosetests model

coverage:
	nosetests utils --with-coverage --cover-package=utils
	nosetests model --with-coverage --cover-package=model
