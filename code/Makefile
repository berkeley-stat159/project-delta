test_data:
	wget https://www.dropbox.com/s/vkauzyy9aog958h/bold.nii.gz
	wget https://www.dropbox.com/s/qrf2jqgr1pyr7bi/behavdata.txt
	wget https://www.dropbox.com/s/7rdn76wty0qq0ml/cond001.txt

test:
	nosetests utils

coverage:
	nosetests utils --with-coverage --cover-package=utils 
