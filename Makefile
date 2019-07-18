

install :
	python setup.py install

clean :
	rm -rf __pycache__ build *.egg-info
	rm -rf src/*.egg-info
