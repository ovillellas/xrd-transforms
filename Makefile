

.PHONY : install clean test

install :
	python setup.py install

clean :
	rm -rf __pycache__ build *.egg-info
	rm -rf src/*.egg-info

test: install
	python -m pytest -v
