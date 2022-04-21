

.PHONY : install clean test

install :
	python setup.py install

clean :
	rm -rf __pycache__ build *.egg-info
	rm -rf src/*.egg-info

test: install
	python -m pytest -v

test-dev: install
	python -m pytest -v test_xrd_transforms/test_rotate_vecs_about_axis.py
