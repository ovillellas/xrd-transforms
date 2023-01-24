

.PHONY : install clean test

install :
	python3 -m pip install .

clean :
	rm -rf __pycache__ build *.egg-info
	rm -rf src/*.egg-info

test: install
	python -m pytest -v

test-dev: install
	python -m pytest -v test_xrd_transforms/test_rays_to_xy_planar.py
