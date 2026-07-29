#!/bin/bash

rm -rf build *.egg-info geofikinpy/__pycache__
find geofikinpy -name "*.so" -delete
python setup.py build_ext --inplace