#!/bin/bash

python setup.py build_ext --inplace
mv convop.so .. 
