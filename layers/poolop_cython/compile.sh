#!/bin/bash

python setup.py build_ext --inplace
mv poolop.so .. 
