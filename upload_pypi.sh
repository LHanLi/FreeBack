'''#!/bin/bash
#mypython=!python

# Upload project to pypi

rm -rf ./build
rm -rf ./dist
rm -rf ./FreeBack.egg-info

mypython setup.py sdist bdist_wheel
twine check dist/*
twine upload -u __token__ -p password dist/*
'''