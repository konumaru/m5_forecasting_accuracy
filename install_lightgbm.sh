git clone --recursive https://github.com/microsoft/LightGBM.git
cd LightGBM/python-package
poetry run python setup.py install
cd ../../
rm -rf LightGBM
