0) Run
conda env create -f test_environment.yml

Then activate the environment. If it fails:

1) Make conda enviromnet 
* conda create -n test python=3.6 anaconda

2) Then install the packages
* conda install -c anaconda tensorflow-gpu=2.0.0
* conda install -c conda-forge keras
* conda install -c conda-forge opencv
* conda install -c conda-forge tqdm


3) Put Testing_images in dataset folder
4) Run test.py
