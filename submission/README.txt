Tested on Ubuntu 18.04 LTS

0) Run
conda env create -f test_environment.yml

Then activate the environment. If it fails:

1) Make conda enviromnet 
* conda create -n yourenvname python=3.6 anaconda

2) Then install the packages
* conda install -c anaconda tensorflow-gpu=2.0.0
* conda install -c conda-forge keras
* conda install -c conda-forge opencv
* conda install -c conda-forge tqdm


3) Activate your environment and run and command line:
* pip install efficientnet


4) Put Testing_images in dataset folder
5) Run test.py
