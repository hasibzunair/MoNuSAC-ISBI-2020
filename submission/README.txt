Tested on Ubuntu 18.04 LTS

1) Run
conda env create -f test_environment.yml

Then activate the environment. If it fails:

OR 

Make conda enviromnet 
* conda create -n yourenvname python=3.6 anaconda

Install the packages
* conda install -c anaconda tensorflow-gpu=2.0.0
* conda install -c conda-forge keras
* conda install -c conda-forge opencv
* conda install -c conda-forge tqdm


2) Activate your environment and run and command line:
* pip install efficientnet


4) Put Testing_images folder in dataset
5) Run get_predictions.py
