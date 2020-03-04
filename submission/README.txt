Tested on Ubuntu 18.04 LTS

1) Run
conda env create -f test_environment.yml

Then activate the environment. 

If it fails:

Make conda enviromnet 
* conda create -n yourenvname python=3.6 anaconda

Install the packages
* conda install -c anaconda tensorflow-gpu=2.0.0
* conda install -c conda-forge keras
* conda install -c conda-forge opencv
* conda install -c conda-forge tqdm


2) Activate your environment and run and command line:
* pip install efficientnet

3) Make a folder named 'dataset' in this directory and put 'Testing images' folder in 'dataset'.

5) Run get_predictions.py

6) Results will be stored inside 'dataset' folder under the name 'the_great_backpropagator_MoNuSAC_test_results'
