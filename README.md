# my_ml

This is my implementation of machine larning algorithms from scratch (or a bare-bones numpy implementation rather).

The goal of this project is to A) practice machine learning algos and datastructures, and B) have already pre-written from-scratch code.

## Overview

It should be noted that *all code was written in a bare-bones numpy
implementation.* This means the code is written from scratch without the use of
any machine learning libraries.


- ▾ bin/
    - my_ml       
	- Executable for command line interface.
- ▸ data/            
	- The original Pima dataset.
  - ▾ my_ml/      
	- Main module.
  - ▸ clean/         
	- Home to the DataCleaner class. Contains a CLI for cleaning the Pima data. See below.
  - ▸ cli/           
	- CLI Directory.
  - ▸ model/         
	- Home to RandomForest and DecisionTree classes. Also has a CLI for modeling using RandomForest.
  - ▸ utils/         
	- Various utility functions including scoring metrics.
    - __init__.py
- ▸ output/          
	- Outputs.
- ▸ papers/          
	- Sources and citations.
- ▸ research/        
	- Initial research into the dataset.
- ▾ tests/           
	- Tests.
  - ▸ data/
  - ▸ functional/
  - ▸ unit/
  - CHANGELOG.md
  - Dockerfile
  - LICENSE.md
  - README.md
  - requirements.txt
  - setup.py

## Discussion

### Research/Citations
- The algorithms in this code base were written from scratch. This includes
the decision tree, scoring functions, and the
random forest class. I left sources that I took influences from in the
papers directory. In some cases I commented a link in the code. 

### Engineering
- The code base is written in python
- I added some improvements for speed over the algorithm . These enhancements include:
    - Vectorized numpy code instead of for loops when possible.
    - Multi-processing for the ```_build_forest``` method in the RandomForest
    class. This is a huge speed up.
- Other recommendations for the future:
    - Use a library like tensorflow, pytorch or opencl to distribute vectorized
    tasks over GPUs.
    - Build off an already-implemented, liberally-licensed code base. 
    - Either use cython to create C code, or use C++ with pybind11 
        - It is not always neccessary, but cython allows for a python-readable
        interface to C. C is 10x+ faster than python.

## Setup

``` bash
#### Get the code.
git clone git@github.com:big-c-note/my_ml_from_scratch.git
```
### Virtual Environment (Can Skip)

I would recommend setting up a virtual environment. I use virtualenv, but you
can also use conda for this. It's not 100% neccessary, but it will help get the
right package versions. 

These instructions are for Mac.

``` bash
#### Install virtualenv if you don't have it.
pip3 install --user virtualenv

#### Create virtual environment.
python3 -m venv path/to/env

#### Activate virtual env.
source path/to/env/bin/activate

#### Deactivate (once you no longer need the virtual env).
deactivate
```

### Install Dependencies

After activating your virtual environment, you can install all the neccessary
dependencies with the following commands.

``` bash
#### Install the dependencies.
cd path/to/top/level/my_ml/directory
pip3 install -r requirements.txt

#### Install the my_ml module.
cd path/to/top/level/my_ml/directory
pip3 install -e .
```

## Run the Code!

### Model

The model CLI will run the user supplied parameters for RandomForest. You can easily
adjust the code to iterate over many parameters. Each run will output the
neccessary cross validations and test scoring metrics to a created research directory.

``` bash
my_ml model --help
my_ml model

#### Alternatively
cd path/to/top/level/my_ml/directory
python3 my_ml/model/runner.py
```
### Clean

You can run the cli command to clean the data. It is uneccesary to do this as
I have supllied pre-cleaned code. 

Be aware that there is a ```random_state``` set unless you supply the flag and
value ```--random_state None```.

``` bash
my_ml clean --help
my_ml clean

#### Alternatively
cd path/to/top/level/my_ml/directory
python3 my_ml/clean/runner.py
``` 

## Exploring the Outputs

You'll find all of the model data in the output/model directory. You will
see the charts output by the ```my_ml model``` command. I left one run
in there with the user requested parameters. You can open this file and explore
the contents like so:

``` python 
#### Open interactive python.
ipython 

import joblib
cat = joblib.load('path/to/results/dir/server.gz')

#### This is the model that was trained on the training data.
cat['model']

#### You can run the model on the test data if you like.
random_forest = cat['model']
data = joblib.load('tests/data/server.gz')
X_test = data['_X_test']
predictions = random_forest.predict(X_test)

#### You can see the metrics for both the cross validation and the test data.
cat['test']
cat['cross_val']
```

## Code Examples

The code can be applied to other two-class supervised classification problems
with imbalance. Simply pip install the module. Read the doc strings for
RandomForest, but to get started, here is an example.

``` python
from my_ml.model.random_forest import RandomForest
random_forest = RandomForest(S=100)
random_forest.fit(X_train, y_train)
predictions = random_forest.predict(X_test)
``` 
### Running Tests

To ensure I wasn't introducing small bugs throughout development, I wrote
tests for the classes and some methods. This will also help you ensure there
are no version issues with the different dependencies, etc.

``` bash
cd path/to/top/level/my_ml/directory
pytest
```
