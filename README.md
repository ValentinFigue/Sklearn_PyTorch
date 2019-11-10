# Sklearn_PyTorch

Transposition in PyTorch to Sklearn models such as Random Forest or SVM

## Goal

This module has been created to propose some very classical machine learning algorithms such as Random Forest or SVM which can be used directly within Pytorch and does not require Sklearn to be functional. This leads to different advantages : 

- The different model can be used directly with CUDA and does not require to exchange data between the CPU and GPU. 
- The model can be coupled with a convolutional neural network for instance and does not prevent the gradient to be automatically propagated.
- The user can easily use the same data to train either deep learning models or models offered by this library.

## Installation

### Requirements

This package needs only the following libraries to be functional :

- Pytorch
- Numpy

It does not require Sklearn to be used contrary to other machine learning usual packages and can be used with Python 2 or 3.

### Installation

You can install easily the package and its dependencies directly through pip : 

```console
python setup.py install
```

The package will be automatically integrated in your python and can be directly imported as other usual packages such as numpy :

```python
import Sklearn_PyTorch
```

### Developer installation 

In the case you don't want to integrate this package within your python because you want to create a python project based on this module or just develop new features, you can download the requirements packages through pip by typing the following command in the terminal :

```console
pip install -r requirements.txt
```

In the case, you prefer use this package and its dependencies in a virtual environment, which is highly recommended, you can do it both directly by typing :

```console
make install
```

Once you've downloaded the different dependencies you can test the consistency of the package with the following test command : 

```console
make tests
```

## Examples

The code below shows quickly how to use the library to use random forest on the regression problem.

```python
# Import of the model
from Sklearn_PyTorch import TorchRandomForestClassifier

# Initialisation of the model
my_model = TorchRandomForestClassifier(nb_trees=100, nb_samples=3, max_depth=5, bootstrap=True)

# Definition of the input data
import torch
my_data = torch.FloatTensor([[0,1,2.4],[1,2,1],[4,2,0.2],[8,3,0.4], [4,1,0.4]])
my_label = torch.LongTensor([0,1,0,0,1])

# Fitting function
my_model.fit(my_data, my_label)

# Prediction function
my_vector = torch.FloatTensor([1,2,1.4])
my_result = my_model.predict(my_vector)
```

## Documentation

The documentation can be generated through sphinx via the following commands : 

```console
cd docs
make html
```

Once the documentation generated, you can read it by opening the *docs/_build/html/index.rst* in a web browser.

## The future

Feel free to share and add some new features. I've created this package for fun and be pleased to help any one who tries to bring some new stuffs !
