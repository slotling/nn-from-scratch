# neural network from scratch
- **no machine learning libraries at all** (tensorflow/pytorch/sklearn...)
- numpy/matplotlib/other utility libraries are used
- data ([source](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)) is manually downloaded and placed in `src/data/`

![figure](img/output.png)
around 90% train accuracy on small sample of dataset, very slow gradient descent, and doesn't perform that good on the test set

### docs
- `src/main.py` - training file
- `src/main_notrain.py` - prediction file using saved parameters from the training file
- `src/parameters/` - where parameters are saved
- `src/model.py` - neural network architecture + training procedures (backpropagation!)
- `src/emath.py` - math functions
- `src/dataload.py` - utility csv loading stuff

### setup
- `pip install -r requirements.txt`
- `pip freeze >> requirements.txt`