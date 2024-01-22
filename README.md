[![Python tests](https://github.com/emirkmo/rawsight_ML/actions/workflows/python-app.yml/badge.svg)](https://github.com/emirkmo/rawsight_ML/actions/workflows/python-app.yml)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/emirkmo/092a8fbf51f22d7ccf5fc01059f5d5d4/raw/rawsight_badge.json)
# RawSight ML: Raw Python/Numpy ML Algorithms
### Grants insight into ML algorithms with simple, object-oriented, raw `Python`/`Numpy` implementations tested against `SKLearn`/`Tensorflow`
libraries.

The library is developed based on labs for Stanford ML (Andrew NG's/Deeplearning.ai's) new Machine Learning course on Coursera and the repo includes solutions that use the underlying `rawsight` library developed by me to cover the algorithms from the course. 
However,these are implemented in Python following Object-Oriented Design principles, vectorization by default with `numpy`, and clean code practices
such as SOLID for the software design. Elements of functional design patterns are also used since these are made convenient by Python. Some thoughts on design decisions can be found further below. 

The code is written to be easily extensible and maintainable without having to do
significant refactoring. Such as adding new cost functions, models, normalizations, regularizations,
activation-functions for neural-nets, etc. The CI tests against `SKlearn` and `tensorflow` should ensure
that the developed code is accurate and extension does not break functionality.

For the labs, there are course and week directories, with a single `.py` file per lab,
which ONLY contains the solution code written by me to complete the lab. The code
uses the `rawsight`
Instructions and plotting
are left to the
actual lab jupyter notebook, which is easier to use, and can be found by enrolling
in the course on Coursera.
The code is written by me to complete the lab, so is not copied. The 

## Design decisions

**models:** models (e.g., 'LinearModel' 'LogisticModel') implement the 'Model' class interface which is a 'Protocol' class. There is a 'BaseLinearModel' Mixin, inherited by these two models, which implements helper functons such as verification and shard methods and attributes. This base mixin also abstractly implements the 'Model' Protocol. This is a variant of the Strategy pattern. 

**cost functions:** Abstract base class 'AbstractCostFunction' defines the interface. 'CostFunction' is a factory class which creates CostFunctions with the correct cost function and gradient, such as negative log for logistic regression, as well as optionally aconcrete 'Regularization' strategy. The costs and gradients are defined as private functions living in cost_functions.py, which is the only file that needs to be modified to add new cost functions. This is a Factory pattern variant.

**normalization:** Normalizer class defines the interface, and using Strategy pattern various normalizers are implemented such as ZScoreNorm or MaxNorm. The normalization parameters required for denormalization are saved in the class instance.

**optimization:** Currently only implements batched gradient descent with and without explicit regularizaton, defined as functions. Can be used with 'functools.partial' to pass configured instances around to modify the hyper parameters.

**data:** datasets are stored in a 'DataSet' 'dataclass', with helper methods to make test train splits, and can simply be initalized from a pandas dataframe. Datasets can extact feature amd target varialbles, normalize or denormalize features given a Normalizer instance (default ZScoreNorm). Loader functions are provided with example of the housing dataset from the course to initialize a DataSet and create training samples from it.


Suggestions for improvement, comments, insights into software design, requests for new algorithms or questions about algorithm implementation are welcome!

Note: Some of my lab work has been uploaded here for future reference.
