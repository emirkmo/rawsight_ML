[![Python tests](https://github.com/emirkmo/andrew-ng-ML-course-labs/actions/workflows/python-app.yml/badge.svg)](https://github.com/emirkmo/andrew-ng-ML-course-labs/actions/workflows/python-app.yml)
# Labs for Stanford ML (Andrew NG's) Machine Learning Course on Coursera

Implemented in Python and following Object-Oriented Design principles,
including SOLID, but also using elements of functional design patterns. see design decisions below. Results tested against sklearn.

The code is written to be easily extensible and maintainable without having to do
significant refactoring. Such as adding new cost functions, models, etc.

Each Course is a directory and each week has a single .py file per lab. The code is
written by me to complete the lab, so is not copied. The plotting is left to the
actual lab jupyter notebook, which is easier to use, and can be found by enrolling
in the course on Coursera. 

## Design decisions

**models:** models (e.g., 'LinearModel' 'LogisticModel') implement the 'Model' class interface which is a 'Protocol' class. There is a 'BaseLinearModel' Mixin, inherited by these two models, which implements helper functons such as verification and shard methods and attributes. This base mixin also abstractly implements the 'Model' Protocol. This is a variant of the Strategy pattern. 

**cost functions:** Abstract base class 'AbstractCostFunction' defines the interface. 'CostFunction' is a factory class which creates CostFunctions with the correct cost function and gradient, such as negative log for logistic regression, as well as optionally aconcrete 'Regularization' strategy. The costs and gradients are defined as private functions living in cost_functions.py, which is the only file that needs to be modified to add new cost functions. This is a Factory pattern variant.

**normalization:** Normalizer class defines the interface, and using Strategy pattern various normalizers are implemented such as ZScoreNorm or MaxNorm. The normalization parameters required for denormalization are saved in the class instance.

**optimization:** Currently only implements batched gradient descent with and without explicit regularizaton, defined as functions. Can be used with 'functools.partial' to pass configured instances around to modify the hyper parameters.

**data:** datasets are stored in a 'DataSet' 'dataclass', with helper methods to make test train splits, and can simply be initalized from a pandas dataframe. Datasets can extact feature amd target varialbles, normalize or denormalize features given a Normalizer instance (default ZScoreNorm). Loader functions are provided with example of the housing dataset from the course to initialize a DataSet and create training samples from it.




Suggestions for improvement are welcome!
