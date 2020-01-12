# linear_regression_implementation
Python implementation from scratch - Linear Regression

 This is a python implementation of Linear (simple and L2/Ridge) and Logistic Regression using the AdaGrad optimization algorithm. In the file linreg.py, I have defined 3 classes for the same:

 1) LinearRegression621: simple linear regression. It has methods fit() and predict(), just like scikit-learn.
 2) RidgeRegression621: linear regression with L2 regularisation. It has methods predict() and fit().
 3) LogisticRegression621: logistic regression. It has methods fit(), predict() and predict_proba().

 The AdaGrad algorithm is encapsulated inside minimize() function, which is called by above classes and returns the parameter combination that minimizes loss.

 test_class.py and test_regr.py are tests designed to validate our implementation.

 Thanks to @partt for the guidance and support in this school project.
