# -----------------------------------------------------------
# Modeling related
#
# Author: Nicole Sung
# Created: 9/14/2023
# Modified: 10/4/2023
# 
# -----------------------------------------------------------

# Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Regression
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import math

# General
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import warnings, time

# Define a dictionary of regressor names and their corresponding models
REGRESSOR_FUNCTIONS = {
    "Naive": DummyRegressor(strategy="mean"),
    "Baseline": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Elastic Net Regression": ElasticNet(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    
}

# Define a dictionary of classifier names and their corresponding models
CLASSIFIER_MODELS = {
    'Naive': DummyClassifier(strategy='most_frequent'),
    'DecisionTree': DecisionTreeClassifier(),
    'Dummy': DummyClassifier(),
    'SVC': SVC(),
    'RandomForest': RandomForestClassifier(),
    'GaussianNB': GaussianNB(),
    'MLP': MLPClassifier()
}

class CustomClassifierPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.results = {}

    def evaluate_classifiers(self):
        for name, model in CLASSIFIER_MODELS.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            self.results[name] = y_pred

    def get_results(self):
        return self.results

def evaluate_models_pipe(X, y, model_selection, test_size=0.3, random_state=1, suppress_warnings=True):
    if suppress_warnings:
        warnings.filterwarnings('ignore')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    custom_pipeline = CustomClassifierPipeline(X_train, y_train, X_test, y_test)
    custom_pipeline.evaluate_classifiers()

    return custom_pipeline.get_results()



def model_analysis(X_train, X_test, y_train, y_test, model_type, model, additional_info = False):
    matrix = None
    # Start timing
    start_time = time.time()

    # Fit the model
    model.fit(X_train, y_train)
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Make predictions
    y_pred = model.predict(X_test)
    
    print(f"{model.__class__.__name__} Model Fitting Time: {elapsed_time:.3f} seconds")

    # Calculate the error
    if model_type == "classification":
        matrix = compute_classification_matrix(y_test, y_pred, model)
    else:
        matrix = compute_regression_matrix(y_test, y_pred, model, additional_info)
    return model, y_pred, matrix

# Evaluates a list of models and returns the best classifier based on accuracy.
def evaluate_models(X, y, model_selection, test_size=0.3, random_state=1, suppress_warnings=True):

    if suppress_warnings:
        warnings.filterwarnings('ignore')

    best_model = None
    best_metric = 0 if model_selection == 'classification' else float('inf')
    best_y_pred = None
    best_name = ""
    MODEL_FUNCTIONS = CLASSIFIER_FUNCTIONS if model_selection == 'classification' else REGRESSOR_FUNCTIONS

    print(f"\nPhase 4: {model_selection.capitalize()} Modeling")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    for name, model_func in MODEL_FUNCTIONS.items():
        model, y_pred, metric = model_analysis(X_train, X_test, y_train, y_test, model_selection, model_func)
        
        # Store best model result
        if (model_selection == 'classification' and metric['accuracy'] > best_metric) or \
           (model_selection == 'regression' and metric['mse'] < best_metric):
            best_metric = metric['accuracy'] if model_selection == 'classification' else metric['mse']
            best_model = model
            best_y_pred = y_pred
            best_name = name


    print("="*60)
    print(f"Best Model: {best_name} with Metric: {best_metric:.3f}")

    return best_model

# Compute the accuracy of predictions.
def compute_classification_matrix(y_true, y_pred, model):
    metrics = {}
    accuracy = accuracy_score(y_true, y_pred)
    metrics['accuracy'] = accuracy
    print(f"{model.__class__.__name__} Accuracy: {accuracy*100:.3f}%")
    print("-"*60)
    return metrics

def compute_regression_matrix(y_true, y_pred, model, additional_info):
    metrics = {}
    error = mean_squared_error(y_true, y_pred)
    metrics['mse'] = error
    print(f"{model.__class__.__name__} MSE: {error:.3f}")
    
    # Additional regression metrics
    if additional_info:
        residuals = y_true - y_pred
        SSE = np.sum(residuals**2)
        SST = np.sum((y_true - np.mean(y_true))**2)
        SSM = SST - SSE
        r2 = r2_score(y_true, y_pred)
        
        metrics['SSE'] = SSE
        metrics['SST'] = SST
        metrics['SSM'] = SSM
        metrics['r2'] = r2
        
        print(f"{model.__class__.__name__} SSE: {SSE:.3f}")
        print(f"{model.__class__.__name__} SST: {SST:.3f}")
        print(f"{model.__class__.__name__} SSM: {SSM:.3f}")
        print(f"{model.__class__.__name__} R^2: {r2:.3f}")

    print("-"*60)
    return metrics

def predict_confidence(clf, X):
    # Predict labels and get confidence scores or probability estimates.
    
    # Predict labels
    pred_labels = clf.predict(X)
    
    # Try to get probability estimates
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)
        # Assuming binary classification; take the probability of the second class
        pred_confidence = probs[:, 1]
    # For classifiers like SVM without probability estimates but with decision_function
    elif hasattr(clf, "decision_function"):
        # Distance from the decision boundary can be a measure of confidence
        pred_confidence = clf.decision_function(X)
    else:
        raise ValueError("The classifier does not have a method to provide confidence values.")
    
    return pred_labels, pred_confidence


def decision_tree_classifier(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def naive_classifier(X_train, y_train, X_test, y_test):
    model = DummyClassifier(strategy='most_frequent')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def support_vector_classifier(X_train, y_train, X_test, y_test):
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def random_forest_classifier(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def naive_bayes_classifier(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def mlp_classifier(X_train, y_train, X_test, y_test):
    model = MLPClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def naive_regression(X_train, y_train, X_test, y_test):
    model = DummyRegressor(strategy="mean")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def ridge_regression(X_train, y_train, X_test, y_test):
    model = Ridge()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def lasso_regression(X_train, y_train, X_test, y_test):
    model = Lasso()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def elastic_net_regression(X_train, y_train, X_test, y_test):
    model = ElasticNet()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def decision_tree_regressor(X_train, y_train, X_test, y_test):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def random_forest_regressor(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def gradient_boosting_regressor(X_train, y_train, X_test, y_test):
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def support_vector_regressor(X_train, y_train, X_test, y_test):
    model = SVR()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def mlp_regressor(X_train, y_train, X_test, y_test):
    model = MLPRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

