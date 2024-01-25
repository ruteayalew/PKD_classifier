import pandas as pd
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn.linear_model import Ridge
from sklearn import linear_model 

# save trained model as a pkl file, pass FULL desired path to file
def save_model_pkl(model, name):
    model_pkl_file = str(name)
    
    with open(model_pkl_file, 'wb') as file:  
        pickle.dump(model, file)
        
# not necessary but optional visualization of predictions as dataframe
def show_pkl_as_df(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    df_data = pd.DataFrame(data)
    return df_data

# model evaluation 
def model_eval(y_test, predictions):
    squared_error_val = mean_squared_error(y_test, predictions)
    absolute_error_val = mean_absolute_error(y_test, predictions)

    squared_error = "{:.2%}".format(squared_error_val)
    absolute_error = "{:.2%}".format(absolute_error_val)

    print( 
      'mean_squared_error : ', squared_error) 
    print( 
      'mean_absolute_error : ', absolute_error)
    
 # create regression model
def mult_regression_model(x_train, y_train, x_test, y_test):
    model  = LinearRegression()
    #print("x_train shape:", x_train.shape)
    #print("x_test shape:", x_test.shape)

    #print("\ny_train shape:", y_train.shape)
    #print("y_test shape:", y_test.shape)
    
    #fit
    model = model.fit(x_train, y_train)
    
    # Use the model for prediction on the selected features of the test set    
    predictions = model.predict(x_test)
    
    #print("\ny_test shape:", y_test.shape)
    #print("predictions shape:", predictions.shape)

    return predictions

def ridge_reg(x_train, y_train, x_test, y_test):
    rdg = Ridge(alpha = 0.5)
    rdg = rdg.fit(x_train, y_train)
    rdg_pred = rdg.predict(x_test)
    score = "{:.2%}".format(rdg.score(x_test,y_test))
    print('RGD score:',score)
    
    model_eval(y_test,rdg_pred)
    
    return rdg_pred

def lasso_reg(x_train, y_train, x_test, y_test):
    # Build lasso model
    lasso_model = linear_model.Lasso(alpha=0.01)
    lasso_model = lasso_model.fit(x_train, y_train)

    # predict on test data and return predictions
    lasso_prediction = lasso_model.predict(x_test)
    
    score = "{:.2%}".format(lasso_model.score(x_test,y_test))
    print('Lasso score:',score)

    # evaluate and print results
    model_eval(y_test, lasso_prediction) 
    
    return lasso_prediction
