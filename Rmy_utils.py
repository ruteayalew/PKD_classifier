import os
import csv
import pandas as pd
import logging
import hashlib
import json
import requests
import rds2py
from io import BytesIO
import gzip
import rds2py
import scipy.sparse
import polars as pl
import pathlib

# ***NEW FUNCTIONS!!!

logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

def read_url_func(url):
    
    response = requests.get(url)
    
    # Send an HTTP HEAD request to the URL to check the headers
    response_head = requests.head(url)
    
    # Check the 'Content-Encoding' header
    content_encoding = response_head.headers.get('Content-Encoding', '')
    file_extension = os.path.splitext(url)[1] 
    
    if response.status_code == 200:
        if 'gzip' in content_encoding or file_extension == '.gz':
            # Fetch the compressed CSV data
            response_content = BytesIO(response.content)
            
            # Decompress the content and read into a pandas DataFrame
            with gzip.GzipFile(fileobj=response_content) as decompressed:
                df = pl.read_csv(decompressed.read(), separator = ',', ignore_errors=True )
            return df
        else: 
            print(f"File extension: {file_extension}")
            print(f"Content encoding: {content_encoding}")
            # Parse the data from the response into a DataFrame
            # df = pl.read_csv(url)  # You can adjust this line depending on the data format
            # return df
    else:
        print('Failed to request data')

# Automated File Reading Based on User-Initialized Variables. Returns dataframe
def get_data(input_dir, url=None, file_name=None, file_path=None, read_type='url', **kwargs):
    if read_type == 'url':
        df = read_url_func(url) 
        print(f'{"<" * 10} URL File Read Successfully {">" * 10}')
        save_to_csv(df, input_dir, file_name)
        df.head
        return df
    elif read_type == 'disk':
        full_path = os.path.join(input_dir, file_path) # create path to download existing file
        print(full_path)
        file_extension = os.path.splitext(file_path)[1] # get the file type 
        if file_extension == '.csv':
            df = pl.read_csv(file_path, separator = ",", ignore_errors=True)
            return df
        if file_extension == '.rds': # note the save_to_csv statements are commented out bc runtime is too long
            matrix, dim_1 = read_rds(full_path) 
            print(f'{"<" * 10} RDS File Read Successfully {">" * 10}')
            df = pl.DataFrame(matrix, schema=dim_1)
            #save_to_csv(df, input_dir, file_name) 
            #print('SUCCESSFULLY SAVED DATAFRAME TO CSV!')
            return df

def save_to_csv(file, output_dir, name, announce = True):
    ''' 
    Saves a dataframe to a CSV file.
    '''
    full_path = os.path.join(output_dir, name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file.write_csv(full_path, separator=",")
    if announce: 
            print(f'{"<" * 10} Processed data saved to {os.path.join(output_dir, name)} {">" * 10}')

def numeric_only(df):
    """
    Create a copy of the DataFrame with only numeric columns.
    """
    # Select numeric columns
    numeric_columns = df.select_dtypes(include='number')
    
    # Create a copy with only numeric columns
    df_numeric = numeric_columns.copy()
    
    return df_numeric

def object_df(df):
    """
    Create a copy of the DataFrame with only non-numeric aka object datatype columns.
    """
    # Select numeric columns
    obj_col = df.select_dtypes(include = 'object')
    
    # Create a copy with only numeric columns
    df_object = obj_col.copy()
    
    return df_object

def one_hot_encode_top_values(df, num_top_values=5):
    """
    Performs one-hot encoding using the top 'num_top_values' most common data points for each attribute.

    """
    df_copy = df.copy()

    encoded_columns = []

    for col in df.columns:
        for value in top_values:
            new_col_name = f"{col}_{value}"
            df_copy[new_col_name] = (df_copy[col] == value).astype(int)
            encoded_columns.append(new_col_name)

    return df_copy[encoded_columns]

    # parameters: original df, dummy df, and target_Attribute variable
def return_target_col(df, df_dummy, target_attribute):
    df_full = df_dummy
    df_full[target_attribute] = df[target_attribute]
    return df_full

def str_attribute_to_len(df, col_name):
    column_name = str(col_name)
    # Compute string lengths for the specified column
    string_lengths = df[column_name].astype(str).apply(len)

    return string_lengths

def read_rds(file_path):
    rObj = rds2py.read_rds(file_path)
    data_dims = rObj["attributes"]["Dim"]["data"]
    #dim_0 = rObj["attributes"]["Dimnames"]["data"][0]["data"]
    dim_1 = rObj["attributes"]["Dimnames"]["data"][1]["data"]

    i_dgCMatrix = rObj["attributes"]["i"]["data"]
    p_dgCMatrix = rObj["attributes"]["p"]["data"]
    none_zero_values_dgCMatrix = rObj["attributes"]["x"]["data"]

    sparse_matrix = scipy.sparse.csc_matrix((none_zero_values_dgCMatrix, i_dgCMatrix, p_dgCMatrix), shape=data_dims)
    dense_matrix = sparse_matrix.toarray()
    return dense_matrix, dim_1

