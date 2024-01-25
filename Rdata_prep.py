import pandas as pd
import math
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler


# mapping new attributes from other dataframes function    
def df_mapping(df1, df2, mapping_attribute = None, key_attribute = None):
    new_df = df2
    new_vals = {}
    count = 1
    unique_instances = df1[mapping_attribute].unique()
    mapping = dict(zip(df1[key_attribute], df1[mapping_attribute]))
    for i in df2.columns:
        new_vals[i] = str(mapping[i]+','+ str(count))
        count = count+1
    print(new_vals)
    new_df = new_df.rename(new_vals)
    #new_df.with_columns(pl.Series(name=mapping_attribute, values=map_values)) 
    #new_df[mapping_attribute] = map_values
    #if new_index == True:
        #new_index = ["{}_{}".format(key_attribute, mapping_attribute) for key_attribute, mapping_attribute in zip(transpose_df.index, map_values)]
        #new_df.index = new_index
    return new_df

def unique_group(df1, grouping_attribute, df2=None):

    # create dictionary of unique instances of passed attribute in first dataframe
    unique_instances = df1[grouping_attribute].unique() 
    grouped_columns = {grouping_attribute: [] for grouping_attribute in unique_instances}
    
    # fill dictionary with second dataframe's data
    if df2 is not None: 
        for col in df2.columns:
            for grouping_attribute in unique_instances:
                if grouping_attribute in col:
                    grouped_columns[grouping_attribute].append(col)
                    break
    # fill dictionary with first dataframe's data
    else:
        for col in df1.columns:
            for grouping_attribute in unique_instances:
                if grouping_attribute in col:
                    grouped_columns[grouping_attribute].append(col)
                    break
    return grouped_columns
    

def aggregate_by_groups(df, grouped_columns):
    # Create an empty Polars DataFrame to store the aggregated results
    aggregated_df = pd.DataFrame()
    
    for group_name, cols in grouped_columns.items():
        if cols:  # Check if any columns exist for that group
            # Calculate the mean for the selected columns
            #mean_values = df.select(*cols).mean(axis=1)
            mean_values = (df.select(cols).mean(axis=1)).to_pandas()
            aggregated_df[group_name] = mean_values

    return aggregated_df


def norm_df(df):
    df_normalized = df.copy()

    # Normalize each column using Min-Max normalization
    for col in df_normalized.columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)

    return df_normalized

def numeric_only(df):
    """
    Create a copy of the DataFrame with only numeric columns.
    """
    # Select numeric columns
    numeric_columns = df.select_dtypes(include='number')
    
    # Create a copy with only numeric columns
    df_numeric = numeric_columns.copy()
    
    return df_numeric

def drop_duplicates(df): 
    print('\nDuplicate row removal:')
    print('Sample count before: ', len(df.index))
    df_no_duplicates = df.drop_duplicates()
    print('Sample count after: ', len(df_no_duplicates.index))

    return df_no_duplicates

def drop_null(df):
     # Drop rows with null values
    df_no_nulls = df.dropna()

    # Drop rows with 0 entries
    df_no_nulls = df_no_nulls[(df_no_nulls != 0).all(axis=1)]
    print('\nNull row removal:')
    print('Sample count before: ', len(df.index))
    print('Sample count after: ', len(df_no_nulls.index))
    
    return df_no_nulls

def drop_out_of_domain(df, std = 2):
    # Get only numeric data to identify rows with out-of-domain properties
    df_numeric = numeric_only(df)
    
    # Calculate the mean and standard deviation of all row means
    print('\nOut-of-domain row removal:')
    all_rows_mean = df_numeric.mean(axis=1)
    all_rows_mean_mean = all_rows_mean.mean()
    all_rows_mean_std = all_rows_mean.std()
    threshold_std = std
    threshold = all_rows_mean_mean + threshold_std * all_rows_mean_std
    print('Threshold =', threshold_std, ' standard deviations')

    out_of_domain_indices = []
    
    # Iterate over rows and check for out-of-domain properties
    for idx, row in df_numeric.iterrows():
        row_mean = row.mean()
        if row_mean > threshold:
            out_of_domain_indices.append(idx)
            #print(f"Row {idx} has out-of-domain properties.") 
    
    df_reduced = df.drop(out_of_domain_indices)
    print('Number of rows with out-of-domain properties: ',len(out_of_domain_indices))
    print('\nSample count before: ', len(df.index))
    print('Sample count after: ', len(df_reduced.index))

    return df_reduced

def split_data(df, test_size=0.3, random_state=None):
    #Splits a DataFrame into training and test sets.

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    return train_df, test_df

# Get target attribute's column into Y
def set_y(df, target_attribute):
    Y = df.loc[:, target_attribute]
    return Y

# Get non-target attribute columns into X
def set_x(df, target_attribute):
    X = df.loc[:, df.columns != target_attribute]
    return X

def normalize(X):
    # Normalize data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X

def remove_non_cat(df, non_cat_col):
    
    filtered_columns = [col for col in df.columns if col not in non_cat_col]
    filtered_df = df[filtered_columns]

    return filtered_df
