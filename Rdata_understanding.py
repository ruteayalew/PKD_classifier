import pandas as pd
import math
from sklearn.ensemble import IsolationForest
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor 


#These functions were aided by ChatGPT

def set_target_attribute(attribute):
    target_attribute = str(attribute)
    print('Target attribute saved as: ',target_attribute)
    
    return target_attribute

def set_format(row_def, column_def):
    """
    Logs the specified data format-if samples/attributes are rows/columns- given user input.
    """
    df_format = None

    
    if (column_def != 'features') and (column_def != 'attributes'):
        df_format = 'incorrect'
    if column_def == 'samples':
        df_format = 'incorrect'
    if row_def != 'samples':
        df_format = 'incorrect'
    else:
        df_format = 'correct'
        
    return df_format


def report_description(df, row_def, column_def):
    '''
    Reports data description including dimensionality, sample count, attribute count and datatype distribution
    '''
    df_format = None

    print('GENERATING INITIAL DESCRIPTION OF DATA:')
    
    #prints dimensions
    print('\nData dimensions:',df.shape)
    
    #print format value: either correct or incorrect and will need transposing later
    df_format = set_format(row_def, column_def)
    if df_format == 'correct':
        print('Data has proper formatting with rows=samples and columns=attributes')
    if df_format == 'incorrect':
        print('Data has improper formatting with rows=attributes and columns=samples')
    
    #counting samples and attributes
    if df_format == 'correct':
        print('\nSample Count = ')
        # Count the number of rows (samples)
        print(len(df))
    
        print('Attribute Count = ')
        # Count the number of columns (attributes)
        print(len(df.columns))
    
    if df_format == 'incorrect':
        print('\nSample Count = ')
        # Count the number of columns (samples)
        print(len(df.columns))
    
        print('Attribute Count = ')
        # Count the number of rows (attributes)
        print(len(df))

    #calls check_datatypes method to log initial distribution of datatypes that exist
    data_types = check_datatypes(df)

        
def check_datatypes(df):
    """
    Calculates the percentage of data for each datatype in a DataFrame.
    
    Returns:
    dict: Dictionary with datatypes as keys and their corresponding percentage of presence.
    """
     # Get unique datatypes
    unique_datatypes = df.dtypes.unique()

    # Print unique datatypes
    print('\nData types that are present:')
    for datatype in unique_datatypes:
        print(datatype)

    #print report of dataframe quality of samples
def report_quality(df):
    '''
    Reports quality assessment of samples(rows) by checking for duplicate rows, rows with nulls, and rows with out-of-domain properties
    '''
    # Check for duplicate rows based on non-dictionary columns
    print('Number of duplicate rows: ',count_duplicate_rows(df))
    
    # Check for missing values in each row
    rows_with_missing_values = df[df.isnull().any(axis=1)]
    num_rows_with_missing_values = len(rows_with_missing_values)

    # Print the results
    print(f"Number of rows with missing values: {num_rows_with_missing_values}")

    # Get only numeric data to identify rows with out-of-domain properties
    # Calculate the mean and standard deviation of all row means
    print('\nIdentifying rows with out-of-domain properties:')
    df_numeric = numeric_only(df)
    all_rows_mean = df_numeric.mean(axis=1)
    all_rows_mean_mean = all_rows_mean.mean()
    all_rows_mean_std = all_rows_mean.std()
    threshold_std = 2
    threshold = all_rows_mean_mean + threshold_std * all_rows_mean_std
    print('Threshold =', threshold_std, ' standard deviations')

    out_of_domain_indices = []
    
    # Iterate over rows and check for out-of-domain properties
    for idx, row in df_numeric.iterrows():
        row_mean = row.mean()
        if row_mean > threshold:
            out_of_domain_indices.append(idx)
            #print(f"Row {idx} has out-of-domain properties.")
    print('Number of rows with out-of-domain properties: ',len(out_of_domain_indices))
    
    
def hashable(row):
    """
    Convert each row to a hashable representation to support methods that compare samples 
    """
    items = row.items()
    hashable_items = [(k, str(v) if not isinstance(v, (int, float)) else v) for k, v in items]
    return frozenset(hashable_items)

def count_duplicate_rows(df):
    """
    Counts the number of duplicate rows in a DataFrame. 
    """
    num_duplicate_rows= None
        
    # Count duplicate rows 
    num_duplicate_rows = df.duplicated().sum()
    
    return num_duplicate_rows
    
def numeric_only(df):
    """
    Create a copy of the DataFrame with only numeric columns.
    """
    # Select numeric columns
    numeric_columns = df.select_dtypes(include='number')
    
    # Create a copy with only numeric columns
    df_numeric = numeric_columns.copy()
    
    return df_numeric

def drop_null(df):
    df_no_nulls = df.dropna()
    print('\nNull row removal:')
    print('Sample count before: ', len(df.index))
    print('Sample count after: ', len(df_no_nulls.index))
    
    return df_no_nulls
    
def skew_report(df):
    df_numeric = numeric_only(df) #calls method to get numeric df and make list of indices
    col_skew = 0 #individual skew measurement for each column
    
    #counters to total each type of skew
    total_skew = 0
    left_skew = 0
    right_skew = 0 
    symmetric = 0
    
    #for each column named in the series returned by the index slice, check skew and count frequency
    for col in df_numeric:
        col_skew = df[col].skew()
        if col_skew < 0:
            left_skew = left_skew + 1
        if col_skew > 0:
            right_skew = right_skew + 1
        if col_skew == 0:
            symmetric = symmetric + 1

    total_skew = left_skew + right_skew
    
    print("Skewness report:")
    print("Number of attributes with left skew: ", left_skew)
    print("Number of attributes with right skew: ", right_skew)
    print("Number of attributes with no skew/ are symmetric: ", symmetric)
    print("\n")
    print("Ratio of total skewed attributes to symmetric attributes = ", total_skew, ":", symmetric)    
    
    if symmetric > total_skew:
        print("Therefor, the data is mostly symmetrical")
    if total_skew > symmetric:
        print("Therefor, the data is mostly skewed")
        
def check_collinearity(df, target_attribute):
    df_no_null = df.dropna()
    attributes = pd.DataFrame(df_no_null.iloc[:, df_no_null.columns!=target_attribute])
    # VIF dataframe 
    vif_data = pd.DataFrame() 
    vif_data["feature"] = attributes.columns 

    # calculating VIF for each feature 
    vif_data["VIF"] = [variance_inflation_factor(attributes.values, i) 
                              for i in range(len(attributes.columns))] 

    print(vif_data)
    
# check value counts for each categorical variable 
def val_count(df):
    top_5_value_counts_names = []

    for col in df.columns:
        print(f"Top 5 value counts for '{col}':")
        value_counts = df[col].value_counts().head(5)
        print(value_counts)
        print("\n")
        top_5_value_counts_names.extend(value_counts.index.tolist())

    return top_5_value_counts_names