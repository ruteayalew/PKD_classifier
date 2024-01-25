import pandas as pd
class SaveThreshold:
    def __init__(self, 
                 name, 
                 drop_col, 
                 add_col, 
                 is_z_score, 
                 is_row_mean, 
                 outlier_threshold, 
                 scalar,
                 is_correlation, 
                 is_normalization, 
                 is_selection, 
                 is_PCA, 
                 correlation_threshold, 
                 feature_selection_variance_threshold, 
                 is_visualise,
                 mode):
        """
        Initializes the SaveThreshold class with various thresholds and flags.

        Parameters:
        - name (str): Name associated with the threshold instance.
        - drop_col (list): List of columns to be dropped.
        - add_col (list): List of columns to be added.
        - is_z_score (bool): Flag indicating whether to use Z-score for outliers.
        - is_row_mean (bool): Flag indicating whether to use row mean for outliers.
        - outlier_threshold (float): Threshold value for identifying outliers.
        - is_scalar (bool): Flag indicating whether scalar transformation is applied.
        - is_correlation (bool): Flag indicating whether correlation-based feature selection is applied.
        - is_normalization (bool): Flag indicating whether normalization is applied.
        - is_selection (bool): Flag indicating whether feature selection is applied.
        - is_PCA (bool): Flag indicating whether PCA is applied for dimensionality reduction.
        - correlation (float): Correlation threshold for feature selection.
        - feature_selection_variance_threshold (float): Variance threshold for feature selection.
        - is_visualise (bool): Flag indicating whether to visualize PCA.
        """
        self.name = name
        self.drop_col = drop_col
        self.add_col = add_col
        self.is_z_score = is_z_score
        self.is_row_mean = is_row_mean
        self.outlier_threshold = outlier_threshold
        self.scalar = scalar
        self.is_correlation = is_correlation
        self.is_normalization = is_normalization
        self.is_selection = is_selection
        self.is_PCA = is_PCA
        self.correlation_threshold = correlation_threshold
        self.feature_selection_variance_threshold = feature_selection_variance_threshold
        self.is_visualise = is_visualise
        self.mode = mode
        self.selected_col = []
        self.pca = None

    def add(self, col_list):
        self.selected_col.append(col_list)
    
    def set_scalar(self, scalar):
        self.scalar = scalar


    def set_pca(self, pca):
        self.pca = pca


    def load(self, new_df):
        if self.is_correlation or self.is_selection:
            # Debugging: Print the type of self.selected_col and new_df.columns
            selected_col_list = [str(col) for col in self.selected_col]
            
            # Now proceed with the filtering
            valid_columns = [col for col in selected_col_list if col in new_df.columns]
            print("Surprice")
            print(valid_columns)
            new_df = new_df[valid_columns]
            print(new_df.shape)


        if self.is_normalization:
            new_df = self.scalar.transform(new_df)
        print(new_df.shape)

        if self.is_PCA:
            new_df = self.pca.transform(new_df)
        print(new_df.shape)
        return new_df