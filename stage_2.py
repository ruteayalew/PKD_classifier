import data_understanding as d2
class stage_2:
  def __init__(self, fname, lname):
    self.firstname = fname
    self.lastname = lname

    # Define CheckQualityTransformer
    class Check_Quality():
        def transform(self, X):
            d2.check_quality(X)
            return X

        def fit(self):
            return self
        
    class Data_Cleaning():
        def transform(self, X):
            d2.data_cleaning(X)
            return X

        def fit(self):
            return self
              

       
