# Kaggle - Intro to Machine Learning

  ## How Models Work
  
    * Decision Tree
    
    * fitting (or training) = capturing patterns from data
    
    * leaf = bottom of the tree where we make our predictions
 
  ## Basic Data Exploration
  
    * shape
    
    * describe

  ## Your First Machine Learning Model


  ## Model Validation


  ## Underfitting and Overfitting


  ## Random Forests


  ## Machine Learning Competitions

    * Housing Prices Competition for Kaggle Learn Users

  ## Bonus Lesson: Intro to AutoML


# Kaggle - Intermediate Machine Learning

  ## Introduction

    * Review (RandomForest)

  ## Missing Values

    * Mainly 3 Approaches
    
      - Drop columns with missing values
    
      - Imputation
    
      - Imputation+

  ## Categorical Variables
    
    * Mainly 3 Approaches

      - Drop Categorical Variables

      - Ordinal Encoding
      
        + from sklearn.preprocessing import OrdinalEncoder
        
        + Assigning each unique value to a different integer 

        + Ordinal Variables (categorical variables with clear ordering)
        
        + Works well with tree-based models (DecisionTree, RandomForest...)

          - loop over the categorical variables and apply the ordinal encoder separately to each column

      - One-Hot Encoding
        
        + from sklearn.preprocessing import OneHotEncoder

          - Set handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented in the training data
          
          - Setting sparse=False ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix)

          - supply only the categorical columns that we want to be one-hot encoded

        + Creating new columns indicating the presence (or absence) of each possible value in the original data

        + Nominal Variables (does not assume ordering)

        + Generally does not perform well if the categorical variable takes on a large number of values



  ## Pipelines
  
    * from sklearn.pipeline import Pipeline

    * A simple way to keep your data preprocessing and modeling code organized

    * Benefits

      - Cleaner code

      - Fewer bugs

      - Easier to productionize

      - More options for model validation

    * Steps

      - Define preprocessing steps

        + from sklearn.compose import ColumnTransformer

      - Define the model

      - Create & evaluate pipeline
      
  ## Cross-Validation

    * from sklearn.model_selection import cross_val_score

    * Drawbacks of measuring model quality with validation data set

      - ex) From a dataset with 5000 rows, You will typically keep about 20% of the data as a validation dataset, or 1000 rows 
      
        + This leaves some random chance in determining model scores => A model might do well on one set of 1000 rows, even if it would be inaccurate on a different 1000 rows
      
      - Generally, the larger the validation set, the less "noise" there is in our measure of model quality
      
      - However, we can only get a large validation set by removing rows from our training data, and smaller training datasets mean worse models

    * So Cross-Validation!

      - Running our modeling process on different subsets of the data to get multiple measures of model quality
      
      - Begin by dividing the data into 5 pieces, each 20% of the full dataset (or, breaking the data into 5 "folds")

      - Hold out data from each n-th fold & use everything except the n-th fold for training the model 
      
        + The holdout set is then used to get a n-th estimate of model quality

        + Repeat this process, using every fold once as the holdout set
        
        + Putting this together, 100% of the data is used as holdout at some point, ending up with a measure of model quality that is based on all of the rows in the dataset
    
    * Benefits & usage

      - Gives a more accurate measure of model quality (but could take longer time to run)
        
        + For **small** datasets, where extra computational burden isn't a big deal, you should run cross-validation

        + For **larger** datasets, a single validation set is sufficient (though defining small & large may differ)
        
        + <https://scikit-learn.org/stable/modules/model_evaluation.html> (evaluating model quality)

  ## XGBoost
  

  ## Data Leakage
