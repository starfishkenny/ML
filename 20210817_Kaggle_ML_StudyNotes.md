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
        
        + Assigning each unique value to a different integer 

        + Ordinal Variables (categorical variables with clear ordering)
        
        + Works well with tree-based models (DecisionTree, RandomForest...)

        + from sklearn.preprocessing import OrdinalEncoder

          - loop over the categorical variables and apply the ordinal encoder separately to each column

      - One-Hot Encoding

        + Creating new columns indicating the presence (or absence) of each possible value in the original data

        + Nominal Variables (does not assume ordering)

        + Generally does not perform well if the categorical variable takes on a large number of values

        + from sklearn.preprocessing import OneHotEncoder

          - Set handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented in the training data
          
          - Setting sparse=False ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix)

          - supply only the categorical columns that we want to be one-hot encoded

