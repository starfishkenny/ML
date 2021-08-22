
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
      
      + > from sklearn.preprocessing import OrdinalEncoder
        
      + Assigning each unique value to a different integer 

      + Ordinal Variables (categorical variables with clear ordering)
        
      + Works well with tree-based models (DecisionTree, RandomForest...)

        * loop over the categorical variables and apply the ordinal encoder separately to each column

    - One-Hot Encoding
        
      + > from sklearn.preprocessing import OneHotEncoder

        * Set handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented in the training data
          
        * Setting sparse=False ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix)

        * supply only the categorical columns that we want to be one-hot encoded

      + Creating new columns indicating the presence (or absence) of each possible value in the original data

      + Nominal Variables (does not assume ordering)

      + Generally does not perform well if the categorical variable takes on a large number of values

## Pipelines
  
  * > from sklearn.pipeline import Pipeline

  * A simple way to keep your data preprocessing and modeling code organized

  * Benefits

    - Cleaner code

    - Fewer bugs

    - Easier to productionize

    - More options for model validation

  * Steps

    - Define preprocessing steps

      + > from sklearn.compose import ColumnTransformer

    - Define the model

    - Create & evaluate pipeline
      
## Cross-Validation

  * > from sklearn.model_selection import cross_val_score

  * Drawbacks of measuring model quality with validation data set

    - ex) From a dataset with 5000 rows, You will typically keep about 20% of the data as a validation dataset, or 1000 rows 
      
      + This leaves some random chance in determining model scores
        => A model might do well on one set of 1000 rows, even if it would be inaccurate on a different 1000 rows
      
    - Generally, the larger the validation set, the less "noise" there is in our measure of model quality
      
    - However, we can only get a large validation set by removing rows from our training data, and smaller training datasets mean worse models

  * So Cross-Validation!

    - Running our modeling process on different subsets of the data to get multiple measures of model quality
      
    - Begin by dividing the data into 5 pieces, each 20% of the full dataset (or, breaking the data into 5 "folds")

    - Hold out data from each n-th fold & use everything except the n-th fold for training the model 
      
      + The holdout set is then used to get a n-th estimate of model quality

      + Repeat this process, using every fold once as the holdout set
        
      + Putting this together, 100% of the data is used as holdout at some point, 
        ending up with a measure of model quality that is based on all of the rows in the dataset
    
  * Benefits & usage

    - Gives a more accurate measure of model quality (but could take longer time to run)
        
      + For **small** datasets, where extra computational burden isn't a big deal, you should run cross-validation

      + For **larger** datasets, a single validation set is sufficient (though defining small & large may differ)

      + <https://scikit-learn.org/stable/modules/model_evaluation.html> (evaluating model quality)

## XGBoost

  * > from xgboost import XGBRegressor

  * `Gradient boosting` = a method that goes through cycles to iteratively add models into an ensemble
    
    - Begins by initializing the ensemble with a single model, whose predictions can be pretty naive
    (Even if its predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors)
    
    - Then, start the cycle:
      
      + 1. Use the current ensemble to generate predictions for each observation in the dataset
           To make a prediction, add the predictions from all models in the ensemble
      
      + 2. These predictions are used to calculate a loss function (ex. mean squared error)
      
      + 3. Use the loss function to fit a new model that will be added to the ensemble
           Specifically, determine model parameters so that adding this new model to the ensemble will reduce the loss
           ("gradient" in "gradient boosting": refers to the fact that we'll use gradient descent on the loss function to determine the parameters in this new model)
      
      + 4. Add the new model to ensemble
      
      + 5. Repeat

  * ` XGBoost` = extreme gradient boosting

    - An implementation of gradient boosting with several additional features focused on performance and speed

    - Scikit-learn has another version of gradient boosting, but XGBoost has some technical advantages

    - Parameter tuning

      ```
      my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
      my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
      ```

      + **n_estimators**

        * Specifies how many times to go through the modeling cycle
        
        * It is equal to the number of models that we include in the ensemble

        * Too low a value causes underfitting, which leads to inaccurate predictions on both training data and test data

        * Too high a value causes overfitting, which causes accurate predictions on training data, but inaccurate predictions on test data

        * Typical values range from 100-1000, though this depends a lot on the **learning_rate**

      + **early_stopping_rounds**

        * Offers a way to automatically find the ideal value for n_estimators
        
        * Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for n_estimators
        
        * It's smart to set a high value for n_estimators and then use early_stopping_rounds to find the optimal time to stop iterating

        * Since random chance sometimes causes a single round where validation scores don't improve, you need to specify a number for how many rounds of straight deterioration to allow before stopping

        * When using early_stopping_rounds, also need to set aside some data for calculating the validation scores - this is done by setting the eval_set parameter
      
      + **learning_rate**

        * Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the learning rate) before adding them in

        * This means each tree we add to the ensemble helps us less
        
        * So, we can set a higher value for n_estimators without overfitting
        If we use early stopping, the appropriate number of trees will be determined automatically.

        * In general, a small learning rate and large number of estimators will yield more accurate XGBoost models, though it will also take the model longer to train since it does more iterations through the cycle

       + **n_jobs**

        * On larger datasets where runtime is a consideration, you can use parallelism to build your models faster
        
        * It's common to set the parameter n_jobs equal to the number of cores on your machine
        
        * On smaller datasets, this won't help
          
          - The resulting model won't be any better, so micro-optimizing for fitting time is typically nothing but a distraction
        
        * But, it's useful in large datasets where you would otherwise spend a long time waiting during the fit command.

## Data Leakage

  * Happens when your training data contains information about the target, but similar data will not be available when the model is used for prediction
  
  * This leads to high performance on the training set (and possibly even the validation data), but the model will perform poorly in production

  * In other words, leakage causes a model to look accurate until you start making decisions with the model, and then the model becomes very inaccurate

  * Two main types of leakage: target leakage and train-test contamination

  * **Target leakage**

    - Occurs when your predictors include data that will not be available at the time you make predictions

    - It is important to think about target leakage in terms of the timing or chronological order that data becomes available, 
    not merely whether a feature helps make good predictions

    - To prevent this type of data leakage, any variable updated (or created) after the target value is realized should be excluded

  * **Train-Test Contamination**

    - Occurs when you aren't careful to distinguish training data from validation data
    
    - `Validation` = a measure of how the model does on data that it hasn't considered before
    
      + Can corrupt this process in subtle ways if the validation data affects the preprocessing behavior = `train-test contamination`
    
    - If your validation is based on a simple train-test split, exclude the validation data from any type of fitting, including the fitting of preprocessing steps
      
      + This is easier if you use scikit-learn pipelines
      
      + When using cross-validation, it's even more critical that you do your preprocessing inside the pipeline