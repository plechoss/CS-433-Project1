# Project 1 : The Higgs boson
By Clara Bonnet, Lisa Dratva and Michal Pleskowicz.<br/>CS-433 @ EPFL

### Folder structure:
* scripts:
    * cross_validation.py : helper methods to perform cross-validation
    * data_preparation.py : helper methods for data loading and preprocessing
    * global_variables.py : defined global variables used throughout the project
    * implementations.py : implementations of all the required ML methods and more
    * performances.py : helper methods for assessing the accuracy of our predictions
    * run.py : our step-by-step data preprocessing and hyperparameter tuning as well as the final prediction pipeline
* data:
    * train.csv : training data
    * test.csv : testing data
    
### Usage:
Running the run.py file produces the final prediction in a submission.csv file. See the report for more details on the methods used, as well as the code itself for details of implementations. 

### Requirements:
We developed and tested the code with Python 3.7.1 and Numpy 1.15.4.