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
* report.pdf : report with details of methods and approaches used in this project
* Project1_CV : a Jupyter Notebook with all the cross-validation processing
### Usage:
Running the run.py file produces the final prediction in a submission.csv file. 

### Requirements:
We developed and tested the code with Python 3.7.1, Jupyter Notebook 5.7.4 and Numpy 1.15.4.