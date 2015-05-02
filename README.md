Sean Bowen-Williams
Brian Tang
Asher Rieck

Program tested in python 2.7

To run the program type the following command:

python decisiontree.py

The program can also be run with the following paramters in this order:

python decisiontree.py training_file validate_file test_file target_attribute pruning

The training, validate, and test file parameters are paths to the appropriate files. Target attribute is the output attribute (i.e. winner). Pruning determines wether or not pruning is on (either 1 or 0).

If run with no parameters, the default values are:

train_file : btrainsmall.csv
validate_file : bvalidate.csv
test_file : btest.csv
target_attribute : winner
pruning : 1