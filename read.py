import sys
import csv

def get_training():

	filename = 'btrain.csv'
	pruning = 1

	if len(sys.argv) > 1:	
		filename = sys.argv[1]

	if len(sys.argv) > 2:
		pruning = sys.argv[2]

	f = open(filename)
	train = csv.reader(f)

	data = []
	for row in train:
		data.append(row)
	print data



get_training()