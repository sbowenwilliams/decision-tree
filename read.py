import sys
import csv
import math
import operator

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
		row = [x.strip(' ') for x in row]
		data.append(row)

	headers = data[0]
	data.remove(data[0])

	header = {}
	for i in range(len(headers)):
		header[headers[i]] = i

	print header['winner']
	print data[1][12]

	return header, data

def entropy(attributes, data, targetattribute):


    # Initialize variables
    frequencyvalues = {}
    dataentropy = 0.0
    i = 0
    try:
        i = attributes[targetattribute]
    except ValueError:
        pass

    # find the frequency of each entry in the dataset
    for entry in data:
        if frequencyvalues.has_key(entry[i]):
            frequencyvalues[entry[i]] += 1.0
        else:
            frequencyvalues[entry[i]] = 1.0

    # Calculate the entropy of each frequency value in dataset using
    # the entropy formula
    for freq in frequencyvalues.values():
        dataentropy += ((-freq/len(data)) * math.log(freq/len(data), 2))

    return dataentropy


header, data = get_training()
print entropy(header, data, 'weather')

