import sys
import csv
import math

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

#	with open('btrain2.csv','w') as csvfile:
#		writer = csv.writer(csvfile,delimiter=',')
#		for i in range(0,len(data)):
#			prin = True
#			for j in range(0,len(headers)):
#				if(data[i][j] == '?'):
#					prin = False
#			if(prin):
#				writer.writerow(data[i])

def entropy(header, data, target):
	frequency_val = {}
	data_entropy = 0.0
	i = 0
	try:
		i = header[target]
	except ValueError:
		pass

	for row in data:
		if frequency_val.has_key(row[i]):
			frequency_val[row[i]] += 1
		else:
			frequency_val[row[i]] = 1
	for freq in frequency_val.values():
		data_entropy += ((-freq/len(data)) * math.log(freq/len(data), 2))

	return data_entropy




get_training()

