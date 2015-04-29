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
		row = [x.strip(' ') for x in row]
		data.append(row)

	headers = data[0]
	data.remove(data[0])

	header = {}
	for i in range(len(headers)):
		header[headers[i]] = i

	print header['winner']

#	with open('btrain2.csv','w') as csvfile:
#		writer = csv.writer(csvfile,delimiter=',')
#		for i in range(0,len(data)):
#			prin = True
#			for j in range(0,len(headers)):
#				if(data[i][j] == '?'):
#					prin = False
#			if(prin):
#				writer.writerow(data[i])


get_training()
