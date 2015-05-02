import sys
import csv

'''
data = open("10k output.txt", "r")
#data = csv.reader(in_file)
out_file = open("testout.csv", "w")
writer = csv.writer(out_file)
for row in data:
	value = data[row][13]
	writer.writerow(row)
in_file.close()    
out_file.close()
'''
csv.field_size_limit(sys.maxsize)

with open("10k output.txt", "rb") as infile, open("test_output.csv", "wb") as outfile:
    in_txt = csv.reader(infile, delimiter = ',')
    out_csv = csv.writer(outfile)
    out_csv.writerows(in_txt)
infile.close()    
outfile.close()
