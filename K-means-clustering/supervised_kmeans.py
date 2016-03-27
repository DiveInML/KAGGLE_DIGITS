#Imports
import numpy as np
import sys
import profile
import csv

#Variables
INPUT_FILE = "../datasets/sample_train.csv"
TEST_FILE = "../datasets/test.csv"
OUTPUT_FILE = "output.txt"
clusters={}
label_count=[0]*10

def dist(a,b):
	l = len(a)
	d=0
	for i in range(0,l):
		d += (a[i]-b[i])**2
	return d

def train():	
	with open(INPUT_FILE) as f:
		next(f)
		for line in f:
			curr_row = [int(x) for x in line.split(',')]
			label_count[curr_row[0]]+=1
			try:
				l = len(curr_row)-1
				for x in range(0,l):
					clusters[curr_row[0]][x]+=curr_row[x+1]
			except:
				clusters[curr_row[0]]=curr_row[1:]
		for x in range(0,10):
			for y in range(0,len(clusters[x])):
				clusters[x][y]=(clusters[x][y]*1.0)/(label_count[x]*1.0)


def test():
	results=[]
	with open(TEST_FILE) as f:
		next(f)
		line_num=0
		for line in f:
			line_num+=1
			curr_pixels = [int(x) for x in line.split(',')]
			final_dist = sys.maxint
			curr_cluster = 0
			for x in range(0,10):
				curr_dist = dist(curr_pixels,clusters[x])
				if curr_dist < final_dist:
					final_dist=curr_dist
					curr_cluster=x		
			results.append((curr_cluster))
	# save results
	np.savetxt('output1000.csv', np.c_[range(1,line_num+1),results], 
		delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
	
if __name__ == '__main__':
	#profile.run('print start(); print test()')		
	train()
	print(label_count)
	test()