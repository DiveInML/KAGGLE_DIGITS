#Imports
import numpy as np
import sys
import profile
import csv

#Variables
INPUT_FILE = "../datasets/sample_train.csv"
TEST_FILE = "../datasets/test.csv"
OUTPUT_FILE = "output.txt"
INPUT_SIZE = -1
ROW_SIZE=0
DATA = {}
center_dist={}
cluster_info={}
clusterList=[]

class Cluster():
	
	def __init__(self,id):
		self.memberList = []
		self.id=id

	def update_center(self):
		l = len(self.center)
		self.center = [0]*l
		for member in self.memberList:
			for i in range(0,ROW_SIZE):
				self.center[i]+=DATA[member][i]
		self.center=[(x*1.0)/(len(self.memberList)*1.0+0.0000001) for x in self.center]		
	
	def set_center(self,center):
		self.center=center

	def size(self):
		return len(self.memberList)

def random_centers():
	return np.random.randint(1,INPUT_SIZE,size=10)


def dist(a,b):
	l = len(a)
	d=0
	for i in range(0,l):
		d += (a[i]-b[i])**2
	return d
		
def init():
	global INPUT_SIZE,clusterList,DATA,ROW_SIZE,cluster_info,center_dist;
	clusterList = [Cluster(x+1) for x in range(0,11)]
	with open(INPUT_FILE) as f:
		for line in f:
			INPUT_SIZE+=1
			if INPUT_SIZE>1:
				curr_row=[int(x) for x in line.split(',')]
				DATA[INPUT_SIZE-1]=curr_row
				ROW_SIZE=len(DATA[INPUT_SIZE-1])
	INPUT_SIZE-=1
	centers=random_centers()
	with open(INPUT_FILE) as f:
		cluster_count = 0
		for x in range(1,INPUT_SIZE+1):
			if x in list(centers):
				cluster_count+=1
				clusterList[cluster_count].set_center(DATA[x])

	for x in range(1,INPUT_SIZE+1):
		center_dist[x]=sys.maxint	
		cluster_info[x]=-1
		
def print_results():
	total_size=0
	for x in range(1,11):
		total_size+=clusterList[x].size()
		print("Cluster Size of "+str(x) +" is " + str(clusterList[x].size()))
	print(total_size)
def clustering():
	global INPUT_SIZE,clusterList,DATA,ROW_SIZE,cluster_info,center_dist;
	change=0
	for x in range(1,INPUT_SIZE+1):
		curr_cluster = 0
		curr_cluster_change=0
		for y in range(1,11):
			new_dist = dist(DATA[x],clusterList[y].center)
			if new_dist < center_dist[x]: 
				center_dist[x]=new_dist
				curr_cluster=y
				change=1
				curr_cluster_change=1
		if curr_cluster_change:		
			if cluster_info[x] != -1:
				clusterList[cluster_info[x]].memberList.remove(x)
			cluster_info[x]=curr_cluster
			clusterList[curr_cluster].memberList.append(x)
	if change==0:
		return 0			
	for x in range(1,11):
		clusterList[x].update_center()
	return 1

def test():
	global INPUT_SIZE,clusterList,DATA,ROW_SIZE,cluster_info,center_dist;
	results=[]
	with open(TEST_FILE) as f:
		next(f)
		line_num=0
		for line in f:
			line_num+=1
			curr_pixels = [int(x) for x in line.split(',')]
			final_dist = sys.maxint
			curr_cluster = 0
			for x in range(1,11):
				curr_dist = dist(curr_pixels,clusterList[x].center)
				if curr_dist < final_dist:
					final_dist=curr_dist
					curr_cluster=x		
			results.append((curr_cluster))
	# save results
	np.savetxt('output1000.csv', np.c_[range(1,line_num+1),results], 
		delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
	

def start():
	init()
	iter_cnt=0
	while 1:
		iter_cnt+=1
		if not clustering():
			print_results()
			break

if __name__ == '__main__':
	#profile.run('print start(); print test()')		
	start()
	test()