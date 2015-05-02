# Sean Bowen-Williams
# Asher Rieck
# Brian Tang
# EECS 349 Spring 2015

import sys
import math
import random
import copy
import operator
from __future__ import division
import time

from collections import Counter

############################################################################
class TreeNode():
	def predict(self, e):
		if isinstance(self, TreeLeaf):
			return self.result
		else:
			if self.numeric:
				if e[self.attr] <= self.splitval and '<=' in self.branches:
					return self.branches['<='].predict(e)
				elif '>' in self.branches:
					return self.branches['>'].predict(e)
				else:
					return '0'
			else:
				try:    
					out = self.branches[e[self.attr]].predict(e)
				except:
					return '0'
				return out

	def print_normal_form(self, path):
		if isinstance(self, TreeLeaf):
			if self.result == '1':
				print "("+ str(path) + ") OR"
				return path
			else:
				return False

		else:
			for branch_name, branch in self.branches.iteritems():
				if self.numeric:
					list_name = self.attr_name + " " + branch_name + " "+ str(self.splitval)
				else: 
					list_name = self.attr_name + " is " + branch_name
				new_path = path + [list_name]
				branch.print_normal_form(new_path)
		return path

	def list_nodes(self, nodes):
		if isinstance(self, TreeLeaf):
			nodes.append(self)
			return nodes
		nodes.append(self)
		for branch_name, branch in self.branches.iteritems():
			nodes = branch.list_nodes(nodes)
		return nodes

############################################################################

class TreeLeaf(TreeNode):
	def __init__(self, target_class):
		self.result = target_class

	def __repr__(self):
		return "This is a TreeLeaf with result: {0}".format(self.result)

	def fork(self):
		self.__class__ = MiddleTreeNode
		self.result = None
############################################################################

class MiddleTreeNode(TreeNode):
	def __init__(self, attr_arr):
		self.attr =  attr_arr[0]
		self.splitval =  attr_arr[1]
		self.numeric = attr_arr[2]
		self.attr_name = attr_arr[3]
		self.mode = attr_arr[4]
		self.branches = {}

	def toLeaf(self, target_attribute):
		self.__class__ = TreeLeaf
		self.result = target_attribute

	def add_branch(self, val, subtree, default):
		self.branches[val] = subtree

	def __repr__(self):
		return "\nThis node is a fork on {0}, with {1} branches.\nMost instances in it are {2}.".format(self.attr_name,len(self.branches),self.mode)
############################################################################
class Data:
	def __init__(self, filename, numeric_attrs, test=False):
		self.filename = filename
		self.numeric_attrs = numeric_attrs
		self.parse()
		if not test:
			self.fix_question()

	def parse(self):
		with open(self.filename) as f:
			original = f.read()

			self.instances = [line.split(',') for line in original.split("\n")]

			if self.instances[-1]==['']:
				del self.instances[-1]

			self.attr_names = self.instances.pop(0)
			for header in self.attr_names:
				self.attr_names[self.attr_names.index(header)] = header.strip(' ')


			for instance in self.instances:
				for a in range(len(self.numeric_attrs)):
					if instance[a] == '?':
						continue
					if self.numeric_attrs[a]:
						instance[a] = float(instance[a])


	def fix_question(self):
		groups = [0, 1]
		fill_values = 2*[[None] * len(self.attr_names)]
		for g in groups:
			group_instances = [e for e in self.instances if e[-1] == str(groups[g])]
			for attr in range(len(self.attr_names[:-1])):
				if self.numeric_attrs[attr]:
					not_missing = [e[attr] for e in group_instances if e[attr] != '?']      
					fill_values[g][attr] = int(sum(not_missing) / len(not_missing))
				else:
					nominal_vals = [e[attr] for e in group_instances]
					fill_values[g][attr] = Counter(nominal_vals).most_common()[0][0]

		for instance in self.instances:
			for attr in range(len(self.attr_names)):
				if instance[attr]=='?' and instance[-1]=='0':
					instance[attr] = fill_values[0][attr]
				elif instance[attr]=='?' and instance[-1]=='1':
					instance[attr] = fill_values[1][attr]
		return True

############################################################################

def getOccurences(data, attributes, target_attribute):
	occurences = {}
	i = attributes.index(target_attribute)
	for row in data:
		if row[i] in occurences:
			occurences[row[i]] += 1 
		else:
			occurences[row[i]] = 1
	return occurences

############################################################################

def entropy(data, attributes, target_attribute):

	dataEntropy = 0.0
	occurences = getOccurences(data, attributes, target_attribute)
	for freq in occurences.values():
		dataEntropy += (-freq/len(data)) * math.log(freq/len(data), 2) 
	return dataEntropy

############################################################################

def gain(data, attributes, attr, targetAttr, numeric_attrs):

	currentEntropy = entropy(data, attributes, targetAttr)
	subsetEntropy = 0.0
	i = attributes.index(attr)
	best = 0

	if numeric_attrs[i]:
		order = sorted(data, key=operator.itemgetter(i))
		subsetEntropy = currentEntropy
		for j in range(len(order)):
			if j==0 or j == (len(order)-1) or order[j][-1]==order[j+1][-1]:
				continue

			currentSplitEntropy = 0.0
			subsets = [order[0:j], order[j+1:]]

			for subset in subsets:
				setProb = len(subset)/len(order)
				currentSplitEntropy += setProb*entropy(subset, attributes, targetAttr)

			if currentSplitEntropy < subsetEntropy:
				best = order[j][i]
				subsetEntropy = currentSplitEntropy
	else:
		valFrequency = getOccurences(data, attributes, attr)

		for val, freq in valFrequency.iteritems():
			valProbability =  freq / sum(valFrequency.values())
			dataSubset     = [entry for entry in data if entry[i] == val]
			subsetEntropy += valProbability * entropy(dataSubset, attributes, targetAttr)
	
	return [(currentEntropy - subsetEntropy),best]

############################################################################
def selectAttr(data, attributes, target_attribute, numeric_attrs):
	best = False
	bestCut = None
	maxGain = 0
	for a in attributes[:-1]:
		newGain, cut_at = gain(data, attributes, a, target_attribute, numeric_attrs) 
		if newGain>maxGain:
			maxGain = newGain
			best = attributes.index(a)
			bestCut = cut_at
	return [best, bestCut]
############################################################################

def one_class(data):
	first_class = data[0][-1]
	for e in data:
		if e[-1]!=first_class:
			return False
	return True
############################################################################

def mode(data, index):  
	L = [e[index] for e in data]
	return Counter(L).most_common()[0][0]

############################################################################
def makeSplit(data, attr, splitval, numeric_attrs):
	isNum = numeric_attrs[attr]
	positive_count = 0

	if isNum:
		subsets = {'<=': [], 
					'>': []}
		for row in data:
			if row[-1]=='1':
				positive_count += 1
			if row[attr]<=splitval:
				subsets['<='].append(row)
			elif row[attr]>splitval:
				subsets['>'].append(row)
	else:
		subsets = {}
		for row in data:
			if row[-1]=='1':
				positive_count += 1
			if row[attr] in subsets:
				subsets[row[attr]].append(row)
			else:
				subsets[row[attr]] = [row]
	negative_count = len(data)-positive_count
	if positive_count > negative_count:
		majority = '1'
	else:
		majority = '0'

	out = {"splitOn": splitval, "branches": subsets, "numeric": isNum, "mode": majority}
	return out

############################################################################

def makeDecisionTree(data, attributes, default, target_attribute, iteration, numeric_attrs):
	iteration += 1

	if iteration > 10:
		return TreeLeaf(default)
	if not data:
		tree = TreeLeaf(default)
	elif one_class(data):
		tree = TreeLeaf(data[0][-1])
	else:
		best_attr = selectAttr(data, attributes, target_attribute, numeric_attrs)
		if best_attr is False:
			tree = TreeLeaf(default)

		else:
			split_examples = makeSplit(data, best_attr[0], best_attr[1], numeric_attrs) #new decision tree with root test *best_attr*
			best_attr.append(split_examples['numeric'])
			best_attr.append(attributes[best_attr[0]])
			best_attr.append(split_examples["mode"])
			tree = MiddleTreeNode(best_attr)
			for branch_lab, branch_examples in split_examples['branches'].iteritems():
				if not branch_examples:
					break
				sub_default = mode(branch_examples, -1)
				subtree = makeDecisionTree(branch_examples, attributes, sub_default, target_attribute, iteration, numeric_attrs)
				tree.add_branch(branch_lab, subtree, sub_default)
	return tree

############################################################################
def tree_accuracy(data, dt):
	count = 0
	correct_predictions = 0
	for row in data:
		count += 1
		pred_val = dt.predict(row)
		if row[-1]==pred_val:
			correct_predictions+=1
	accuracy = 100*correct_predictions/len(data)
	return accuracy

###########################################################################
def test_tree(data, dt):
	for row in data:
		row[-1] = dt.predict(row)
	return data
############################################################################
def prune_tree(tree, nodes, validation_examples, old_accuracy):
	percentage = 0.2
	nodes = random.sample(nodes, int(percentage*(len(nodes))))
	reduction_cap = 1000
	while reduction_cap >0:
		reduction = []
		for n in nodes:
			if isinstance(n, TreeLeaf):
				nodes.pop(nodes.index(n))
				continue
			else:
				target_class = n.mode
				n.toLeaf(target_class)
				new_accuracy = tree_accuracy(validation_examples, tree)
				diff = new_accuracy - old_accuracy
				reduction.append(diff)
				n.fork()
		if reduction != []:
			max_red_at = reduction.index(max(reduction))
			if isinstance(nodes[max_red_at], MiddleTreeNode):
				nodes[max_red_at].toLeaf(nodes[max_red_at].mode)
			nodes.pop(max_red_at)
			reduction_cap = max(reduction)
			old_accuracy = tree_accuracy(validation_examples, tree)
		else:
			reduction_cap = 0

	print "New accuracy: " + str(new_accuracy) + "%"
	return [tree, new_accuracy]

############################################################################
def main():
	now = time.time()
	
	train_file = 'btrainsmall.csv'
	validate_file = 'bvalidate.csv'
	test_file = 'btest.csv'
	target_attribute = "winner"
	pruning = 1

	if len(sys.argv) > 1:	
		train_file = sys.argv[1]

	if len(sys.argv) > 2:
		validate_file = sys.argv[2]

	if len(sys.argv) > 3:
		test_file = sys.argv[3]

	if len(sys.argv) > 4:
		target_attribute = sys.argv[4]

	if len(sys.argv) > 5:
		pruning = int(sys.argv[5])

	numeric_attrs = []
	
	with open(train_file) as f:
		original = f.read()

		instances = [line.split(',') for line in original.split("\n")]

		if instances[-1]==['']:
			del instances[-1]

		for i in range(len(instances[0])):
			try:
				if int(instances[1][i]):
					numeric_attrs.append(False)
			except ValueError:
				numeric_attrs.append(True)
		numeric_attrs[7] = True

	train_data = Data(train_file, numeric_attrs)
	validation_data = Data(validate_file, numeric_attrs)
	test_data = Data(test_file, numeric_attrs,  True)

	
	#build tree
	default = mode(train_data.instances, -1)
	learned_tree = makeDecisionTree(train_data.instances, train_data.attr_names, default, target_attribute, 0, numeric_attrs)
	
	print "Training file: " +str(train_file)
	train_accuracy = tree_accuracy(train_data.instances, learned_tree)
	print "Accuracy: " + str(train_accuracy)

	validation_accuracy = tree_accuracy(validation_data.instances, learned_tree)
	print "Validation Accuracy= " + str(validation_accuracy)
	print "Pre-pruning:\n"
	dnf = learned_tree.print_normal_form([])
	nodes = learned_tree.list_nodes([])
	print "Nodes:" + str(len(nodes))

	prePruningTime = time.time() - now	
	print "Pre-pruning Runtime = " + str(prePruningTime) + "\n"
	if pruning == 1:
		pruned_learned_tree = prune_tree(learned_tree, nodes, validation_data.instances, validation_accuracy)

		print "Post-pruning:\n"
		dnf_pruned = pruned_learned_tree[0].print_normal_form([])
		nodes_pruned = pruned_learned_tree[0].list_nodes([])
		print "Nodes:" + str(len(nodes_pruned))

	# print "Tested Set:\n"
	# tested_set = test_tree(test_data.instances, pruned_learned_tree[0])
	# print tested_set

	totalTime = time.time() - now
	print "\nDonezo:"
	print "Runtime = " + str(totalTime)
	print "Train accuracy: " + str(train_accuracy)
	print "Validation pre-pruning accuracy: " + str(validation_accuracy)
	print "Pre-pruning tree size: " + str(len(nodes))

	if pruning == 1:
		print "Validation post-pruning accuracy: " + str(pruned_learned_tree[1])
		print "Post-pruning tree size: " + str(len(nodes_pruned))

if __name__ == "__main__":
	main()
