# Sean Bowen-Williams
# Asher Rieck
# Brian Tang
# EECS 349 Spring 2015

from __future__ import division

import sys
import math
import random
import copy
import operator

from tree import TreeNode, TreeLeaf, MiddleTreeNode
from data import Data

import time

from collections import Counter



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

def calculateGain(data, attributes, attr, targetAttr, numeric_attrs):

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
	bestAttr = False
	bestCut = None
	
	#Edit this to change learning rate
	maxInfoGain = 0
	
	for a in attributes[:-1]:
		newGain, cut_at = calculateGain(data, attributes, a, target_attribute, numeric_attrs) 
		if newGain>maxInfoGain:
			maxInfoGain = newGain
			bestAttr = attributes.index(a)
			bestCut = cut_at
	return [bestAttr, bestCut]
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
def getAccuracy(data, dt):
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
				target_attribute = n.mode
				n.toLeaf(target_attribute)
				accuracy_update = getAccuracy(validation_examples, tree)
				difference = accuracy_update - old_accuracy
				reduction.append(difference)
				n.fork()
		if reduction != []:
			max_red = reduction.index(max(reduction))
			if isinstance(nodes[max_red], MiddleTreeNode):
				nodes[max_red].toLeaf(nodes[max_red].mode)
			nodes.pop(max_red)
			reduction_cap = max(reduction)
			old_accuracy = getAccuracy(validation_examples, tree)
		else:
			reduction_cap = 0

	print "Pruned Tree Accuracy: " + str(accuracy_update) + "%"
	return [tree, accuracy_update]

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
	train_accuracy = getAccuracy(train_data.instances, learned_tree)
	print "Accuracy: " + str(train_accuracy)

	validation_accuracy = getAccuracy(validation_data.instances, learned_tree)
	print "Validation Accuracy= " + str(validation_accuracy)
	print "Pre-pruning:\n"
	dnf = learned_tree.print_normal_form([])
	nodes = learned_tree.list_nodes([])
	print "Nodes:" + str(len(nodes))

	prePruningTime = time.time() - now	
	print "Pruned Runtime = " + str(prePruningTime) + "\n"
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
	print "Trained Accuracy: " + str(train_accuracy)
	print "Validation Unpruned Accuracy: " + str(validation_accuracy)
	print "Unpruned Tree Size: " + str(len(nodes))

	if pruning == 1:
		print "Validation Pruned Accuracy: " + str(pruned_learned_tree[1])
		print "Pruned Tree Size: " + str(len(nodes_pruned))

if __name__ == "__main__":
	main()
