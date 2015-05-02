class TreeNode():
	def predict(self, e):
		if isinstance(self, Leaf):
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

	def print_dnf(self, path):
		if isinstance(self, Leaf):
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
				branch.print_dnf(new_path)
		return path

	def list_nodes(self, nodes):
		if isinstance(self, Leaf):
			nodes.append(self)
			return nodes
		nodes.append(self)
		for branch_name, branch in self.branches.iteritems():
			nodes = branch.list_nodes(nodes)
		return nodes


class Leaf(TreeNode):
	def __init__(self, target_attribute):
		self.result = target_attribute

	def __repr__(self):
		return "Leaf: {0}".format(self.result)

	def fork(self):
		self.__class__ = MiddleNode
		self.result = None


class MiddleNode(TreeNode):
	def __init__(self, attr_arr):
		self.attr =  attr_arr[0]
		self.splitval =  attr_arr[1]
		self.numeric = attr_arr[2]
		self.attr_name = attr_arr[3]
		self.mode = attr_arr[4]
		self.branches = {}

	def toLeaf(self, target_attribute):
		self.__class__ = Leaf
		self.result = target_attribute

	def add_branch(self, val, subtree, default):
		self.branches[val] = subtree

	def __repr__(self):
		return "\nFork on {0}, {1} branches.".format(self.attr_name,len(self.branches))