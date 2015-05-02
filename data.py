from collections import Counter

class Data:
	def __init__(self, filename, numeric_attrs, test=False):
		self.filename = filename
		self.numeric_attrs = numeric_attrs
		self.parse()
		if not test:
			self.replace_missing_values()

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


	def replace_missing_values(self):
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