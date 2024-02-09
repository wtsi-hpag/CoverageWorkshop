import numpy as np

class DataStruct:

	def __init__(self,filename,mem=0.999,skipper=300):
		idx = []
		cov = []
		prev = -1
		precov  =-1
		dataGap = 1000
		# mem = 0.999
		val = -1
		lineCount = 0
		with open(filename) as file:
			for line in file:
				if lineCount == 0:
					entries = line.rstrip().split("=")
					self.Mean = float(entries[1])
				else:
					entries = line.rstrip().split(' ')
					id = int(entries[1])
					jump = id - prev
					while prev != -1 and jump>dataGap:
						# print(id,prev,jump)
						prev+= dataGap
						jump -=dataGap
						val = mem*val

						if prev % skipper == 0:
							idx.append(prev)
							cov.append(val)

					# print(idx,prev,idx-prev)
					redval = int(entries[2])
					if redval > 300:
						redval = precov + np.random.randint(-10,10)
						redval = max(redval,0)
					# 	val = np.max(val,0)
					if val > 0:
						val = mem * val + (1.0 - mem) * redval
					else:
						val = redval

					if id % skipper == 0:
						idx.append(id)
						cov.append(int(round(val)))
					prev = id
					precov = val
				lineCount +=1
		self.Index = np.array(idx).reshape((len(idx),))
		self.Coverage = np.array(cov).reshape((len(idx),))
		# self.Mean = np.mean(self.Coverage)