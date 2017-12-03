import re
import math
from string import punctuation
import random
import sys

def main():
	file_in_name = sys.argv[1]
	file_out_name = file_in_name + '.out'

	with open(file_in_name,'rU') as train_file, open(file_out_name,'w') as output_file:
		content = train_file.read()
		content = content.split('\n\n')
		content = filter(None,content)
		count_instance = len(content)
		n = 5
		size =  -(-count_instance//n)
		accuracy = []

		# print count_instance, size
		# random.shuffle(content)

		for i in range(n):
			test = []
			train = []
			count_sense = {}
			count_feature_sense = {}
			p_sense = {}

			try:
				test = content[i*size:(i+1)*size]
				train = content[:]
				train[i*size:(i+1)*size] = []
			except:
				test = content[i*size:(count_instance+1)]
				train = content[:]
				train[i*size:(count_instance+1)] = []

			count_sense_total = len(train)
			count_test_total = len(test)

			# print count_sense_total
			# print count_test_total
			output = test[:]
			output_file.write('Fold ' + str(i+1) + '\n')

			for case in output:
				if case == '': continue
				case = case.split('\n')
				case = filter(None,case)
				title = re.findall('"(.*?)"',case[1])
				title = ' '.join(title)
				output_file.write(title + '\n')
			output_file.write('\n')

			for instance in train:
				if instance == '': continue
				instance = instance.split('\n')
				instance = filter(None,instance)
				sense = re.findall('%([^"]*)',instance[1])
				sense = ''.join(sense)
				words = instance[3].split()
				bag_train = []

				for j in words:
					if '<head>' in j:
						j = ''
					j = re.sub(r'[{}]+'.format(punctuation), '', j)
					if j != '':
						bag_train.append(j)

				bag_train = set(bag_train)

				if sense not in count_sense:
					count_sense[sense] = 1
					count_feature_sense[sense] = {}
				else:
					count_sense[sense] += 1

				for feature in bag_train:
					if feature not in count_feature_sense[sense]:
						count_feature_sense[sense][feature] = 1
					else:
						count_feature_sense[sense][feature] += 1

			# print count_sense
			# print count_feature_sense

			for key_sense in count_sense:
				p_sense[key_sense] = count_sense[key_sense]/float(count_sense_total)

			# print p_sense
			count_accuracy = 0

			for sample in test:
				if sample == '': continue
				sample = sample.split('\n')
				sample = filter(None,sample)
				sense_test = re.findall('%([^"]*)',sample[1])
				sense_test = ''.join(sense_test)
				words_test = sample[3].split()
				bag_test = []
				score = -1000000

				for j in words_test:
					if '<head>' in j:
						j = ''
					j = re.sub(r'[{}]+'.format(punctuation), '', j)
					if j != '':
						bag_test.append(j)

				feature_smooth = len(bag_test)

				for key_sense in p_sense:
					score_test = 0
					for feature_test in bag_test:
						if feature_test not in count_feature_sense[key_sense]:
							score_test += math.log(1/float(count_sense[key_sense]+feature_smooth))
						else:
							score_test += math.log((count_feature_sense[key_sense][feature_test]+1)/float(count_sense[key_sense]+feature_smooth))

					score_test += math.log(p_sense[key_sense])

					if score_test > score:
						score = score_test
						sense_potential = key_sense

				if sense_potential == sense_test:
					count_accuracy+=1
				else:
					print sense_potential, sense_test, '\n', sample[3], '\n'


			accuracy.append(count_accuracy/float(count_test_total))
			accuracy_aver = sum(accuracy)/n

		print accuracy
		print accuracy_aver

if __name__=='__main__':
	main()