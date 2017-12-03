import sys
import re

# uniquename: dengnan
# Name: Nan Deng

def main():
	trainfile=sys.argv[1]
	testfile=sys.argv[2]

	unique_word=[] #21162
	unique_tag=[] #46
	word_tag_count={} #21162
	tag_count={} #46
	tag_tag_count={} #46

	##################################################   Part 1   ################################################################

	with open(trainfile,'rU') as train_file:
		content=train_file.readlines()
		for sentence in content:
			sentence='start/s '+sentence
			tag_last='s'
			for pair in sentence.split():
				word = pair.split('/')[0]
				tag = pair.split('/')[-1]
				if '|' in tag:
					tag = tag.split('|')[0]

				if word not in unique_word:
					unique_word.append(word)

				if tag not in unique_tag:
					unique_tag.append(tag)
					tag_count[tag]=1
				else:
					tag_count[tag]+=1

				if word not in word_tag_count:
					word_tag_count[word]={}
					word_tag_count[word][tag]=1
				elif tag not in word_tag_count[word]:
					word_tag_count[word][tag]=1
				else:
					word_tag_count[word][tag]+=1

				if tag_last not in tag_tag_count:
					tag_tag_count[tag_last]={}
					tag_tag_count[tag_last][tag]=1
				elif tag not in tag_tag_count[tag_last]:
					tag_tag_count[tag_last][tag]=1
				else:
					tag_tag_count[tag_last][tag]+=1

				tag_last = tag

	# print len(unique_word)
	# print len(unique_tag)
	# print len(word_tag_count)
	# print len(tag_count)
	# print len(tag_tag_count)

	transition={} #46
	emission={} #21162
	emission_base={}

	for key_tag_last in unique_tag:
		transition[key_tag_last]={}
		for key_tag in tag_tag_count[key_tag_last]:
			transition[key_tag_last][key_tag]=tag_tag_count[key_tag_last][key_tag]/float(tag_count[key_tag_last])

	# print transition

	for key_word in unique_word:
		emission[key_word]={}
		emission_base[key_word]={}
		for key_tag in word_tag_count[key_word]:
			emission[key_word][key_tag]=word_tag_count[key_word][key_tag]/float(tag_count[key_tag])

	for base_word in emission:
		score = 0
		for base_tag in emission[base_word]:
			if emission[base_word][base_tag] > score:
				score = emission[base_word][base_tag]
				emission_base[base_word] = base_tag

	# print emission_base
	# print emission

	# print len(transition)
	# print len(emission)

	print 'finish train'

	##################################################   Viterbi Algorithm & Accuracy   ################################################################

	word_count = 0
	accuracy_count = 0
	accuracy_base = 0
	smooth = 0


	with open(testfile,'rU') as test_file, open ('POS.test.out','w') as output_file:
		content=test_file.readlines()
		for sentence in content:

			word_in =[]
			tag_in =[]
			word_smooth = 0
			tag_smooth =0

			for pair in sentence.split():
				word_in.append(pair.split('/')[0])
				tag_in.append(pair.split('/')[-1])

			word_smooth = len(set(word_in))
			tag_smooth = len(set(tag_in))

			sentence='start/s '+sentence
			tag_last='s'

			for pair in sentence.split():
				word = pair.split('/')[0]
				tag_test = pair.split('/')[-1]
				score=0
				
				if word == 'start':
					continue
				
				word_count+=1

				if word not in word_tag_count:
					try:
						float(word)
						tag = 'CD'
					except:	
						if '-' in word:
							tag = 'JJ'
						elif ',' in word:
							tag = 'CD'
						elif ':' in word:
							tag = 'CD'
						elif word[-4:] == 'tive' or word[-4:] == 'able' or word[-4:] == 'tory' or word[-3:] == 'ous' or word[-2:] == 'al'or word[-2:] == 'ic':
							tag = 'JJ'
						elif word[-3:] == 'ize':
							tag = 'VB'
						elif word[-3:] == 'ing':
							try:
								score = 0
								for tag_new in ['JJ', 'VBG', 'NN']:
									if transition[tag_last][tag_new]>score:
										score = transition[tag_last][tag_new]
										tag = tag_new
							except:
								tag = 'JJ'
						elif word[-2:] == 'ly':
							tag = 'RB'
						elif word[0].isupper():
							tag = 'NP'
						elif word[-2:] == 'er':
							tag = 'JJR'
						elif word[-3:] == 'est':
							tag = 'JJS'
						elif "'" in word:
							tag = 'NP'
						elif word[-2:] == 'ed':
							try:
								score = 0
								for tag_new in ['JJ', 'VBD', 'VBN']:
									if transition[tag_last][tag_new]>score:
										score = transition[tag_last][tag_new]
										tag = tag_new
							except:
								tag = 'JJ'
						elif word[-1:] == 's':
							try:
								score = 0
								for tag_new in ['NNS', 'VBZ']:
									if transition[tag_last][tag_new]>score:
										score = transition[tag_last][tag_new]
										tag = tag_new
							except:
								tag = 'NNS'
						elif word[-2:] == 'th':
							tag = 'JJ'
						else:
							tag = 'NN'

					tag_last = tag

					# if tag_test != tag:
					# 	print 'word', word, '|', tag_test, tag

					if tag == tag_test:
						accuracy_count+=1

					continue

				for tag_potential in word_tag_count[word]:

					if tag_potential not in tag_tag_count[tag_last]:
						if (1/(tag_count[tag_potential]+tag_smooth))*(word_tag_count[word][tag_potential]+1)/float(tag_count[tag_potential]+word_smooth) > score:
							score = (1/(tag_count[tag_potential]+tag_smooth))*(word_tag_count[word][tag_potential]+1)/float(tag_count[tag_potential]+word_smooth)
							tag = tag_potential					
						continue

					if ((tag_tag_count[tag_last][tag_potential]+1)/float(tag_count[tag_last]+tag_smooth))*(word_tag_count[word][tag_potential]+1)/float(tag_count[tag_potential]+word_smooth) > score:
						score = ((tag_tag_count[tag_last][tag_potential]+1)/float(tag_count[tag_last]+tag_smooth))*(word_tag_count[word][tag_potential]+1)/float(tag_count[tag_potential]+word_smooth)
						tag = tag_potential
				
				tag_last = tag

				if tag == tag_test:
					accuracy_count+=1

				if tag == emission_base[word]:
					accuracy_base+=1

				output_file.write('{}/{} '.format(word,tag))
			output_file.write('\n')

	accuracy = accuracy_count/float(word_count)
	baseline = accuracy_base/float(word_count)

	print 'Test Accuracy',accuracy
	print 'Baseline Accuracy',baseline

if __name__=='__main__':
	main()




# with open('POS.train','rU') as train_file:	
# 	words=[]
# 	tags=[]
# 	# sentence_level=[]
# 	word_tag=[]	
# 	content=train_file.readlines()
# 	for sentence in content[:50]:
# 		sentence='s/s '+sentence
# 		# whole=[]
# 		for pair in sentence.split():
# 			# whole.append(pair.split('/'))
# 			words.append(pair.split('/')[0])
# 			tags.append(pair.split('/')[-1])
# 			word_tag.append(pair.split('/'))
		# sentence_level.append(whole)
# unique_tag=[]
# for tag in tags:
# 	if tag not in unique_tag and '|' not in tag:
# 		unique_tag.append(tag)
# unique_word=[]
# for word in words:
# 	if word not in unique_word:
# 		unique_word.append(word)
# unique_tag=set(tags)
# unique_word=set(words)
# T=len(unique_tag) #46
# W=len(unique_word) #21163

# total_count=[]
# for tag in unique_tag:
# 	total=0
# 	for i in tags:
# 		if i==tag:
# 			total+=1
# 	total_count.append((tag,total))

# 


# copy_tags=tags[:]
# copy_tags.append('.')
# copy_tags.pop(0)
# bigrams=zip(tags,copy_tags)
# unique_bigram=set(bigrams)
# bigram_count=[]
# for bigram in unique_bigram:
# 	total=0
# 	for i in bigrams:
# 		if i==bigram:
# 			total+=1
# 	bigram_count.append((bigram,total))
# print bigram_count

# word_count=[]
# for word in unique_word:
# 	for tag in unique_tag:
# 		for each in word_tag:
# 			if each[0]==word and each[1]==tag:
# 				total+=1
# 			word_count.append(([word,tag],total))
# print word_count

