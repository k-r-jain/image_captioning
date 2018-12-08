'''
The dataset contains multiple descriptions for each photograph 
and the text of the descriptions requires some minimal cleaning.

First, we will load the file containing all of the descriptions
'''

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
filename = 'datasets/Flickr8k_text/Flickr8k.token.txt'
# load descriptions
doc = load_doc(filename)

'''
Each photo has a unique identifier. This is used in the photo filename and 
in the text file of descriptions. Next, we will step through the list of 
photo descriptions and save the descriptions for each photo. Below defines 
a function named load_descriptions() that, given the loaded document text, will 
return a dictionary of photo identifiers to descriptions.
'''

def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# remove filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# initialize desc list for each image
		if image_id not in mapping:
			mapping[image_id] = []
		# save desc
		mapping[image_id].append(image_desc)
	return mapping
 
# parse descriptions
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))

'''
The descriptions are already tokenized and easy to work with. 
We will clean the text in the following ways in order to reduce 
the size of the vocabulary of words we will need to work with:

Convert all words to lowercase.
Remove all punctuation.
Remove all words that are one character or less in length (e.g. a).
'''

import string
 
def clean_descriptions(descriptions_list):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	# for each list of description
	for imgKey, descriptions in descriptions_list.items():
		# for each letter
		descIdx = 0
		for desc in descriptions:
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# store as string
			descriptions_list[imgKey][descIdx] =  ' '.join(desc)
			# update descIdx
			descIdx = descIdx + 1
 
# clean descriptions
clean_descriptions(descriptions)
# summarize vocabulary
flat_list = [desc for sublist in descriptions.values() for desc in sublist]
'''
for sublist in l:
    for item in sublist:
        flat_list.append(item)
'''
all_tokens = ' '.join(flat_list).split()
vocabulary = set(all_tokens)
print('Vocabulary Size: %d' % len(vocabulary))

'''
Finally, we save the dictionary of image identifiers and descriptions 
to a new file named descriptions.txt, with one image identifier and description per line.
'''

# save descriptions to file, one per line
def save_doc(description_list, filename):
	lines = list()
	for image_key, descriptions in description_list.items():
		for desc in descriptions:
			lines.append(image_key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
 
# save descriptions
save_doc(descriptions, 'descriptions.txt')