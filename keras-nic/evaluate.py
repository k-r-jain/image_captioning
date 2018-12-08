from keras.models import model_from_yaml
from nltk.translate.bleu_score import corpus_bleu
from caption_model import word_for_id, load_set, train_test_split, load_clean_descriptions, load_photo_features, create_tokenizer, seed, define_model, data_generator
from numpy import array
from numpy import argmax
from pandas import DataFrame
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from pickle import load
from random import seed
import os

from keras.models import model_from_yaml
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
# from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Embedding
from keras.layers.merge import concatenate
from keras.layers.pooling import GlobalMaxPooling2D

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text
 
# evaluate the skill of the model
def evaluate_model(model, descriptions_list, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	# step over the whole set
	total_1_blue_score = 0.00
	total_4_blue_score = 0.00
	img_count = 0

	for key, descriptions in descriptions_list.items():
		print key
		label_desc = list()
		blue_score = 0
		predict = generate_desc(model, tokenizer, photos[key], max_length).replace('startseq','').replace('endseq','')
		for desc in descriptions:
			print 'Actual: ' + desc
			label_desc.append(desc.replace('startseq','').replace('endseq','').split())
			# max 5 desc per image
			if len(actual) >= 5:
				break
		# calculate image BLEU score
		print 'Predicted: ' + predict
		blue_1_score = sentence_bleu(label_desc, predict.split(), smoothing_function=SmoothingFunction().method1)
		blue_4_score = sentence_bleu(label_desc, predict.split(), smoothing_function=SmoothingFunction().method4)
		print 'Blue 1 Score: %f' % blue_1_score
		print 'Blue 4 Score: %f' % blue_4_score
		total_1_blue_score = total_1_blue_score + blue_1_score
		total_4_blue_score = total_4_blue_score + blue_4_score
		img_count = img_count + 1
	# calculate total BLEU score
	bleu_1 = (total_1_blue_score / img_count) * 100.00
	bleu_4 = (total_4_blue_score / img_count) * 100.00
	print 'Avg BLUE-1 Score: %f' % bleu_1
	print 'Avg BLUE-4 Score: %f' % bleu_4
	return (bleu_1, bleu_4)

# # load dev set
# filename = 'datasets/Flickr8k_text/Flickr_8k.devImages.txt'
# dataset = load_set(filename)
# print('Dataset: %d' % len(dataset))

# # train-test split
# train, test = train_test_split(dataset)

# my change
train = load_set('datasets/Flickr8k_text/Flickr_8k.trainImages.txt')
# my change
test = load_set('datasets/Flickr8k_text/Flickr_8k.testImages.txt')

# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
test_descriptions = load_clean_descriptions('descriptions.txt', test)
flat_list_train_descriptions = [desc for sublist in train_descriptions.values() for desc in sublist]
flat_list_test_descriptions = [desc for sublist in test_descriptions.values() for desc in sublist]
print('Descriptions: train=%d, test=%d' % (len(flat_list_train_descriptions), len(flat_list_test_descriptions)))
# photo features
features_file = 'processed-data/features.pkl'
train_features = load_photo_features(features_file, train)
test_features = load_photo_features(features_file, test)
print('Photos: train=%d, test=%d' % (len(train_features), len(test_features)))
# prepare tokenizer
tokenizer = create_tokenizer(flat_list_train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max(len(s.split()) for s in list(flat_list_train_descriptions))
print('Description Length: %d' % max_length)
 

model_name = 'baseline_model'
model_yalm_file = 'processed-data/'+model_name+'.yalm'

# load YAML and create model
yaml_file = open(model_yalm_file, 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("processed-data/model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = evaluate_model(loaded_model, test_descriptions, test_features, tokenizer, max_length)

# # generate description
# key = "3482974845_db4f16befa"
# predicted = generate_desc(loaded_model, tokenizer, test_features[key], max_length)
# # store actual and predicted
# # print('Actual:    %s' % desc)
# print('Predicted: %s' % predicted)