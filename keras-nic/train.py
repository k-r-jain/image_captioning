from caption_model import load_set, train_test_split, load_clean_descriptions, load_photo_features, create_tokenizer, seed, define_model, evaluate_model, data_generator
from pandas import DataFrame
from datetime import datetime
import keras
import os

# load (image, description) files
train = load_set('datasets/Flickr8k_text/Flickr_8k.trainImages.txt')
test = load_set('datasets/Flickr8k_text/Flickr_8k.testImages.txt')

# load descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
test_descriptions = load_clean_descriptions('descriptions.txt', test)

# flatten the description to a single list of descriptions
flat_list_train_descriptions = [desc for sublist in train_descriptions.values() for desc in sublist]
flat_list_test_descriptions = [desc for sublist in test_descriptions.values() for desc in sublist]

print('Descriptions: train=%d, test=%d' % (len(flat_list_train_descriptions), len(flat_list_test_descriptions)))

# load photo features
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
 
# define experiment
model_name = 'baseline_model'
verbose = 1

'''
n_epochs_lstm = 70
n_epochs_joint = 30
n_photos_per_update = 5
'''

n_epochs_lstm = 1
n_epochs_joint = 1
n_photos_per_update = 20
n_batches_per_epoch = int(len(train) / n_photos_per_update)
n_repeats = 1 #3
seed(1)

# Callback functions to show on tensorboard (tensorboard --logdir ./tensorboard/fashion)
tbLogDir = './tensorboard/my-nic/' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
tbCallBack = keras.callbacks.TensorBoard(log_dir=tbLogDir, histogram_freq=0, write_graph=True, write_images=True)

# run experiment
train_results, test_results = list(), list()
for i in range(n_repeats):
	
	# define the model
	model = define_model(vocab_size, max_length)

	# train only LSTM layers first
	for layer in model.layers:
		if layer.name not in ["lstm_2", "dense_3", "dense_4"]:
			layer.trainable = False

	print(model.summary())

	# fit model
	model.fit_generator(data_generator(train_descriptions, train_features, tokenizer, max_length, n_photos_per_update), steps_per_epoch=n_batches_per_epoch, epochs=n_epochs_lstm, verbose=verbose, callbacks=[tbCallBack])

	# train all layers
	for layer in model.layers:
		layer.trainable = True

	print(model.summary())

	# fit model
	model.fit_generator(data_generator(train_descriptions, train_features, tokenizer, max_length, n_photos_per_update), steps_per_epoch=n_batches_per_epoch, epochs=n_epochs_joint, verbose=verbose, callbacks=[tbCallBack])

    # save model to YAML
	model_yalm_file = 'processed-data/'+model_name+'.yalm'
	if os.path.exists(model_yalm_file):
		os.remove(model_yalm_file)
	model_yaml = model.to_yaml()
	with open(model_yalm_file, "w") as yaml_file:
	    yaml_file.write(model_yaml)

	# serialize weights to HDF5
	model_weight_file = "processed-data/model.h5"
	if os.path.exists(model_weight_file):
		os.remove(model_weight_file)

	model.save_weights(model_weight_file)
	print("Saved model to disk")
    