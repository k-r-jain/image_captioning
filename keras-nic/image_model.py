import os
from pickle import dump
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.layers import Input
 
# extract features from each photo in the directory
def extract_features(directory):
	# load the model
	in_layer = Input(shape=(224, 224, 3))
	model = InceptionV3(include_top=False, weights='imagenet', input_tensor=in_layer)
	# print(model.summary())
	# extract features from each photo
	
	image_list = os.listdir(directory)
	number_of_images = len(image_list)
	features = dict()
	for counter, name in enumerate(image_list):
		# display progress
		if counter % 100 == 0:
			print('%.2f %% completed' % round(100 * counter/number_of_images,2))

		# load an image from file
		filename = directory + '/' + name
		image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		# print('>%s' % name)
	return features
 
# extract features from all images
directory = 'datasets/Flickr8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))

# save to file

image_features_file = 'processed-data/features.pkl'

if os.path.exists(image_features_file):
  os.remove(image_features_file)

dump(features, open(image_features_file, 'wb'))