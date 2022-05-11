# ML code adapted from https://github.com/spmallick/learnopencv/tree/master/Keras-ImageNet-Models

import numpy as np
# import prebuilt models and util functions
from tensorflow.keras.applications import mobilenet, vgg16
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class CatPredictor:

    def __init__(self):
        # init the models
        self.vgg_model = vgg16.VGG16(weights='imagenet')
        self.mobilenet_model = mobilenet.MobileNet(weights='imagenet')

    def load_batch_image(self, filename):
        """
        Loads an image for classificaiton
        Arguments:
            filename: a string of the file location
        Returns:
            The loaded image in batch format
        """
        # load an image in PIL format
        print("\nLoading image `" + filename + "`...")
        original = load_img(filename, target_size=(224, 224))
        print('PIL image size', original.size)
        # convert the PIL image to a numpy array
        # IN PIL - image is in (width, height, channel), in Numpy - image is in (height, width, channel)
        numpy_image = img_to_array(original)
        print('numpy array size', numpy_image.shape)
        # Convert the image / images into batch format
        # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
        image_batch = np.expand_dims(numpy_image, axis=0)
        print('image batch size', image_batch.shape)
        return image_batch

    def print_predictions(self, label):
        """
        Prints out the decoded predictions of an imagenet classifier
        Arguments:
            label: a label array in imagenet format (transformed with `decode_predictions()` from `imagenet_utils`)
        """
        for id in range(len(label[0])):
            print(label[0][id][1], ':', label[0][id][2])

    def predict(self, filename):
        """
        Predicts if an image is a cat
        Arguments:
            filename: a string of the file location
        Returns:
            Dictionary with one key for each model prediction result (`vgg16` and `mobilenet`)
        """
        # load the image in batch format
        image_batch = self.load_batch_image(filename)

        # VGG
        ###################################################
        # prepare the image for the VGG model
        processed_image = vgg16.preprocess_input(image_batch.copy())
        # get the predicted probabilities for each class
        predictions = self.vgg_model.predict(processed_image)
        # convert the probabilities to imagenet class labels
        label_vgg = decode_predictions(predictions)
        ###################################################

        # MobileNet
        ###################################################
        # prepare the image for the MobileNet model
        processed_image = mobilenet.preprocess_input(image_batch.copy())
        # get the predicted probabilities for each class
        predictions = self.mobilenet_model.predict(processed_image)
        # convert the probabilities to imagenet class labels
        label_mobilenet = decode_predictions(predictions)
        ###################################################

        return {
            'vgg16': label_vgg,
            'mobilenet': label_mobilenet
        }
