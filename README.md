# Flowers Recognition

This repository contains some notebooks for addressing the [Kaggle Flower Recognition Challenge](https://www.kaggle.com/alxmamaev/flowers-recognition). The dataset consists of 4242 images of flowers, belonging to 5 different categories.

## tf_flowers_complete_tensorflow_dataset.ipynb
This notebook address the classification problem using the  [Tensorflow Dataset API](https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?version=stable). 

Two models are trained, both based on the same deep CNN architecture. The first one is a plain model trained without
any optimization, while the second one is trained using data augmentation and dropout. 

All the images of the dataset have been resized to 224x224 pixels and they have been standard normalized (e.g. each image has mean equal to 0 and std deviation equal to 1).

The data augmentation consists of random cropping and mirroring and it has been performed using the API exposed by the tensorflow.image module, which has also been used for performing the resizing and the standardization of all images.

## tf_flowers_complete_image_generator.ipynb
This notebook uses the [Image Preprocessing module](https://keras.io/preprocessing/image/) provided by Keras. 

As well as in ``tf_flowers_complete_tensorflow_dataset.ipynb``, even in this notebook two models have been trained using the same CNN architecture. The main difference from the former notebook is that in this case no image standardization has been performed and the data is fed to the model using the Keras ImageDataGenerator class, which has also been used for performing data augmentation.