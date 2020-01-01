# Flowers Recognition

This repository contains some notebooks for addressing the [Kaggle Flower Recognition Challenge](https://www.kaggle.com/alxmamaev/flowers-recognition), which dataset consists of 4242 images of flowers, belonging to 5 different categories.

In all the notebooks, the dataset has been split into training, validation and test in the following percentages, respectively: 75%, 15% and 10%. All the images have also been resized to 224x224 pixels. For evaluating the trained models, the adopted metrics are precision, recall and F1 score, all both micro and macro averaged. 

## tf_flowers_complete_tensorflow_dataset.ipynb
This notebook address the classification problem using the  [Tensorflow Dataset API](https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?version=stable). 

Two models are trained, both based on the same deep CNN architecture. The first one is a plain model trained without
any regularization, while the second one is trained using data augmentation and dropout. 

All the images have been standard normalized (e.g. each image has mean equal to 0 and std deviation equal to 1).

The data augmentation consists of random cropping and mirroring and it has been performed using the API exposed by the tensorflow.image module, which has also been used for performing the resizing and the standardization of all images.

The best model has been trained for 45 epochs with early stopping (patience=10) and scored the following results: 

|Average Type |Precision |Recall |F1
|--- |--- |--- |---
|Micro|0.87|0.87|0.87
|Macro|0.86|0.86|0.86





## tf_flowers_complete_image_generator.ipynb
This notebook uses the [Image Preprocessing](https://keras.io/preprocessing/image/) module provided by Keras. 

As well as in ``tf_flowers_complete_tensorflow_dataset.ipynb``, even in this notebook two models have been trained using the same CNN architecture. The main difference from the former notebook is that in this case no image standardization has been performed and the data is fed to the model using the Keras ImageDataGenerator class, which has also been used for performing data augmentation. Moreover, the data augmentation in this case is performed using zooming, rotation, horizontal mirroring and brightness adjustment. 

The best model has been trained for 45 epochs with early stopping (patience=10) and scored the following results: 

|Average Type |Precision |Recall |F1
|--- |--- |--- |---
|Micro|0.78|0.78|0.78
|Macro|0.79|0.79|0.78
