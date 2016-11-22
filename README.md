# Social Image Description Platform
## Overview 
This project presents a neural network that learns annotating an image with a set tags.

The neural network is also deployed on a photo sharing website where users can upload images.

### Architecture
The neural network is a multilabel classifier is composed of a deep convolution neural network and a multilayer perceptron.
First the image is sent to the CNN which embeds the image into a fix-length vector. Then, the multilayer perceptron predicts the labels that describe the best the image.
<div style="text-align:center">
<img src="https://raw.githubusercontent.com/Nlte/social-image-platform/master/docs/architecture.jpg"/>
</div>
The model uses transfer learning with Inception-v3 as it shows state of the art performances on the ILSVR challenge.
For more information you can check out this [tensorflow-tutorial](https://www.tensorflow.org/versions/r0.9/how_tos/image_retraining/index.html)

This github repo is divided into 2 parts : 
- img_platform : contains all the sources for the website.
- ml_core : contains the sources for the classifier. It can be run independently of the website.

## Running the neural network
Requirements
- tensorflow (tested on 0.11) 
- numpy

### Preprocess the data
This project is based on the MIRFLICKR 25K dataset.

>M. J. Huiskes, M. S. Lew (2008). The MIR Flickr Retrieval Evaluation. ACM International Conference on Multimedia Information Retrieval (MIR'08), Vancouver, Canada

NOTE : The dataset consists of images, annotations and metadata. It takes arround 3.5 GB on the hard-drive.

First, download the dataset : 
```sh
cd ml_core/data/
# download images and annotations. Outputs "annotations.json" file
./download_dataset.sh
```
Then, run the processing script :
```sh
python build_tfr_data.py
```
This script converts the annotations.json into tensorflow records files : `train-???-008.tfr` , `test-???-004.tfr` , `val-???-001.tfr` and store them into `output/`.
Each record file consists of proto examples containing the image name and the labels associated with it.

### Train the network
Training the full network end-to-end is computation intensive (one epoch takes arround 4h on a macbook pro with CPU only). Therefore in this project, we'll be only training the classifier that follows the CNN, there will be no fine-tuning of Inception.
We need to extract and save the cnn feature of each image.
To run the caching script :
```sh
cd ..
python cache_bottlenecks.py
```
This script will run each image in the CNN once to extract the image embedding vector. Each vector in stored in the output directory as `name-of-the-image.jpg.txt` . 
It takes arround 1h30 to process the 25000 images from the dataset.

Once the caching is done, we can train the classifier :

```sh
# The hyperparameters can be modified in the class ModelConfig of configuration.py
python train.py
```
Running the evaluation of the trained model is done with :
```sh
# The results will be saved under '<model_str>' in 'results.csv' file
python evaluate.py --model_str="1hidden-1500"
```
It is also posssible to run inference on an image with :
```sh
python inference.py --image="../docs/lake.jpg"
```
Below is an example of prediction
<div style="text-align:center">
<img src="https://raw.githubusercontent.com/Nlte/social-image-platform/master/docs/example_inference.png" width="300" height="300"/>
</div>

## Running the website

<img src="https://raw.githubusercontent.com/Nlte/social-image-platform/master/docs/frontpage.png" />


The website was built with the Django-Angular stack.

Requirements : 
- Django (tested on 1.10.2)
- djangorestframework (tested on 3.5.0)
- drf-nested-routers (tested on 0.11.1)
- tensorflow (tested on 0.11)
- node 
- bower 

```sh
cd img_platform/
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```
