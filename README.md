# Social Image Description Platform
This project is part of Udacity MLND program.
The report is available here : 

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

### References

- Ali Sharif Razavian, Hossein Azizpour, Josephine Sullivan and Stefan Carlsson. CNN Features off-the-shelf: an Astounding Baseline for Recognition. KTH, 2014. [[link]](https://arxiv.org/abs/1403.6382)
- Min-Ling Zhang and Zhi-Hua Zhou. A Review on Multi-Label Learning Algorithms. IEEE Transactions on Knowledge and Data Engineering, 2014. [[link]](http://cse.seu.edu.cn/people/zhangml/files/TKDE'13.pdf)
- Maxime Oquab, Leon Bottou, Ivan Laptev and Josef Sivic. Learning and Transferring Mid-Level Image Representations using Convolutional Neural Networks. In CVPR, 2014. [[link]](http://www.di.ens.fr/willow/pdfscurrent/oquab14cvpr.pdf)
- Oriol Vinyals, Alexander Toshev, Samy Bengio and Dumitru Erhan. Show and Tell: A Neural Image Caption Generator. In CVPR, 2014. [[link]](https://arxiv.org/abs/1411.4555)
- Mohammad S Sorower. A Literature Survey on Algorithms for Multi-label Learning. Oregon State University, 2010. [[link]](http://people.oregonstate.edu/~sorowerm/pdf/Qual-Multilabel-Shahed-CompleteVersion.pdf)
- Mark J. Huiskes and Michael S. Lew. The MIR Flickr Retrieval Evaluation. ACM International Conference on Multimedia Information Retrieval, 2008. [[link]](http://press.liacs.nl/mirflickr/mirflickr.pdf)
- Antonio Torralba and Alexei A. Efros. Unbiased Look at Dataset Bias. In CVPR, 2011. [[link]](https://people.csail.mit.edu/torralba/publications/datasets_cvpr11.pdf)
- Fei-Fei Li, Andrej Karpathy, Justin Johnson. Stanford Course CS231n: Convolutional Neural Networks for Visual Recognition.  [[Website]](http://cs231n.stanford.edu)
- Sebastian Ruder. An overview of gradient descent optimization algorithms. [[Website]](http://sebastianruder.com/optimizing)
- Chris Shallue. Show and Tell: A Neural Image Caption Generator. [[GitHub]](https://github.com/cshallue/models/tree/master/im2txt)



## Running the website

<img src="https://raw.githubusercontent.com/Nlte/social-image-platform/master/docs/frontpage.png" />


The website was built with the Django-Angular stack. There is no need to train a model before running the server. A pretrained model has already been saved in the repo.

Requirements : 
- Django (tested on 1.10.2)
- djangorestframework (tested on 3.5.0)
- drf-nested-routers (tested on 0.11.1)
- tensorflow (tested on 0.11)
- node [[download-page]](https://nodejs.org/en/download/)

It is preferable to create a virtual environment before proceeding to the installation of the website.
```sh
conda create --name platform  # or mkvirtualenv
conda activate platform
```
```sh
cd img_platform/
pip install -r requirements.txt  # install the django+packages
npm install -g bower  # install bower package manager
npm install
bower install
python manage.py runserver --noreload
```
