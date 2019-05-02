# One Shot Face-Recognition using Siamese Network
A Face Recognition Siamese Network implemented using Keras. Siamese Network is used for one shot learning which do not require extensive training samples for image recognition.

![App Demo](https://github.com/adityajn105/Face-Recognition-Siamese-Network/blob/master/screenshots/test1.png) 


## Getting Started
Here I will explain how to setup the environment for training and the run the face recognition app, also I will breif you about basics of One-Shot Learning and Siamese Network.

### Prerequisites
You will need Python 3.X.X with some packages which you can install direclty using requirements.txt.
> pip install -r requirements.txt

### Training the network:
1. Run the following file in src directory to download train and eval dataset. Both are fetched parallely and may take an hour.
> python datafetch.py

2. To train the model, run the following, model will be saved as "saved_best".
> python train.py

### Running the app
1. Use main.py to run the app which requires following flags '-db' database of faces, '-m' path to saved model, '-i' path of all images to detect
>  python main.py -db ../database -m ../saved_best -i ../myself.jpg image2.jpg

2. Currently app gives accuracy between 75% to 85% on eval data depending on the samples taken.

## One Shot Learning
Deep neural networks are really good at learning from high dimensional data like images or spoken language, but only when they have huge amounts of labelled examples to train on. Humans on the other hand, are capable of one-shot learning - if you take a human who’s never seen a spatula before, and show them a single picture of a spatula, they will probably be able to distinguish spatulas from other kitchen utensils with astoundingly high precision. 

Recently there have been many interesting papers about one-shot learning with neural nets and they’ve gotten some good results and one of them is using Siamese Network.

## Siamese Network
It is an approach to getting a neural net to do one-shot classification is to give it two images and train it to guess whether they have the same category. Then when doing a one-shot classification task described above, the network can compare the test image to each image in the support set, and pick which one it thinks is most likely to be of the same category. So we want a neural net architecture that takes two images as input and outputs the probability they share the same class.

If we just concatenate two examples together and use them as a single input to a neural net, each example will be matrix multiplied(or convolved) with a different set of weights, which breaks symmetry. Sure it’s possible it will eventually manage to learn the exact same weights for each input, but it would be much easier to learn a single set of weights applied to both inputs. So we could propagate both inputs through identical twin neural nets with shared parameters, then use the absolute difference as the input to a linear classifier - this is essentially what a siamese net is. Two identical twins, joined at the head, hence the name.

![Siamese Network](https://github.com/adityajn105/Face-Recognition-Siamese-Network/blob/master/screenshots/siamese.png) 

The output is squashed into [0,1] with a sigmoid function to make it a probability. We use the target t = 1 when the images have the same class and t = 0 for a different class. It’s trained with logistic regression. This means the loss function should be binary cross entropy between the predictions and targets. There is also a L2 weight decay term in the loss to encourage the network to learn smaller/less noisy weights and possibly improve generalization

When it does a one-shot task, the siamese net simply classifies the test image as whatever image in the support set it thinks is most similar to the test image

## Authors
* Aditya Jain : [Portfolio](https://adityajn105.github.io)

## Licence
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/adityajn105/Face-Recognition-Siamese-Network/blob/master/LICENSE) file for details

## Acknowledements
* [This article](https://sorenbouma.github.io/blog/oneshot/)


