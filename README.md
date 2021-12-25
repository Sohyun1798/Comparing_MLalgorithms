# Classification task for 3 datasets with 7 different machine learning algorithms
CMPUT566 project

Sohyun Park

CCID: sohyun2

sohyun2@ualberta.ca

## Goal of the project

In this project, I aim to track why a model is adequate for given datasets and the factors that optimise the model to achieve the results. To adress this, I chose three datasets representing three areas that deep neural networks usually apply - Vision, Natural Language Processing(NLP and Audio Speech. Unlike ordinary cases, I used seven machine learning techniques other than deep natural networks to the datasets: Linear Regression, Logistic Regression, Decision Tree, Support Vector Machine(SVM), Naive Bayes, k Nearest Neighbours(kNN) and Random Forest. Based on the distinct characteristics of each dataset, I found out more clearly what advantages and disadvantages exist when the seven algorithms are applied to each dataset. Additionally, the project has shown through experiments that we can achieve high performance without necessarily using deep natural networks when performing classification tasks with the three dataset types.

## Datasets

- Vision: MNIST

- NLP: IMDB review dataset

- Audio and speech: FSDD

## Machine Learning Models

1. Linear Regression

2. Rogistic Regression

3. Decision Tree

4. Support Vector Machine

5. k Nearest Neighbours

6. Naive Bayes

7. Random Forest

## Experiments

I trained the machine learning models on one machine without any GPU. The maximum time taken for the training step was approximately 527 seconds, while overall running time took about 937.068 seconds. The codes were written on Jupyter Notebook, which will be provided with this paper.

### 1. Vision

The provider of the MNIST dataset has already divided it into a training dataset and a test dataset. There are 60,000 examples in the training set and 10,000 examples in a test set. For the classification task, the model's input data is image converted into numerical values and normalised with value 256. The outputs are numbers in the range from 0 to 9.

There was no change from the basic setup under the scikit-learn framework regarding the settings of the adopted machine learning models. In contrast, n\_neighbors value for k Nearest Neighbours model and max\_depth value for Random Forest model set as ten because there are ten different values in the labels.

### 2. NLP

I used the **TfidfVectorizer** function provided by the scikit-learn library to vectorise the natural languages in the IMDB dataset. The vectorising review data goes through the models as inputs with a pipeline, and the outputs are two kinds of labels: positive(1) or negative(0).

The dataset was divided with rates 8 to 2, where eight is for training data, and two is for test data. Regarding the settings of the models, there was no change from the basic setup under the scikit-learn framework. In contrast, n\_neighbors value for k Nearest Neighbours model and max\_depth value for Random Forest model set as two because there are two different labels(positive or negative) in the dataset.

### 3. Audio Speech

I used the **librosa(==0.8.1)** library for pre-processing the FSDD dataset. I extracted mfcc from the audio data and converted it to numerical values with the librosa library. And then, the converted mfcc values got reshaped, making it from 3-dimensional data to 2-dimensional data. I employed mfcc values going through the process as inputs of the models for the audio classification task, and the outputs are numbers in the range from 0 to 9.

The dataset was divided with rates 8 to 2, where eight is for training data, and two is for test data. Regarding the settings of the models, there was no change from the basic setup under the scikit-learn framework. In contrast, n\_neighbors value for k Nearest Neighbours model and max\_depth value for Random Forest model set as ten because there are ten different values in the labels.

## Reference

[1]Shahadat Uddin, Arif Khan, Md Ekramul Hossain, and Mohammad Ali Moni. Comparing differ-ent supervised machine learning algorithms for disease prediction.BMC Medical Informaticsand Decision Making, 19, 2019.

[2]Sanghyeon An, Minjun Lee, Sanglee Park, Heerin Yang, and Jungmin So. An ensemble of simpleconvolutional neural network models for mnist digit recognition, 2020.

[3]Siyu Ding, Junyuan Shang, Shuohuan Wang, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang.Ernie-doc: A retrospective long-document modeling transformer, 2021.

[4]Royal Jain.  Improving performance and inference on audio classification tasks using capsulenetworks, 2019.

[5]Yann LeCun, Corinna Cortes, and CJ Burges.  Mnist handwritten digit database.ATT Labs[Online]. Available: http://yann.lecun.com/exdb/mnist, 2, 2010.

[6]Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and ChristopherPotts. Learning word vectors for sentiment analysis. InProceedings of the 49th Annual Meetingof the Association for Computational Linguistics: Human Language Technologies, pages 142–150,  Portland,  Oregon,  USA, June 2011. Association for Computational Linguistics.   URLhttp://www.aclweb.org/anthology/P11-1015.

[7]Zohar Jackson, César Souza, Jason Flaks, Yuxin Pan, Hereman Nicolas, and Adhish Thite.Jakobovski/free-spoken-digit-dataset: v1.0.8, August 2018. URLhttps://doi.org/10.5281/zenodo.1342401.
