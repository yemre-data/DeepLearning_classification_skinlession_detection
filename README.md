# DeepLearning_classification_skinlession_detection
## INTRODUCTION

Skin disorders are a type of disease that starts from benign scars and extends to the disease that causes permanent damage and even to skin cancer. Rapid and automated skin lesion detection holds great promise for early diagnosis and prevention with the development of computer vision systems. In this project, starting with simple artificial neural network models that can detect these diseases, continuing with complex transfer learning models which are EfficientNet,Resnet, and finally combining the two inputs with the combined method,it has been ensured that 3 diseases are output. By testing these models, it is also observed that different hyper parameters are experimented to understand their effects and how changes occur in the general model.
### What is the Skin Disorders?
Skin disorders are the superficial and deep diseases and differences that occur in the skin.
Skin disorders vary greatly in symptoms and severity. They can be temporary or permanent and may be painless or painful. [1]
### What is the Skin Lesions?
A lesion is any damage or abnormal change 
in the tissue of an organism, usually caused by disease or trauma.[2] A skin lesion is an abnormality in skin cells. Anything on the skin that changes the appearance of healthy skin.

### SCIENTIFIC PROBLEM
According to World Health Organization, skin diseases are among the most common of all human health afflictions and affect almost 900 million people in the world at any time.[3]  Acne, mycosis, herpes, atopic dermatitis, eczema, so many various forms which can have consequences on the quality of life of the people who suffer from them. As reported by Derma Survey in 2013, the average waiting time for regular visits is 36 days in Europe.[4] If we look for France, it is over 20 days. Detection and follow-up of dermatological diseases takes long days and patients wear out as psychology and physically in this process.

### METHODOLOGY
1)Problem Understanding/ Defining Research Question
How to reach the good accuracy on skin disorders medical images     multiple classification by applying deep learning methods?

1)Problem Understanding/ Defining Research Question
How to reach the good accuracy on skin disorders medical images     multiple classification by applying deep learning methods?

3) Data Preparation
Balance data
There are a few balancing methods. These methods are shown below.
a.Over-sampling
-Random Over-Sampling
SMOTE (Synthetic Minority Oversampling Technique)
-Data Augmentation
b.Under-sampling
c.Weighted Class Approach
-Exploratory Data Analysis for Metadata
The dataset also contains meta-information about the patient’s age group (from 0 to 85), the anatomical site (six possible sites) and the sex (male/female). The metadata has more empty columns so I started to drop them, as well as there were missing values. I filled them with median and most frequent values. In order to find important features for my future use, I found feature importance with a machine learning classification algorithm. These are shown in the figure below. We don't use it for a feature classification here because confirm type . Patients cannot perform their control without analysis (histopathology).

4) Modelling
I would largely focus on the EfficienNet and combined method[8] pretrained net with ImageNet dataset. This model family contains eight different models that are structurally similar and follow certain scaling rules for adjustment to larger image sizes. The smallest version B0 uses the standard input size 224 × 224. Larger versions, up to B7, use increased input size while also scaling up network width (number of feature maps per layer) and network depth (number of layers).[9] In this project, B4, which matches my image resolution, was used.

5) Combined Method(Mixed Input) and Evaluation
In this section, I would like to see the our importance feature effects on model. According to Gulli and Pal [10], mixed-data neural networks are more complicated in structure but they are more accurate at making predictions than those using only one type of data in the training process. Method working path is shown in the side graph. First we create MLP(multilayer perceptron) for metadata and second CNN model for image data then we are concatenate two architecture with their outputs. Finally we are adding one or two dense layer to get classification result.

### DEPLOYMENT AND RECOMMENDATIONS
Deployment is one of the most important aspects of the data science project. Many end users have difficulty using products made without an interface. There are many deployment method and libariries both Python and R for data science project. However, nowadays Streamlit is most famous one as you can see in Github Star Graph. So I decided to make my deployment part with Streamlit. 

### CONCLUSION
-List skills developed:
Learn building of Multiply Layer Perceptron, Convolutional, Neural Network, Transfer Learning.
To improve understanding of hyper parameters of Deep Learning methods.
PyCharm and git using
Anaconda environment problem solving skills
Communication
-Classification Model :
I experimented a lot of different architectures from the base Neural Network model to complex which are MLP,CNN, ResNet, EfficientNet.Like every model, my model needs improvement. I'm thinking of developing the model in the future.
-API(Application Programming Interface) of my Project

### ACKNOWLEDGEMENTS
Firstly I would like to extend my sincere thanks to Liliana Baquero, Romain Rouyer, Gabrielle Fauste, Anthony Jahn(Challenge Hub Team)  who gave me mentoring sessions and demo day exercise. I would also like to extend my deepest gratitude to Dr. Amodsen Chotia  who changed my perspective by mentoring me every week. And finally many thanks to Dr. Sophie Pène , she was my mainly supervisor of Challenge Hub.  
