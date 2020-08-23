# Face_recognition using keras

To develop a face detection system using FaceNet and an SVM classifier to identify people from photographs, We will use the pre-trained Keras FaceNet model provided by Hiroki Taniai in this tutorial. 

Downloaded the model file and placed it in our current working directory with the filename ‘facenet_keras.h5‘.

# Training data
It was trained on MS-Celeb-1M dataset and expects input images to be color, to have their pixel values whitened (standardized across all three channels), and to have a square shape of 160×160 pixels.

# Pre-trained models
we used FaceNet model which directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. Once this space has been produced, tasks such as face recognition, verification and clustering can be easily implemented using standard techniques with FaceNet embeddings as feature vectors.

Our method uses a deep convolutional network trained to directly optimize the embedding itself, rather than an intermediate bottleneck layer as in previous deep learning approaches. To train, we use triplets of roughly aligned matching / non-matching face patches generated using a novel online triplet mining method. The benefit of our approach is much greater representational efficiency: we achieve state-of-the-art face recognition performance using only 128-bytes per face.

# Face alignment using MTCNN
One problem is that the Dlib face detector misses some of the hard examples (partial occlusion, silhouettes, etc). This makes the training set too "easy" which causes the model to perform worse on other benchmarks. To solve this, other face landmark detectors has been tested. One face landmark detector that has proven to work very well in this setting is the Multi-task CNN. 

We can use the mtcnn library to create a face detector and extract faces for our use with the FaceNet face detector models in subsequent sections.

# Preprocessng

The first step is to load an image as a NumPy array, which we can achieve using the PIL library and the open() function. We will also convert the image to RGB, just in case the image has an alpha channel or is black and white.

Next, we can create an MTCNN face detector class and use it to detect all faces in the loaded photograph.The result is a list of bounding boxes, where each bounding box defines a lower-left-corner of the bounding box, as well as the width and height.

# Training data
First, all of the photos in the ‘train‘ dataset are loaded, then faces are extracted, resulting in 93 samples with square face input and a class label string as output. Then the ‘val‘ dataset is loaded, providing 25 samples that can be used as a test dataset.

Both datasets are then saved to a compressed NumPy array file called ‘5-celebrity-faces-dataset.npz‘ that is about three megabytes and is stored in the current working directory.

# Create face embeddings
The next step is to create a face embedding.The classifier model that we want to develop will take a face embedding as input and predict the identity of the face. The FaceNet model will generate this embedding for a given image of a face.

we used the FaceNet model to pre-process a face to create a face embedding that can be stored and used as input to our classifier model since FaceNet model is both large and slow to create a face embedding.

We therefore, pre-compute the face embeddings for all faces in the train and test (formally ‘val‘) sets in our 5 Celebrity Faces Dataset.

# Perform Face Classification
Used a Linear Support Vector Machine (SVM) when working with normalized face embedding inputs. This is because the method is very effective at separating the face embedding vectors. We can fit a linear SVM to the training data using the SVC class in scikit-learn and setting the ‘kernel‘ attribute to ‘linear‘. We may also want probabilities later when making predictions, which can be configured by setting ‘probability‘ to ‘True‘.

It is a good practice to normalize the face embedding vectors. It is a good practice because the vectors are often compared to each other using a distance metric.

In this context, vector normalization means scaling the values until the length or magnitude of the vectors is 1 or unit length. This can be achieved using the Normalizer class in scikit-learn. It might even be more convenient to perform this step when the face embeddings are created in the previous step.

# Evaluaton
First, we need to select a random example from the test set, then get the embedding, face pixels, expected class prediction, and the corresponding name for the class.
Next, we can use the face embedding as an input to make a single prediction with the fit model.
We predict both the class integer and the probability of the prediction.

A different random example from the test dataset will be selected each time the code is run. We then get the name for the predicted class integer, and the probability for this prediction.

