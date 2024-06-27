# Model architecture:
## **Introduction to U-Net Architecture**:  

![image12](assets/pic12.png)

### General U-Net architecture**, 

- Generally has three key features: 

- The first is the symmetric structure which gives it the "U" shape and divides it into two main paths: the contracting (downsampling) path and the expanding (upsampling) path. 

- The second is the contracting path which is used for capturing the context in the image. 

- It consists of repeated applications of two 3x3 convolutions, each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation.

- At each downsampling step, the number of feature channels is doubled

- The third is the expanding path which enables precise localisation for accurate segmentation. 

- This consists of upsampling of the feature map followed by a 2x2 convolution (“up-convolution”) that halves the number of feature channels, a concatenation with the cropped feature maps from the contracting path, and two 3x3 convolutions, each followed by a ReLU. 

- The cropping is necessary due to the loss of border pixels in every convolution.

### My custom U-Net architecture**

![image14](assets/pic14.png)

- I have tailored the U-Net model to specifically address the challenges presented by COVID-19 CT scan segmentation.

- The model was defined with an input size of 128x128 pixels following the foundational structure of U-Net.

- The contracting path layers started with a 16-filter convolutional layer, introducing dropout after the first and each subsequent convolutional block to prevent overfitting. 

- This path doubles the filters while reducing the spatial dimensions to capture higher-level features. 

- For the bottleneck layer I we used a 256-filter convolutional block, serving as the bridge between the contracting and expanding paths. 

### Expanding Path Layers:

- The model uses transpose convolutions to upsample the feature maps, gradually reducing the number of filters. 

- Each upsampling step is followed by a concatenation with the feature map from the corresponding level to preserve spatial information lost during downsampling. 

- The output layer concludes with a 1x1 convolution that maps the final feature maps to the desired output segmentation map. 

- The use of a sigmoid activation function indicates our focus on binary classification (infected vs. non-infected areas). 

### Loss Functions and Metrics: 

- For the segmentation task, I employ a combination of loss functions, including Binary Crossentropy for its effectiveness in binary classification tasks, and Dice Loss to directly optimise for the overlap between the predicted segmentation and the ground truth

### Metrics: 

- Accuracy and the Dice coefficient are used as metrics

### Training and Validation: 

- The dataset is split into training, validation, and test sets. 60, 20, 20

- Also, employed data augmentation techniques to increase the diversity of the training dataset, helping the model to generalise better and reduce the risk of overfitting. 

