#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Visualization1.png "Visualization"
[image2]: ./examples/Normalized.png "Normalized"
[image3]: ./examples/training-v-accuracy.png "Training-v-accuracy"
[image4]: ./examples/40.png "Traffic Sign 1"
[image5]: ./examples/3.png "Traffic Sign 2"
[image6]: ./examples/33.png "Traffic Sign 3"
[image7]: ./examples/17.png "Traffic Sign 4"
[image8]: ./examples/28.png "Traffic Sign 5"
[image9]: ./examples/softmax.png "Softmax results"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

The file you are reading is of course, the writeup. A link to the project [source code](https://github.com/Christopheraburns/carND-L2/Traffic_Sign_Classifier.ipynb)
Please note the python pickle files: train.p, test.p and valid.p are NOT stored in the code repo.  Downloading and unzipping them to the root of the project folder to run the notebook.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the basic python to provide a summary of the  data set:

* The number of items in the Training data set ?
* The number of items in the testing data set ?
* The number of items in the validation data set?
* The shape of a Training set data ?
* The shape of a Testing set data ?
* The shape of a Validation set data ?
* The number of unique classes/labels in the data set ?

####2. Include an exploratory visualization of the dataset.

As an exploratory visualization of the data set. I elected to display one image of each classification from each of the data sets. (This picture only includes the first 3 of 43)  I found this
to be very helpful as it showed me how the image quality can vary from one data set to another.  Viewing the data in this manner prompted me to use an equalization function on the images.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I began by shuffling the data sets

I then used the OpenCV Library to:
* Convert the image to Grayscale (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
* Equalize the image histogram (cv2.equalizeHist(image))

I then uses the Numpy library to:
* reshape each image (np.array) to 32,32,1 
* Normalize each image

Finally, I output the following data
* The standard deviation after normalization (usually consistent to within 5 thousandths)
* Verification of the data shapes after equalization and normalization (should be 32,32,1)



![alt text][image2]



####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following variation of the LeNet algorithm:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution     		| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 	    	| 1X1 stride, VALID padding, output 10X10X16    |
| RELU					|         										|
| Max Pooling			| 2X2 stride, outputs 5X5X16        			|
| Convolution			| 1X1 stride, VALUD padding, output 1X1X400 	|
| RELU					|												|
| DropOut				| DropOut probability set to .5					|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

After 30+ interations of parameters, I settled on the following parameter settings:

* epochs			60
* batch size 		100
* mu 				0
* sigma 			.09
* Learning rate 	.001
* Dropout setting 	.5

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
	I tried to use the original template that came with the lessons in Project 2.  I chose it for convenience.
* What were some problems with the initial architecture?
	I could not get a validation accuracy over 90%
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
	The initial model had over fitting issues.  The training data was consistently in the high 90 percentile, but the validation was very low 40-60 percentiles
	I utilized a modified LeNet CNN for my second and final option.  My validation came in in the low 90s and via parameter tuning I was able to get it over the required 93%
* Which parameters were tuned? How were they adjusted and why?
	Basically, after testing 30+ changes, It was the number of Epochs and batch size that had the most influence.

![alt text][image3]

	I plotted the Training and Validation accuracy with the MatPlotLib library and I saw that after 20 or 30 epochs the trend lines looked like they could climb higher with a few more epochs.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
	The LeNet CNN was used, although that architecture may have worked better with RGB images.
* Why did you believe it would be relevant to the traffic sign application?
	The images are very grainy and my using several convolutions I had hoped the primitive shapes would be more apt to be recognized
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 	It increased my validation accuracy by an average of 5 percent.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Roundabout mandatory  | Priority Road   								| 
| 60 km     			| 60 km 										|
| Right Turn ahead		| Right turn ahead								|
| No entry	      		| No entry					 					|
| Children Crossing		| Children Crossing     						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

| Image			        |    Best Guess	| Guess 2 | Guess 3 | Guess 4 |	Guess 5 |	
|:---------------------:|:--------------|---------|---------|---------|---------| 
| Roundabout mandatory  | 12 ()			| 38      | 35		| 42	  | 32		|
| 60 km     			| 3	 			| 10	  | 2		| 1  	  | 38		|
| Right Turn ahead		| 33 			| 11	  | 35		| 26	  | 1		|
| No entry	      		| 17			| 12	  | 13		| 14	  | 20		|
| Children Crossing		| 28			| 23	  | 30		| 35	  |	34		|	

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


