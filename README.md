# Tau-Ethiopic-Digit-Recognition-using-Multicolumn-DNN
This project is a competition organzied by Pattern Recognition and Machine Learning 2020-2021 course at Tampere University. The topic is about recognize Ethiopic digit using machine learning method.
Our team using a Multi-column deep neural network [1]. Specific of our implementation is 5 columns of DNN with input size of images for each column are (28, 28), (28, 26). (28, 24), (28, 22), (28, 20). 
We have run on 50 epoch, batch size of 100, and annealing learning rate start with 0.001, multiplied with 0.993 after each epoch, until it reaches 0.00003. 
We have used Cross Entropy Loss function and Adam optimizer.

Reference: 
[1] Cireşan, Dan, and Ueli Meier. "Multi-column deep neural networks for offline handwritten Chinese character classification." 2015 international joint conference on neural networks (IJCNN). IEEE, 2015.
