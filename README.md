# Tau-Ethiopic-Digit-Recognition-using-Multicolumn-DNN
This project is a competition organzied by Pattern Recognition and Machine Learning 2020-2021 course at Tampere University. The topic is about recognizing Ethiopic digit in Afro-MNIST dataset [1] using machine learning method.

Our team using a Multi-column deep neural network [2]. Specific of our implementation is 5 columns of DNN with input size of images for each column are (28, 28), (28, 26). (28, 24), (28, 22), (28, 20). 

We have run on 50 epoch, batch size of 100, and annealing learning rate start with 0.001, multiplied with 0.993 after each epoch, until it reaches 0.00003. 
We have used Cross Entropy Loss function and Adam optimizer.

The repository needs to be organized.

Reference: 

[1] Wu, Daniel J., Andrew C. Yang, and Vinay U. Prabhu. "Afro-MNIST: Synthetic generation of MNIST-style datasets for low-resource languages." arXiv preprint arXiv:2009.13509 (2020).

[2] Cireşan, Dan, and Ueli Meier. "Multi-column deep neural networks for offline handwritten Chinese character classification." 2015 international joint conference on neural networks (IJCNN). IEEE, 2015.
