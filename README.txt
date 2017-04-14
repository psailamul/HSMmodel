This directory contains python implementation of the HSM model.



Dependencies: numpy, scipy, param and Theano python packages. 
The first three are available via pip.



The directory contains three files:


TheanoVisionModel.py          - contains abstract class for implementing vision based system identification models using Theano library.

HSM.py                        - contains the actual implementation of the HSM model, as a subclass of TheanoVisionModel class.

fitting.py                    - contains example code how to fit the HSM model using scipy fmin_tnc method.
 This file contains a function 
				that takes a training set and inputs and validation set and inputs, and performs the model fitting.



Please note that soon a more complete library will be available via github. 

Visit http://antolik.net for more information.



