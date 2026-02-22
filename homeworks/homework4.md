# Homework 4 (Learning from Demonstration with CNMPs)

In this homework, you will collect demonstrations that consist of (t, e_y, e_z, o_y, o_z) where e and o are the end-effector and the object cartesian coordinates with subscripts denoting the relevant axis. The code for collecting demonstrations is provided in `homework4.py`. The robot randomly moves its end-effector in the y-z plane, sometimes hitting the object and sometimes not. The height of the object is random and provided from the environment as well.

You will train a CNMP with the following dataset: {(t, e_y, e_z, o_y, o_z)_i, h_i}_{i=0}^N where h is the height of the object. Here, t will be the query dimension, h will be the condition to be given to the decoder, and other dimensions will be target dimensions while training the CNMP. In other words, given several context points (with all dimensions provided), the model will be asked to predict the end-effector and the object positions given the time and the height of the object.

Realize at least 100 tests with randomly generated observations and queries and compute the mean squared error between the predicted and the ground truth values. Plot these errors (mean and std) in a bar plot with two bars, 1 for the object and 1 for the end-effector. [Here is such a bar plot with 3 bars](https://github.com/yildirimyigit/pemp/blob/main/mindchange/loss_comparison_on_adroit_hammer.ipynb).

Note that in each test, the number of observations and queries can take random values between 1 and {n_context, n_target}.
