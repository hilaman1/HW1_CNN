r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
'''
For k = 1 we label an image based only on the closest image to it in the training set. In many cases this will suffice, but in some cases relying on just 1 image is not good enough, and so increasing k up to k = 3 improves generalization according to the results we got. 
Statistically, a relatively small amount of closest images generalizes better than just 1 image.

If k is increased beyond that, the bigger k is, the more images that do not belong to the same label as the image are involved in the labeling process of the image, and thus interfere with it, which causes the accuracy of the prediction to drop. 
We can see this explicitly in the results we obtained, and so we conclude that increasing k further (further than k = 3 in this case) will lead to worse generalization for unseen data.
'''
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
'''

The selection of $\Delta>0$ is arbitrary for the SVM loss because we treat errors based on the $\lambda$ parameter. The errors won't be so bad if it's very small. 
We will concentrate more on minimizing $||w||$. Otherwise, each error has a greater weight.
There is a trade-off between the data loss and the regularization loss.
The role of $\Delta$ is similar but reversed - the smaller $\Delta$ we can violate more, and the factor of $||w||$ More meaning, which is like a big regularization.
That's why just changine the value is enough, and we can choose that $\Delta>0$.
'''
"""

part3_q2 = r"""
'''
1. In the sample image, the yellow area corresponds with the white area and the blue and green areas correspond with the black area, so the class gives a higher score.
We can see that the yellow lines represent the class number, and that's what the model is learning.
We can see that the images with the miss classification, the lighter areas correspond to the lighter areas of the chosen class.

2. KNN divides the training set according to the level of similarity to the current sample and makes a decision according to the inputs that are most similar to this image. 
On the other hand, SVM performs an initial division according to the distribution of the training set and adjusts the tested input to some of the regions.
Therefore, an input that is significantly different from the training set may produce worse results in SVM compared to KNN and on the other hand, inputs from close classes (for example 5 and 6) may be poorly classified by KNN.

'''

"""

part3_q3 = r"""
'''
1. We see a coordinated convergence of the train and the validation and since the train drops a little below the validation but they converge together, and thus this is a good learning rate.
If the interval between the train and the validation was larger and we would see a larger decrease in the loss of the train compared to fixation and even an increase in the loss of the validation, we would say that this is over fitting, but this is not the case.
The amount of data in the 5 epochs was enough to get to the convergence point of the model and the following epochs did not contribute significantly to the learning process.
Furthermore, to save running time, it is common to add a mechanism that interrupts the learning process if for several epochs the parameters do not change, as in this graph.
If the learning rate was too high, we would see a high instability of the loss function because we will miss the global minimum.
If it was too low the slope would be lower and might not convergence.

2. We see that the model is highly under fitting. The model converges neither in train nor in validation. 
This corresponds to a too-high learning rate. Another evidence of an excessively high learning rate is the instability in the validation, which indicates a movement around some local minimum point.
'''

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
'''
The ideal pattern to see in a residual plot is all residuals equal 0, meaning that $y - \hat{y}$ is 0, hence the prediction is perfect.
Based on the residual plots we got above we can say that fitting with nonlinear features improves the results comparing the 2 plots, we see that the mse is lower than mse5, and rsq is bigger than rsq5 for both non linear features and when performing CV (which means we train on less data).
'''
"""

part4_q2 = r"""
'''
a.
We think you used `np.logspace` instead of `np.linspace` because if gives a wider range of values, additionally we used polynomial features in the model, so the logspace could lower the complexity, hence generalize better.
b.
We trained the model with 2 arrays of hyper parameters degree_range of length 3 and lambda_range of length 20. 
in each iteration in the CV we take different combination of parameters(1 parameter from each array), it means we have 20 * 3 = 60 combinations. additionally we divided the dataset to 3 folds (k=3) that means we did all the above 3 times.
 In conclusion we fitted the model 60 * 3 = 180 times.
'''

"""

# ==============