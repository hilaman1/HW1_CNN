r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
For k = 1 we label an image based only on the closest image to it in the training set. In many cases this will suffice,
but in some cases relying on just 1 image is not good enough, and so increasing k up to k = 3 improves generalization according 
to the results we got. Statistically, a relatively small amount of closest images generalizes better than just 1 image.

If k is increased beyond that, the bigger k is, the more images that do not belong to the same label as the image are involved in the
labeling process of the image, and thus interfere with it, which causes the accuracy of the prediction to drop. 
We can see this explicitly in the results we obtained, and so we conclude that increasing k further 
(further than k = 3 in this case) will lead to worse generalization for unseen data.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
The ideal pattern to see in a residual plot is all residuals equal 0, meaning that $y - \hat{y}$ is 0, hence the prediction is perfect.
Based on the residual plots we got above we can say that fitting with nonlinear features improves the results
comparing the 2 plots, we see that the mse is lower than mse5, and rsq is bigger than rsq5 for both non linear features and when performing CV (which means we train on less data).
"""

part4_q2 = r"""
a.
We think you used `np.logspace` instead of `np.linspace` because if gives a wider range of values, additionally we used polynomial features in the model,
so the logspace could lower the complexity, hence generalize better.

b.
We trained the model with 2 arrays of hyper parameters degree_range of length 3 and lambda_range of length 20. 
in each iteration in the CV we take different combination of parameters(1 parameter from each array), it means we have 20 * 3 = 60 combinations.
additionally we divided the dataset to 3 folds(k=3) that means we did all the above 3 times.
In conclusion we fitted the model 60 * 3 = 180 times.
"""

# ==============
