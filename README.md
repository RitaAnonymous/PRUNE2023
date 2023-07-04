# PRUNE2023
Code and data for PRUNE: A Patching Based Repair Framework for Certifiable Unlearning of Neural Networks
## Use PRUNE to unlearn

1.Randomly select data points in the training set to simulate data to unlearn.

2.Clustering of data points to unlearn is performed to select representative data points.(No clustering if class unlearning.)

3.Generate patch networks for these representative points to confound the model predictions for these data points.

