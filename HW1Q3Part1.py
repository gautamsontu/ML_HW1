import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numpy import linalg as LA

# Loading the dataset
wine_df = pd.read_csv(r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW1\wine+quality\winequality-white.csv',
                      delimiter=';')
wine_data = wine_df.to_numpy()
print("DataFrame shape:", wine_df.shape)

features_list = list(wine_df.columns[:-1])  # Excluding the last column which is wine quality
print("Features in order:", features_list)

# Setting up plot aesthetics for better visualization
plt.rc('font', size=18)
plt.rc('axes', titlesize=16, labelsize=14)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=10)
plt.rc('figure', titlesize=18)

def custom_format(x):
    if np.isclose(x, 0):
        return "0.000"
    else:
        return "{:0.3f}".format(x)

num_samples = wine_data.shape[0]  # Total number of samples
class_labels = wine_df.iloc[:, -1].values  # Extracting quality scores as class labels
features = wine_df.iloc[:, :-1].values  # Separating features
num_classes = 11  # Defining the number of quality classes
num_features = 11  # Defining the number of features

# Initializing arrays for mean vectors and covariance matrices
mean_vectors = np.zeros(shape=[num_classes, num_features])
covariance_matrices = np.zeros(shape=[num_classes, num_features, num_features])

# Calculating mean vectors and covariance matrices for each class
for i in range(num_classes):
    class_data = features[(class_labels == i)]
    mean_vectors[i, :] = np.mean(class_data, axis=0)

    # Handling classes not present in the dataset
    if i not in class_labels:
        covariance_matrices[i, :, :] = np.eye(num_features)
    else:
        # Regularizing covariance matrices to avoid ill-conditioning
        covariance_matrices[i, :, :] = np.cov(class_data, rowvar=False)
        regularization_term = 0.000000005 * (
                    np.trace(covariance_matrices[i, :, :]) / LA.matrix_rank(covariance_matrices[i, :, :])) * np.eye(
            num_features)
        covariance_matrices[i, :, :] += regularization_term

# Defining a loss matrix for 0-1 loss function
loss_matrix = np.ones(shape=[num_classes, num_classes]) - np.eye(num_classes)

# Computing Class-Conditional PDFs for each class
class_conditional_pdfs = np.zeros(shape=[num_classes, num_samples])
for i in range(num_classes):
    if i in class_labels:
        class_conditional_pdfs[i, :] = multivariate_normal.pdf(features, mean=mean_vectors[i, :],
                                                               cov=covariance_matrices[i, :, :])

# Estimating class priors based on sample frequencies
class_priors = np.zeros(shape=[num_classes, 1])
for i in range(num_classes):
    class_priors[i] = np.size(class_labels[np.where((class_labels == i))]) / num_samples

# Calculating total probability for normalization
total_probability = np.sum(class_conditional_pdfs * class_priors, axis=0)

# Computing posterior probabilities
class_posteriors = (class_conditional_pdfs * class_priors) / total_probability

# Determining the minimum expected risk for concluding
expected_risk = np.matmul(loss_matrix, class_posteriors)
decisions = np.argmin(expected_risk, axis=0)
print("Average Expected Risk/ Error Rate:", np.sum(np.min(expected_risk, axis=0)) / num_samples)

# Estimating the confusion matrix for visualization
confusion_matrix = np.zeros(shape=[num_classes, num_classes])
for decision in range(num_classes):
    for actual_label in range(num_classes):
        if actual_label in class_labels and decision in class_labels:
            confusion_matrix[decision, actual_label] = np.size(
                np.where((decision == decisions) & (actual_label == class_labels))) / np.size(
                np.where(class_labels == actual_label))

# Setting NumPy print options with the custom formatter
np.set_printoptions(formatter={'float_kind': custom_format})
print("Confusion Matrix:")
print(confusion_matrix)

# Reset NumPy print options to default if necessary
np.set_printoptions(edgeitems=3, infstr='inf',
                    linewidth=75, nanstr='nan', precision=8,
                    suppress=False, threshold=1000, formatter=None)

# Visualizing the data distribution in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting points for each quality class with distinct markers
unique_classes = np.unique(class_labels)
for quality in unique_classes:
    idx = class_labels == quality
    ax.scatter(features[idx, 0], features[idx, 1], features[idx, 2], label=f'Quality {quality}')

# Setting labels for axes and plot title
ax.set_xlabel('Feature 1 (Fixed Acidity)')
ax.set_ylabel('Feature 2 (Volatile Acidity)')
ax.set_zlabel('Feature 3 (Citric Acid)')
ax.legend(title="Wine Quality")
plt.title('White Wine Quality Data Distribution')
plt.show()