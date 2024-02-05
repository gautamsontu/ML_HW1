import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix

# Ensuring consistent results across runs
np.random.seed(7)

# Mapping numerical activity labels to descriptive names
activity_labels = {
    1: 'Walking',
    2: 'Walking Upstairs',
    3: 'Walking Downstairs',
    4: 'Sitting',
    5: 'Standing',
    6: 'Laying'
}

# Setting up plot aesthetics for better visualization
plt.rc('font', size=18)
plt.rc('axes', titlesize=16, labelsize=14)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=10)
plt.rc('figure', titlesize=18)

# Loading the dataset with sensor features and corresponding activity labels
features_path = r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW1\human+activity\UCI HAR Dataset\UCI HAR Dataset\train\X_train.txt'
labels_path = r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW1\human+activity\UCI HAR Dataset\UCI HAR Dataset\train\Y_train.txt'
features = np.loadtxt(features_path)
labels = np.loadtxt(labels_path)

# Defining functions for Bayesian classification
def calculate_posterior(sample, mean_vector, cov_matrix, prior):
    """Calculates the posterior probability for a given sample."""
    likelihood = multivariate_normal.pdf(sample, mean=mean_vector, cov=cov_matrix)
    posterior = likelihood * prior
    return posterior

def bayesian_classifier(sample, class_stats, total_samples):
    """Determines the class of a sample using Bayesian decision theory."""
    max_posterior = -np.inf
    predicted_class = None
    for label, stats in class_stats.items():
        prior = stats['num_samples'] / total_samples
        posterior = calculate_posterior(sample, stats['mean_vector'], stats['cov_matrix'], prior)
        if posterior > max_posterior:
            max_posterior = posterior
            predicted_class = label
    return predicted_class

# Function to visualize data in 3D PCA-reduced space
def visualize_data_3d_pca(features, labels):
    """Creates a 3D scatter plot of the PCA-reduced feature set."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    for label, activity_name in activity_labels.items():
        class_data = features[labels == label, :3]  # Using the first three principal components
        ax.scatter(class_data[:, 0], class_data[:, 1], class_data[:, 2], label=activity_name)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.legend(title="Activity")
    plt.title('3D Scatter Plot of PCA-Reduced Data from HAR Dataset')
    plt.show()

# Normalizing features and applying PCA
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)
pca = PCA(n_components=10)  # Using the top 10 principal components
principal_components = pca.fit_transform(normalized_features)

# Preparing for Bayesian classification
num_classes = int(np.max(labels) - np.min(labels) + 1)
class_labels = np.arange(np.min(labels), np.max(labels) + 1)
mean_vectors = np.zeros((num_classes, pca.n_components_))
covariance_matrices = np.zeros((pca.n_components_, pca.n_components_, num_classes))
regularization_param = 0.000000005  # Preventing ill-conditioned matrices
for i, class_label in enumerate(class_labels):
    class_specific_data = principal_components[labels == class_label]
    mean_vectors[i] = np.mean(class_specific_data, axis=0)
    covariance_matrices[:, :, i] = np.cov(class_specific_data.T) + regularization_param * np.eye(pca.n_components_)

# Estimating priors and calculating posteriors
class_priors = np.array([np.mean(labels == class_label) for class_label in class_labels])
class_cond_likelihoods = np.array([
    multivariate_normal.pdf(principal_components, mean=mean_vectors[i], cov=covariance_matrices[:, :, i])
    for i in range(num_classes)])
class_posteriors = np.diag(class_priors).dot(class_cond_likelihoods)

# Initializing class statistics
class_stats = {
    class_label: {
        'num_samples': np.sum(labels == class_label),
        'mean_vector': mean_vectors[int(class_label - np.min(labels))],
        'cov_matrix': covariance_matrices[:, :, int(class_label - np.min(labels))]
    } for class_label in class_labels
}
num_samples = principal_components.shape[0]
# Performing classification and calculating error rate
predicted_labels = [bayesian_classifier(sample, class_stats, num_samples) for sample in principal_components]
accuracy = np.mean(predicted_labels == labels)
error_rate = 1 - accuracy
print(f"Average Expected Risk/ Error Rate: {error_rate:.4f}")

# Decision making based on MAP rule
decisions = np.argmax(class_posteriors, axis=0) + int(class_labels[0])

conf_matrix = confusion_matrix(labels, predicted_labels, labels=class_labels)

# Setting the print precision to 3 decimal points and suppresses scientific notation for integers
np.set_printoptions(precision=3, suppress=True, formatter={'float': '{:0.3f}'.format})

# Normalizing the confusion matrix to show probabilities
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)

print("Confusion Matrix:")
print(conf_matrix_normalized)

# Reset NumPy print options to default if necessary
np.set_printoptions(edgeitems=3, infstr='inf',
                    linewidth=75, nanstr='nan', precision=8,
                    suppress=False, threshold=1000, formatter=None)

# Function to visualize the data
visualize_data_3d_pca(principal_components, labels)