import numpy as np
from scipy.linalg import det, inv
from scipy.stats import multivariate_normal
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.integrate import nquad

# Setting the parameters for the Gaussian distributions and class priors
n_samples = 10000  # Total number of samples
P_L0 = 0.35  # Prior probability for class 0
P_L1 = 0.65  # Prior probability for class 1
m0 = np.array([-1, -1, -1, -1])  # Mean vector for class 0
C0 = np.array([[5, 3, 1, -1], [3, 5, -2, -2], [1, -2, 6, 3], [-1, -2, 3, 4]])  # Covariance matrix for class 0
m1 = np.array([1, 1, 1, 1])  # Mean vector for class 1
C1 = np.array([[1.6, -0.5, -1.5, -1.2], [-0.5, 8, 6, -1.7], [-1.5, 6, 6, 0],
               [-1.2, -1.7, 0, 1.8]])  # Covariance matrix for class 1

# Initializing random seed for reproducibility
np.random.seed(0)

# Defining the error function
def errorFn(x1, x2, x3, x4):
    x = [x1, x2, x3, x4]
    return min(multivariate_normal.pdf(x, mean=m0, cov=C0) * P_L0,
               multivariate_normal.pdf(x, mean=m1, cov=C1) * P_L1)

# Defining a function for likelihood ratio
def likelihood_ratio(samples, m0, C0, m1, C1, gamma):
    pdf_L0 = multivariate_normal.pdf(samples, mean=m0, cov=C0)
    pdf_L1 = multivariate_normal.pdf(samples, mean=m1, cov=C1)
    likelihood_ratio = pdf_L1 / pdf_L0
    predictions = likelihood_ratio > gamma
    return predictions, likelihood_ratio

# Function to calculate likelihood ratios with the common covariance matrix
def likelihood_ratio_common_cov(samples, m0, m1, common_cov):
    pdf_L0 = multivariate_normal.pdf(samples, mean=m0, cov=common_cov)
    pdf_L1 = multivariate_normal.pdf(samples, mean=m1, cov=common_cov)
    return pdf_L1 / pdf_L0

# Function to compute empirical error based on a given gamma
def empirical_error(gamma, likelihood_ratios, labels):
    predictions = likelihood_ratios > gamma
    error = np.mean(predictions != labels)
    return error

# Defining the function to compute K(beta) for the Chernoff bound
def K_beta(beta, m0, C0, m1, C1):
    diff = m0 - m1
    C_beta = (1 - beta) * C0 + beta * C1
    inv_C_beta = inv(C_beta)
    term1 = 0.5 * beta * (1 - beta) * np.dot(np.dot(diff.T, inv_C_beta), diff)
    term2 = 0.5 * np.log(det(C_beta) / (det(C0) ** (1 - beta) * det(C1) ** beta))
    return term1 + term2

# Defining the function to compute Pe(beta) for the Chernoff bound
def Pe_beta(beta, m0, C0, m1, C1):
    return np.exp(-K_beta(beta, m0, C0, m1, C1))

# Defining the function to calculate the expected risk
def expected_risk(gamma, B, likelihood_ratios, labels):
    # Predictions based on the likelihood ratio and gamma
    predictions = likelihood_ratios > gamma

    # Calculating false positive and false negative rates
    FP = np.sum((predictions == 1) & (labels == 0)) / np.sum(labels == 0)
    FN = np.sum((predictions == 0) & (labels == 1)) / np.sum(labels == 1)

    # Expected risk
    return P_L0 * FP + B * P_L1 * FN

# PART A

# Generating samples and their labels
samples, labels = [], []
for _ in range(n_samples):
    if np.random.rand() < P_L0:  # Decide to generate from class 0 based on prior
        samples.append(multivariate_normal.rvs(mean=m0, cov=C0))
        labels.append(0)
    else:  # Generate from class 1
        samples.append(multivariate_normal.rvs(mean=m1, cov=C1))
        labels.append(1)
samples = np.array(samples)
labels = np.array(labels)
print("Samples shape:", samples.shape)
print("Labels shape:", labels.shape)

# Computing likelihood ratios
_, likelihood_ratios = likelihood_ratio(samples, m0, C0, m1, C1, gamma=1)

# Finding the optimal gamma by minimizing the empirical error
result = minimize_scalar(empirical_error, bounds=(0, 1), args=(likelihood_ratios, labels), method='bounded')
optimal_gamma = result.x
min_error = result.fun
print("Optimal Gamma:", optimal_gamma)
print("Minimum Empirical Error:", min_error)

# Integration options
opts = {'epsabs': 1.e-2}

# Numerical integration over the feature space
#error, _ = nquad(errorFn, ranges=[[-5, 5], [-5, 5], [-5, 5], [-5, 5]], opts=opts)
#print("Numerical Minimum Error Rate:", error)

# PART B

# Plotting ROC Curve
fpr, tpr, thresholds = roc_curve(labels, likelihood_ratios)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.scatter(fpr[thresholds > optimal_gamma][-1], tpr[thresholds > optimal_gamma][-1], color='red',
            label=f'Optimal γ = {optimal_gamma:.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Plotting Error Rate vs Gamma Plot
gammas = np.linspace(0, 1, 500)
errors = [empirical_error(gamma, likelihood_ratios, labels) for gamma in gammas]
plt.figure(figsize=(10, 5))
plt.plot(gammas, errors, label='Empirical Error Rate')
plt.scatter(optimal_gamma, min_error, color='red', label=f'Minimum Error @ γ = {optimal_gamma:.3f}')
plt.xlabel('Gamma')
plt.ylabel('Error Rate')
plt.title('Error')
plt.show()

# Plot K(beta) as a function of beta
betas = np.linspace(0, 1, 500)
K_betas = [K_beta(beta, m0, C0, m1, C1) for beta in betas]
Pe_betas = [Pe_beta(beta, m0, C0, m1, C1) for beta in betas]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(betas, Pe_betas, label='Chernoff Bound')
plt.xlabel('Beta')
plt.ylabel('P(error)')
plt.title('Chernoff Bound as a function of Beta')
plt.legend()
plt.show()

# Finding beta that minimizes the Chernoff bound
min_Pe_index = np.argmin(Pe_betas)
optimal_beta = betas[min_Pe_index]
min_Pe = Pe_betas[min_Pe_index]

# Computing Bhattacharyya bound
bhattacharyya_bound = Pe_beta(0.5, m0, C0, m1, C1)

print(f"Optimal Beta: {optimal_beta:.4f}")
print(f"Minimum P(error) using Chernoff Bound: {min_Pe:.4f}")
print(f"Bhattacharyya Bound (Beta = 0.5): {bhattacharyya_bound:.4f}")

# PART C

# True covariance matrices, adjusted to diagonal based on true variances
C0_diagonal = np.diag(np.diag([[5, 3, 1, -1], [3, 5, -2, -2], [1, -2, 6, 3], [-1, -2, 3, 4]]))
C1_diagonal = np.diag(np.diag([[1.6, -0.5, -1.5, -1.2], [-0.5, 8, 6, -1.7], [-1.5, 6, 6, 0], [-1.2, -1.7, 0, 1.8]]))

print("Adjusted Diagonal Covariance Matrix for Class 0 (C0_diagonal):")
print(C0_diagonal)
print("\nAdjusted Diagonal Covariance Matrix for Class 1 (C1_diagonal):")
print(C1_diagonal)

# Computing likelihood ratios for all samples using diagonal covariance matrices
_, likelihood_ratios_diagonal = likelihood_ratio(samples, m0, C0_diagonal, m1, C1_diagonal, gamma=1)

# Optimize gamma to minimize the empirical error with diagonal covariance matrices
result_diagonal = minimize_scalar(empirical_error, bounds=(0, 1), args=(likelihood_ratios_diagonal, labels), method='bounded')
optimal_gamma_diagonal = result_diagonal.x
min_error_diagonal = result_diagonal.fun

# Compute ROC curve for diagonal covariance matrices
fpr_diagonal, tpr_diagonal, _ = roc_curve(labels, likelihood_ratios_diagonal)
roc_auc_diagonal = auc(fpr_diagonal, tpr_diagonal)

# Plotting ROC curve for diagonal covariance matrices
plt.figure(figsize=(10, 5))
plt.plot(fpr_diagonal, tpr_diagonal, label=f'Diagonal ROC Curve (area = {roc_auc_diagonal:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # diagonal line for reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Diagonal Covariance Matrices')
plt.legend(loc="lower right")
plt.show()

print(f"Optimal Gamma (Diagonal): {optimal_gamma_diagonal:.4f}")
print(f"Minimum Empirical Error (Diagonal): {min_error_diagonal:.4f}")

# PART D

# Separating the samples by class
samples_0 = samples[labels == 0]
samples_1 = samples[labels == 1]

# Calculating sample covariance matrices for each class
cov_0_sample = np.cov(samples_0, rowvar=False)
cov_1_sample = np.cov(samples_1, rowvar=False)

# Calculating the prior probabilities
prior_0 = len(samples_0) / len(samples)
prior_1 = len(samples_1) / len(samples)

common_cov = prior_0 * cov_0_sample + prior_1 * cov_1_sample

# Calculating likelihood ratios for all samples using the common covariance matrix
likelihood_ratios_common_cov = likelihood_ratio_common_cov(samples, m0, m1, common_cov)

# Optimizing gamma to minimize the empirical error with the common covariance matrix
result_common_cov = minimize_scalar(empirical_error, bounds=(0, 1), args=(likelihood_ratios_common_cov, labels), method='bounded')
optimal_gamma_common_cov = result_common_cov.x
min_error_common_cov = result_common_cov.fun

# Computing ROC curve with the common covariance matrix
fpr_common_cov, tpr_common_cov, _ = roc_curve(labels, likelihood_ratios_common_cov)
roc_auc_common_cov = auc(fpr_common_cov, tpr_common_cov)

# Plotting ROC curve for common covariance matrix
plt.figure(figsize=(10, 5))
plt.plot(fpr_common_cov, tpr_common_cov, label=f'Common Covariance ROC Curve (area = {roc_auc_common_cov:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Common Covariance Matrix')
plt.legend(loc="lower right")
plt.show()
print(f"Optimal Gamma (Common Covariance): {optimal_gamma_common_cov:.4f}")
print(f"Minimum Empirical Error (Common Covariance): {min_error_common_cov:.4f}")

# PART E

# Range of B values
B_values = np.linspace(0, 10, 100)
expected_risks = []

# Computing the expected risk for each value of B
for B in B_values:
    # Calculating gamma for the current value of B
    gamma = (P_L0 / P_L1) * B
    # Calculate the expected risk for the current gamma and B
    risk = expected_risk(gamma, B, likelihood_ratios, labels)
    expected_risks.append(risk)

# Plotting the expected risk as a function of B
plt.figure(figsize=(10, 5))
plt.plot(B_values, expected_risks, label='Expected Risk')
plt.xlabel('Cost B')
plt.ylabel('Expected Risk')
plt.title('Expected Risk as a Function of Cost B')
plt.legend()
plt.grid(True)
plt.show()
