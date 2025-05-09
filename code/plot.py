import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
import matplotlib.pyplot as plt
import os

# Create results folder if it doesn't exist
if not os.path.exists('../results'):
    os.makedirs('../results')

# Load label_list from test.csv
label_df = pd.read_csv('test.csv', index_col=0)

# Determine num_models dynamically
num_models = label_df.shape[0] - 1

# Extract predictions and true_labels
predictions = label_df.iloc[:num_models, :].values
true_labels = label_df.iloc[num_models, :].values

# Compute true accuracies
true_accuracies = [np.mean(predictions[i] == true_labels) for i in range(num_models)]

# Compute true rankings (higher accuracy -> lower rank number)
true_ranks = rankdata([-acc for acc in true_accuracies], method='average')

# Determine top-k for Jaccard coefficient (e.g., top 10% models)
# k = int(0.1 * num_models)
jaccards_k = 10
true_top_k = np.argsort(true_ranks)[:jaccards_k]

# Determine num_experiments by counting random{k}.csv files in model1 directory
experiment_files = [f for f in os.listdir('../experiments/model1') if f.startswith('random') and f.endswith('.csv')]
num_experiments = len(experiment_files)

# Determine num_sample_sizes from one CSV file
with open('../experiments/model1/random0.csv', 'r') as f:
    num_sample_sizes = len(f.readlines())

# Define sample sizes, starting from 35 with step 5
sample_sizes = 35 + 5 * np.arange(num_sample_sizes)

# Initialize arrays to store average Spearman and Jaccard coefficients
avg_spearman = np.zeros(num_sample_sizes)
avg_jaccard = np.zeros(num_sample_sizes)

# Process each sample size
for j in range(num_sample_sizes):
    spearman_coeffs = []
    jaccard_coeffs = []
    # Process each experiment
    for k in range(num_experiments):
        estimated_accuracies = []
        # Collect accuracies for all models at this sample size and experiment
        for i in range(num_models):
            with open(f'../experiments/model{i+1}/random{k}.csv', 'r') as f:
                lines = f.readlines()
                acc = float(lines[j].strip())
                estimated_accuracies.append(acc)
        # Compute estimated rankings (higher accuracy -> lower rank number)
        estimated_ranks = rankdata([-acc for acc in estimated_accuracies], method='average')
        # Compute Spearman coefficient between true and estimated ranks
        spearman_coeff = spearmanr(true_ranks, estimated_ranks)[0]
        spearman_coeffs.append(spearman_coeff)
        # Compute top-k for estimated rankings
        estimated_top_k = np.argsort(estimated_ranks)[:jaccards_k]
        # Compute Jaccard coefficient
        intersection = len(set(true_top_k) & set(estimated_top_k))
        union = len(set(true_top_k) | set(estimated_top_k))
        jaccard_coeff = intersection / union if union != 0 else 0
        jaccard_coeffs.append(jaccard_coeff)
    # Average coefficients over experiments
    avg_spearman[j] = np.mean(spearman_coeffs)
    avg_jaccard[j] = np.mean(jaccard_coeffs)

# Generate the Spearman plot
plt.figure(figsize=(8,6))
plt.plot(sample_sizes, avg_spearman, 'g-', label='SDS', marker='^')
plt.xlabel('Sample Size')
plt.ylabel('Spearman Coefficient')
plt.title('Spearman Coefficient of Ranking for FASHION-MNIST')
plt.grid(True)
plt.legend()
plt.ylim(0, 1)
plt.savefig('../results/sds_fashion_mnist_spearman.png')
plt.close()

# Generate the Jaccard plot
plt.figure(figsize=(8,6))
plt.plot(sample_sizes, avg_jaccard, 'g-', label='SDS', marker='^')
plt.xlabel('Sample Size')
plt.ylabel('Jaccard Coefficient')
plt.title('Jaccard Coefficient of Top-k Models for FASHION-MNIST')
plt.grid(True)
plt.legend()
plt.ylim(0, 1)
plt.savefig('../results/sds_fashion_mnist_jaccard.png')
plt.close()