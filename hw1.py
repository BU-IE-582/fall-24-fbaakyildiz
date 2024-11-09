# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv("hw1_input.csv")

# Standardize the data (PCA is sensitive to the scale of features)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Initialize PCA and fit the model
pca = PCA()
pca.fit(data_scaled)

# Explained variance and cumulative variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Creating a DataFrame for explained variance and cumulative variance
explained_variance_df = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance))],
    'Explained Variance Ratio': explained_variance,
    'Cumulative Variance Ratio': cumulative_variance
})

# Loadings (Eigenvectors) - contribution of each feature to each principal component
loadings = pca.components_
loadings_df = pd.DataFrame(loadings, columns=data.columns)
loadings_df.index = [f'PC{i+1}' for i in range(len(loadings))]

# Displaying all rows and columns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Display the results
print("Explained Variance and Cumulative Variance for Each Principal Component:")
print(explained_variance_df)

print("\nLoadings (Eigenvectors) - Contribution of Each Feature to Each Principal Component:")
print(loadings_df)

# Reset display options if needed
pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")
