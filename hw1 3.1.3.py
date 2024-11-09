import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data files
input_data = pd.read_csv("hw1_input.csv")
output_real = pd.read_csv("hw1_real.csv")
output_imag = pd.read_csv("hw1_img.csv")

# Concatenate input data with real and imaginary parts of output data
combined_data = pd.concat([input_data, output_real, output_imag], axis=1)

# Standardize the combined data (mean=0, variance=1)
scaler = StandardScaler()
combined_data_scaled = scaler.fit_transform(combined_data)

# Initialize PCA and fit the model to the scaled combined data
pca = PCA()
pca.fit(combined_data_scaled)

# Get the explained variance ratio and cumulative variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

# Create a DataFrame for explained variance and cumulative variance
explained_variance_df = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance))],
    'Explained Variance Ratio': explained_variance,
    'Cumulative Variance Ratio': cumulative_variance
})

# Get loadings (eigenvectors) for each principal component
loadings = pca.components_
loadings_df = pd.DataFrame(loadings, columns=combined_data.columns)
loadings_df.index = [f'PC{i+1}' for i in range(len(loadings))]

# Display results
print("Explained Variance and Cumulative Variance for Each Principal Component:")
print(explained_variance_df)

print("\nLoadings (Contribution of Each Feature to Each Principal Component):")
print(loadings_df)
