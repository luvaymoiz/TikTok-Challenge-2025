import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Load Data
file_path = 'final_dataset.parquet'

if not os.path.exists(file_path):
	print(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct directory.")
	exit() 
else:
	try:
		df = pd.read_parquet(file_path)
		print(f"Successfully loaded '{file_path}'. Shape: {df.shape}")
		print(f"Number of rows (lines): {df.shape[0]}")
		print("First 5 rows of the dataset:")
		print(df.head())
	except Exception as e:
		print(f"Error loading Parquet file: {e}")
		exit()

# Feature Selection
features = [
	'rating',
	'text_len',
	'rating_deviation',
	'sentiment_polarity',
	'sentiment_subjectivity',
	'excessive_exclaim',
	'avg_rating',
	# 'num_of_reviews',
	'log_num_reviews',
	#'latitude',
	#'longitude',
	'price_encoded',
	'year',
	'month',
	'weekday',
	'hour',
	'cat_American restaurant',
	'cat_Coffee shop',
	'cat_Department store',
	'cat_Fast food restaurant',
	'cat_Grocery store',
	'cat_Hotel',
	'cat_Mexican restaurant',
	'cat_Other',
	'cat_Pizza restaurant',
	'cat_Restaurant',
	'cat_Shopping mall'
]

# Check if all required features exist in the DataFrame
missing_features = [f for f in features if f not in df.columns]
if missing_features:
	print(f"Error: Missing required features in the dataset: {missing_features}")
	print("Please ensure your 'final_dataset.parquet' contains these columns.")
	features = [f for f in features if f in df.columns] 
	if not features:
		print("No valid features remaining. Exiting.")
		exit()
	else:
		print(f"Proceeding with available features: {features}")

data = df[features].copy()

for col in features:
	if data[col].isnull().any():
		data[col] = data[col].fillna(data[col].mean())
		print(f"Filled NaN values in '{col}' with its mean.")

# Data Preprocessing (Scaling)
scaler = MinMaxScaler()
scaled_data_np = scaler.fit_transform(data)

# Convert to PyTorch tensors
scaled_data_tensor = torch.tensor(scaled_data_np, dtype=torch.float32)

# Split data into training and validation sets
X_train_tensor, X_val_tensor = train_test_split(scaled_data_tensor, test_size=0.2, random_state=42)

# Create DataLoader for batching
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, X_train_tensor) 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Autoencoder Model Definition (PyTorch) 
class Autoencoder(nn.Module):
	def __init__(self, input_dim, encoding_dim):
		super(Autoencoder, self).__init__()
		# Encoder
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, encoding_dim * 2),
			nn.LeakyReLU(0.01), 
			nn.Linear(encoding_dim * 2, encoding_dim),
			nn.LeakyReLU(0.01)   
		)
		
		self.decoder = nn.Sequential(
			nn.Linear(encoding_dim, encoding_dim * 2),
			nn.LeakyReLU(0.01),  
			nn.Linear(encoding_dim * 2, input_dim),
			nn.Sigmoid()
		)
		
	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

input_dim = X_train_tensor.shape[1]

# Dimensionality of the latent space 
encoding_dim = 10 

model = Autoencoder(input_dim, encoding_dim)
print("\nAutoencoder Model Summary (PyTorch):")
print(model) 

# Define Loss function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the Autoencoder
epochs = 50
print(f"\nTraining Autoencoder for {epochs} epochs...")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

for epoch in range(epochs):
	model.train() 
	running_loss = 0.0
	for data_batch, _ in train_loader: 
		data_batch = data_batch.to(device)

		optimizer.zero_grad() 
		outputs = model(data_batch)
		loss = criterion(outputs, data_batch) 
		loss.backward() 
		optimizer.step() 

		running_loss += loss.item() * data_batch.size(0)

	train_loss = running_loss / len(train_dataset)

	# Validation phase
	model.eval() 
	val_loss = 0.0
	with torch.no_grad(): 
		for data_batch_val, _ in val_loader:
			data_batch_val = data_batch_val.to(device)
			outputs_val = model(data_batch_val)
			loss_val = criterion(outputs_val, data_batch_val)
			val_loss += loss_val.item() * data_batch_val.size(0)
	val_loss = val_loss / len(val_dataset)

	print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

print("\nTraining complete.")

# Calculate Reconstruction Error (PyTorch) 
model.eval() 
all_reconstructions = []
with torch.no_grad():
	for i in range(0, len(scaled_data_tensor), batch_size):
		batch = scaled_data_tensor[i:i + batch_size].to(device)
		reconstructed_batch = model(batch)
		all_reconstructions.append(reconstructed_batch.cpu().numpy())

reconstructions_np = np.vstack(all_reconstructions)

# Calculate Mean Squared Error (MSE) between original and reconstructed data
reconstruction_errors = np.mean(np.square(scaled_data_np - reconstructions_np), axis=1)

# Z-scoring the Reconstruction Error 
z_scored_reconstruction_errors = zscore(reconstruction_errors)

# Add the z-scored errors to original dataframe
df['ae_reconstruction_error_zscore'] = z_scored_reconstruction_errors

print("\nReconstruction errors calculated and z-scored.")
print("First 10 reviews with their z-scored autoencoder reconstruction error:")
print(df[['ae_reconstruction_error_zscore'] + features].head(10))

# Identify potential anomalies
threshold = 2.0

# Add binary "is_outlier" column (1 if anomaly, 0 if normal)
df['is_outlier'] = np.where(
    df['ae_reconstruction_error_zscore'] > threshold,
    -1,   # outlier
    1     # normal
)
print(df['is_outlier'].value_counts())

anomalies = df[df['ae_reconstruction_error_zscore'] > threshold]
print(f"\nNumber of potential anomalies (z-score > {threshold}): {len(anomalies)}")

# print more anomalies and all of their original features (for viewing purposes)
if not anomalies.empty:
	print("\nprinting the top 20 anomalies with all their features (sorted by anomaly score):")

	pd.set_option('display.max_rows', 50)        
	pd.set_option('display.max_columns', None)    
	pd.set_option('display.width', 1000)          
	pd.set_option('display.max_colwidth', 150)   

	# Sort by the anomaly score to see the most anomalous first and show the top 20
	print(anomalies.sort_values(by='ae_reconstruction_error_zscore', ascending=False).head(20))

else:
	print("\nNo anomalies detected above the specified threshold.")

# Investigating the Latent Space
print("Investigating the Latent Space")

model.eval()
with torch.no_grad():
	encoded_data_tensor = model.encoder(scaled_data_tensor.to(device))
	encoded_data_np = encoded_data_tensor.cpu().numpy()

encoded_df = pd.DataFrame(encoded_data_np, columns=[f'latent_dim_{i+1}' for i in range(encoding_dim)])

combined_df = pd.concat([df[features].reset_index(drop=True), encoded_df], axis=1)

# Calculate the correlation matrix
correlation_matrix = combined_df.corr()

# Get the correlation of each latent dimension with the original features
latent_correlations = correlation_matrix[encoded_df.columns].loc[features]

print("\nCorrelation of Latent Dimensions with Original Features:")
print(latent_correlations)
print("\nStandard Deviation of each Latent Dimension:")
print(encoded_df.std())

df.to_parquet('dataset_with_anomaly_scores.parquet')
df.to_csv('dataset_with_anomaly_scores.csv', index=False)