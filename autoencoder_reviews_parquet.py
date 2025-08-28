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

# --- 1. Load Data ---
file_path = 'final_dataset.parquet'

if not os.path.exists(file_path):
	print(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct directory.")
	exit() # Exit if file not found
else:
	try:
		df = pd.read_parquet(file_path)
		print(f"Successfully loaded '{file_path}'. Shape: {df.shape}")
		print(f"Number of rows (lines): {df.shape[0]}")
		print("First 5 rows of the dataset:")
		print(df.head())
	except Exception as e:
		print(f"Error loading Parquet file: {e}")
		exit() # Exit if loading fails

# --- 2. Feature Selection ---
features = [
	'rating',
	'text_len',
	'rating_deviation',
	'sentiment_polarity',
	'sentiment_subjectivity',
	'excessive_exclaim',
	'avg_rating',
	# 'num_of_reviews',
	# 'log_num_reviews',
	'latitude',
	'longitude',
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
	features = [f for f in features if f in df.columns] # Proceed with available
	if not features:
		print("No valid features remaining. Exiting.")
		exit()
	else:
		print(f"Proceeding with available features: {features}")

data = df[features].copy()

# Handle potential NaNs in features (e.g., fill with mean or median)
for col in features:
	if data[col].isnull().any():
		data[col] = data[col].fillna(data[col].mean())
		print(f"Filled NaN values in '{col}' with its mean.")

# --- 3. Data Preprocessing (Scaling) ---
scaler = MinMaxScaler()
scaled_data_np = scaler.fit_transform(data) # Keep as numpy for now

# Convert to PyTorch tensors
scaled_data_tensor = torch.tensor(scaled_data_np, dtype=torch.float32)

# Split data into training and validation sets
X_train_tensor, X_val_tensor = train_test_split(scaled_data_tensor, test_size=0.2, random_state=42)

# Create DataLoader for batching
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, X_train_tensor) # Input and target are the same for AE
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# --- 4. Autoencoder Model Definition (PyTorch) ---
class Autoencoder(nn.Module):
	def __init__(self, input_dim, encoding_dim):
		super(Autoencoder, self).__init__()
		# Encoder
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, encoding_dim * 2),
			nn.LeakyReLU(0.01),  # <--- Change to LeakyReLU
			nn.Linear(encoding_dim * 2, encoding_dim),
			nn.LeakyReLU(0.01)   # <--- Change to LeakyReLU
		)
		# Decoder (it's good practice to change it here too)
		self.decoder = nn.Sequential(
			nn.Linear(encoding_dim, encoding_dim * 2),
			nn.LeakyReLU(0.01),  # <--- Change to LeakyReLU
			nn.Linear(encoding_dim * 2, input_dim),
			nn.Sigmoid()
		)
		
	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

input_dim = X_train_tensor.shape[1]
encoding_dim = 10 # Dimensionality of the latent space (can be tuned)

model = Autoencoder(input_dim, encoding_dim)
print("\nAutoencoder Model Summary (PyTorch):")
print(model) # PyTorch prints module structure directly

# Define Loss function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 5. Training the Autoencoder ---
epochs = 50
print(f"\nTraining Autoencoder for {epochs} epochs...")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

for epoch in range(epochs):
	model.train() # Set model to training mode
	running_loss = 0.0
	for data_batch, _ in train_loader: # _ for target, which is same as input
		data_batch = data_batch.to(device)

		optimizer.zero_grad() # Clear gradients
		outputs = model(data_batch) # Forward pass
		loss = criterion(outputs, data_batch) # Calculate loss
		loss.backward() # Backward pass
		optimizer.step() # Update weights

		running_loss += loss.item() * data_batch.size(0)

	train_loss = running_loss / len(train_dataset)

	# Validation phase
	model.eval() # Set model to evaluation mode
	val_loss = 0.0
	with torch.no_grad(): # No gradient calculation needed
		for data_batch_val, _ in val_loader:
			data_batch_val = data_batch_val.to(device)
			outputs_val = model(data_batch_val)
			loss_val = criterion(outputs_val, data_batch_val)
			val_loss += loss_val.item() * data_batch_val.size(0)
	val_loss = val_loss / len(val_dataset)

	print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

print("\nTraining complete.")

# --- 6. Calculate Reconstruction Error (PyTorch) ---
model.eval() # Set model to evaluation mode
all_reconstructions = []
with torch.no_grad():
	for i in range(0, len(scaled_data_tensor), batch_size):
		batch = scaled_data_tensor[i:i + batch_size].to(device)
		reconstructed_batch = model(batch)
		all_reconstructions.append(reconstructed_batch.cpu().numpy())

reconstructions_np = np.vstack(all_reconstructions)

# Calculate Mean Squared Error (MSE) between original and reconstructed data
reconstruction_errors = np.mean(np.square(scaled_data_np - reconstructions_np), axis=1)

# --- 7. Z-scoring the Reconstruction Error ---
z_scored_reconstruction_errors = zscore(reconstruction_errors)

# Add the z-scored errors to your original DataFrame
df['ae_reconstruction_error_zscore'] = z_scored_reconstruction_errors

print("\nReconstruction errors calculated and z-scored.")
print("First 10 reviews with their z-scored autoencoder reconstruction error:")
print(df[['ae_reconstruction_error_zscore'] + features].head(10))

# Identify potential anomalies
threshold = 2.0
anomalies = df[df['ae_reconstruction_error_zscore'] > threshold]
print(f"\nNumber of potential anomalies (z-score > {threshold}): {len(anomalies)}")
# --- MODIFIED PART ---
# print more anomalies and all of their original features
if not anomalies.empty:
	print("\nprinting the top 20 anomalies with all their features (sorted by anomaly score):")

	# Set pandas print options to see all data without truncation
	pd.set_option('display.max_rows', 50)         # Show up to 50 rows
	pd.set_option('display.max_columns', None)    # Show all columns
	pd.set_option('display.width', 1000)          # Widen the output area
	pd.set_option('display.max_colwidth', 150)    # Show more text in the 'text' column

	# Sort by the anomaly score to see the most anomalous first and show the top 20
	print(anomalies.sort_values(by='ae_reconstruction_error_zscore', ascending=False).head(20))

else:
	print("\nNo anomalies detected above the specified threshold.")

	# --- 8. Investigating the Latent Space (Optional) ---
print("\n--- Investigating the Latent Space ---")

# First, get the encoded representation of the entire dataset
model.eval()
with torch.no_grad():
	# We only need the encoder part for this
	encoded_data_tensor = model.encoder(scaled_data_tensor.to(device))
	encoded_data_np = encoded_data_tensor.cpu().numpy()

# Create a DataFrame for the encoded dimensions
encoded_df = pd.DataFrame(encoded_data_np, columns=[f'latent_dim_{i+1}' for i in range(encoding_dim)])

# Combine with the original (unscaled) features to make interpretation easier
combined_df = pd.concat([df[features].reset_index(drop=True), encoded_df], axis=1)

# Calculate the correlation matrix
correlation_matrix = combined_df.corr()

# Get the correlation of each latent dimension with the original features
latent_correlations = correlation_matrix[encoded_df.columns].loc[features]

print("\nCorrelation of Latent Dimensions with Original Features:")
# print with a heatmap-style background for easier reading
print(latent_correlations)
print("\nStandard Deviation of each Latent Dimension:")
print(encoded_df.std())