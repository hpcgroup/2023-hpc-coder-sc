import re
import matplotlib.pyplot as plt
import numpy as np

# Read the data from the file
file_path = 'run_ft.out'
try:
    with open(file_path, 'r') as file:
        data = file.read()
except FileNotFoundError:
    print(f"File {file_path} not found.")
    exit(1)

# Regular expression to extract the required values
pattern = re.compile(r'global batch\s+(\d+)/\s+\d+\s+\|.*?\| loss: ([\d.]+)')

# Lists to hold extracted values
global_batches = []
losses = []

# Extracting data from the file
for match in pattern.finditer(data):
    global_batches.append(int(match.group(1)))
    losses.append(float(match.group(2)))

# Check if data was extracted
if not global_batches or not losses:
    print("No data was extracted. Please check the file format and ensure it contains the expected information.")
    exit(1)

# Print the first few extracted data points for verification
print(f"Extracted {len(global_batches)} data points.")
print("Sample data points:")
for i in range(min(5, len(global_batches))):
    print(f"Global Batch: {global_batches[i]}, Loss: {losses[i]}")

# Function to smooth the data using moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Set the window size for smoothing
window_size = 10

# Apply smoothing
smoothed_losses = moving_average(losses, window_size)

# Adjust the x-axis to match the length of the smoothed data
smoothed_global_batches = global_batches[window_size - 1:]

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(global_batches, losses, label='Original Loss')
plt.plot(smoothed_global_batches, smoothed_losses, label='Smoothed Loss', color='orange')
plt.xlabel('Global Batch')
plt.ylabel('Loss')
plt.title('Loss per Global Batch')
plt.legend()
plt.savefig('loss_per_global_batch.png')
plt.show()
