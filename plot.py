import pickle
import matplotlib.pyplot as plt
import sys

# read loss file path and accuracy file path from input
path = sys.argv[1]

# Read the loss data from the pickle file
with open(path + "losses", "rb") as fp:
    losses = pickle.load(fp)

# Plot the loss data and save the plot
plt.plot(losses)
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(path + "/loss.png")

# Read the accuracy data from the pickle file
with open(path + "results", "rb") as fp:
    accuracy = pickle.load(fp)

# Plot the accuracy data and save the plot
# create a new figure
plt.figure()
plt.plot(accuracy)
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig(path + "/accuracy.png")
