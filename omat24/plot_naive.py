import matplotlib.pyplot as plt

k = [0, 1, 2, 3, 4, 5]
train_loss = [
    302.6110156259903,
    289.23186443161467,
    274.8778200872003,
    242.11153259130535,
    194.26094406300595,
    141.45294407918215,
]  # rattled-300-subsampled
validation_loss = [
    324.5168505207117,
    313.75200336302044,
    307.7340834445781,
    320.42727684195785,
    333.9285407429715,
    344.5931959930532,
]  # rattled-1000

plt.plot(k, train_loss, marker="o", color="blue", label="Train Loss")

plt.plot(k, validation_loss, marker="o", color="red", label="Validation Loss")

# Add labels, legend, and grid
plt.xlabel("k")
plt.ylabel("Loss")
plt.title("Train Loss and Validation Loss versus k")
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
