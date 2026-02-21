from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

print(digits.keys())

print(digits.images[0])
plt.imshow(digits.images[0], cmap='gray')
plt.show()

fig, axes = plt.subplots(2, 5, figsize=(10,5))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i+20], cmap='gray')
    ax.set_title(f"Cyfra: {digits.target[i+20]}")
    ax.axis('off')
plt.show()
