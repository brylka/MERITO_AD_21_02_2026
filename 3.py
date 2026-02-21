from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

print(digits.keys())

print(digits.images[0])
plt.imshow(digits.images[0], cmap='gray')
plt.show()

