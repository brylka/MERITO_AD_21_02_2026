from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

print(digits.keys())

print(digits.images[54])
