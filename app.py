from flask import Flask, render_template, request
from PIL import Image
import numpy as np

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        file = request.files['image']
        if file:
            img = Image.open(file).convert('L')
            img = img.resize((8, 8))
            data = np.array(img)


    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)