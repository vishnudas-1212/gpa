from flask import Flask, render_template, request,url_for
import pickle
import numpy as np

model = pickle.load(open('score.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def main():
    return render_template('first.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    arr = np.array([[data1]])
    pred = model.predict(arr)
    return render_template('result.html',prediction_text="Your gpa is {}".format(pred))


if __name__ == "__main__":
    app.run(debug=True)
