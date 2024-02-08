from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('premodel.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = int(request.form['a'])
    data2 = int(request.form['b'])
    data3 = int(request.form['c'])
    data4 = int(request.form['d'])
    data5 = int(request.form['e'])
    data6 = int(request.form['f'])
    data7 = int(request.form['g'])
    data8 = int(request.form['h'])
    data9 = int(request.form['i'])
    data10 = int(request.form['j'])
    data11 = int(request.form['k'])
    data12 = int(request.form['l'])
    data13  = int(request.form['m'])
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)
     

if __name__ == "__main__":
    app.run(debug=True)















