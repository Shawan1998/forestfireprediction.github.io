from flask import Flask,render_template,request
import numpy as np
import pickle

with open('model.pkl','rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction = model.predict_proba(final)
    output='{0:{1}f}'.format(prediction[0][1],2)

    if output>str(0.5):
        return render_template('index.html',pred1='Forest is in Danger')
    else:
        return render_template('index.html', pred2='Forest is Safe')

app.run()
