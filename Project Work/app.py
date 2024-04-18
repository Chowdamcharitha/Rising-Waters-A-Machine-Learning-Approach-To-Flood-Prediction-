from flask import Flask,render_template,url_for,request,jsonify
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import datetime
import pickle

app = Flask(__name__, template_folder="templates")
import joblib
model = joblib.load('models/flood2.save')
sc=joblib.load('models/transform2.save')

@app.route("/",methods=['GET'])
def home():
	return render_template("index.html")
@app.route("/predict",methods=['GET', 'POST'])
@cross_origin()
def predict():
	if request.method == "POST":
		
		cc = float(request.form['1'])
		# 
		a = float(request.form['2'])
		# 
		ja = float(request.form['3'])
		# 
		ma = float(request.form['4'])
		# 
		ju = float(request.form['5'])
		input_lst = [cc,a,ja,ma,ju ]
		pred = model.predict(sc.transform([input_lst]))
		output = pred
		if output == 1:
			return render_template("flood.html",msg="Possibility of Severe Flood")
		else:
			return render_template("non.html",msg="No Possibility of Flood")
	return render_template("predict.html")

if __name__=='__main__':
	app.run(debug=True)