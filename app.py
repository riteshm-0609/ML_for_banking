from flask import Flask, url_for, render_template, request
from flask_pymongo import PyMongo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from cv2 import imread
#from cv2 import fastNlMeansDenoisingColored
import pytesseract
from PIL import Image
#from sklearn.metrics import accuracy_score
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)
UPLOAD_FOLDER = '/home/ritesh/Desktop/ocr_banking'
#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test3.db'
#app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config["MONGO_URI"] = "mongodb://localhost:27017/test"
mongo = PyMongo(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route("/",  methods = [ 'GET' , 'POST' ])
def home():
	return render_template('start.html') 


@app.route("/afr", methods = ['GET', 'POST']) 
def afr():

	if (request.method == 'POST'):
		file =request.files['file']
		filename = secure_filename(file.filename) # save file 
		filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		file.save(filepath)
		img = cv2.imread(filepath) 

	#ACCOUNT NUMBER
	acc_no = img[350:400,120:600]
	plt.imshow(acc_no)
	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, acc_no)
	acc_no_txt = pytesseract.image_to_string(Image.open(filename),lang='eng')
	os.remove(filename)
	#CUSTOMER ID
	cust_id = img[350:404,854:1170]
	plt.imshow(cust_id)
	filename1 = "{}.jpg".format(os.getpid())
	cv2.imwrite(filename1, cust_id)
	cust_id_txt = pytesseract.image_to_string(Image.open(filename1),lang='eng')
	os.remove(filename1)
	#NAME
	name = img[444:490,290:600]
	plt.imshow(name)
	filename = "{}.jpg".format(os.getpid())
	cv2.imwrite(filename, name)
	name_txt = pytesseract.image_to_string(Image.open(filename),lang='eng')
	os.remove(filename)

	#DATE OF BIRTH
	dob = img[622:670,120:500]
	plt.imshow(dob)
	filename1 = "{}.jpg".format(os.getpid())
	cv2.imwrite(filename1, dob)
	dob_txt = pytesseract.image_to_string(Image.open(filename1),lang='eng')
	os.remove(filename1)

	#MOTHER'S NAME
	mother_name = img[900:945,290:600]
	plt.imshow(mother_name)
	filename = "{}.jpg".format(os.getpid())
	cv2.imwrite(filename, mother_name)
	mom_txt = pytesseract.image_to_string(Image.open(filename),lang='eng')
	os.remove(filename)

	#FATHER'S NAME
	father_name = img[825:880,290:600]
	plt.imshow(father_name)
	filename = "{}.jpg".format(os.getpid())
	cv2.imwrite(filename, father_name)
	dad_txt = pytesseract.image_to_string(Image.open(filename),lang='eng')
	os.remove(filename)
	return render_template('testing.html', acc_no=acc_no_txt,cust_no=cust_id_txt,name_txt = name_txt,dob_txt = dob_txt,mom_txt = mom_txt,dad_txt = dad_txt )
	
@app.route("/upload", methods = ['GET', 'POST']) 
def upload():
	if request.method == 'POST':
		name = request.form["name_txt"]
		customer_no = request.form["customer_no"]
		account_no = request.form["acc_no"]
	dataset = pd.read_csv('/home/ritesh/Desktop/ocr_banking/train.csv')
	X = dataset.iloc[:, [3,5,6,7,8,9,10]].values
	y = dataset.iloc[:, 12].values
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	X = sc.fit_transform(X)
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	labelencoder_X = LabelEncoder()
	X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
	onehotencoder = OneHotEncoder(categorical_features = [5])
	X = onehotencoder.fit_transform(X).toarray()
	labelencoder_y = LabelEncoder()
	y = labelencoder_y.fit_transform(y)
	from sklearn.linear_model import LogisticRegression
	classifier = LogisticRegression(random_state = 0)
	classifier.fit(X, y)
	data = mongo.db.customer.findOne()
	df = pd.Dataframe(list(data))
	labelencoder_df = LabelEncoder()
	df[:, 6] = labelencoder_df.fit_transform(X[:, 6])
	onehotencoder = OneHotEncoder(categorical_features = [6])
	df = onehotencoder.fit_transform(df).toarray()
	y_pred = classifier.predict(df)
	mongo.db.customer.insert({"name":name,"customer_no":customer_no,"account_no":account_no})
	return render_template('uploadsuccessful.html')	
#manager

@app.route("/manager", methods = ['GET', 'POST']) 
def manager():
	customers = mongo.db.customer.find()
	return render_template('mnagae.html')







if __name__ == '__main__':
	app.run(debug=True)	
	
		
