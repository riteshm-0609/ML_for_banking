from flask import Flask, url_for, render_template, request, redirect, session
from flask_pymongo import PyMongo
from flaskext.mysql import MySQL
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
import smtplib
app = Flask(__name__)
UPLOAD_FOLDER = '/home/ritesh/Desktop/ocr_banking'
mysql = MySQL()
app.secret_key = 'rootpasswordgiven'
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'rootpasswordgiven'
app.config['MYSQL_DATABASE_DB'] = 'test'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)
app.config["MONGO_URI"] = "mongodb://localhost:27017/test"
mongo = PyMongo(app)
conn = mysql.connect()
cursor =conn.cursor()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/home",  methods = [ 'GET' , 'POST' ])
def home():
	if 'username' in session:
		return render_template('start.html') 

@app.route('/', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    error = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor.execute('SELECT * FROM employee WHERE username = %s AND password = %s', (username, password))
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account[0]
            session['username'] = account[2]
            # Redirect to home page
            return render_template('start.html')
        else:
            # Account doesnt exist or username/password incorrect
            error = 'Incorrect username/password!'
    # Show the login form with message (if any)
    return render_template('login.html', error=error)

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
	return render_template('testing.html', acc_no=acc_no,cust_no=cust_no,name = name,\
	father = father,mother = mother,email = email, income = income, co_income = co_income,\
	amount = amount, term = term,dependents = dependents)
	
@app.route("/upload", methods = ['GET', 'POST']) 
def upload():
	if request.method == 'POST':
		name = request.form["name"]
		customer_no = request.form["customer_no"]
		account_no = request.form["acc_no"]
		dob = request.form["dob"]
		father = request.form["father"]
		mother = request.form["mother"]
		email = request.form["email"]
		income = request.form["income"]
		co_income = request.form["co_income"]
		amount = request.form["amount"]
		term = request.form["term"]
		dependents = request.form["dependents"]
		credit = request.form["credit"]
		df = [income,co_income,amount,term]
	dataset = pd.read_csv('/home/ritesh/Desktop/ocr_banking/train.csv')
	X = dataset.iloc[:, [6,7,8,9]].values
	y = dataset.iloc[:, 12].values
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	X = sc.fit_transform(X)
	from sklearn.impute import SimpleImputer
	missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'constant', verbose = 0)
	missingvalues = missingvalues.fit(X[:, [0,1,2]])
	X[:, [0,1,2]]=missingvalues.transform(X[:, [0,1,2]])
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	labelencoder_X = LabelEncoder()
	labelencoder_y = LabelEncoder()
	y = labelencoder_y.fit_transform(y)
	from sklearn.linear_model import LogisticRegression
	classifier = LogisticRegression(random_state = 0)
	classifier.fit(X, y)
	#data = mongo.db.customer.findOne()
	#df = pd.Dataframe(list(data))
	y_pred = classifier.predict(df)
	mongo.db.customer.insert({"name":name,"customer_no":customer_no,\
		"account_no":account_no,"dob":dob,"father":father,"mother":mother,"email":email,\
			"income":income,"co_income":co_income,"amount":amount,"term":term,"dependents":dependents})
	return render_template('uploadsuccessful.html')	
#manager

@app.route("/manager", methods = ['GET', 'POST']) 
def manager():
	customers = mongo.db.customer.find()
	return render_template('manage.html', customers = customers)

@app.route("/approve", methods = ['GET', 'POST']) 
def approve():
	account = mongo.db.customer.findOne()
	cust_no = account["cust_no"]
	email = account["email"]
	s = smtplib.SMTP('riteshm.0609@gmail.com', 587)
	s.starttls() 
	s.login("riteshm.0609@gmail.com", "Ritesh.M@06") 
	message = "Congratulations! your loan has been approved"
	s.sendmail("riteshm.0609@gmail.com", email, message) 
	s.quit() 
	mongo.db.customer.deleteOne()
	mongo.db.customer_finished.insert(account)
	cursor.execute('INSERT INTO loan_status VALUES(%d,%s)',(cust_no,"approved"))
	customers = mongo.db.customer.find()
	return render_template('manage.html', customers = customers)

@app.route("/reject", methods = ['GET', 'POST']) 
def reject():
	account = mongo.db.customer.findOne()
	cust_no = account["cust_no"]
	email = account["email"]
	s = smtplib.SMTP('riteshm.0609@gmail.com', 587)
	s.starttls() 
	s.login("sender_email_id", "sender_email_id_password") 
	message = "We are sorry to inform you that your Loan has been Rejected."
	s.sendmail("riteshm.0609@gmail.com", email, message) 
	s.quit() 
	mongo.db.customer.deleteOne()
	mongo.db.customer_finished.insert(account)
	cursor.execute('INSERT INTO loan_status VALUES(%d,%s)',(cust_no,"rejected"))
	customers = mongo.db.customer.find()
	return render_template('manage.html', customers = customers)


if __name__ == '__main__':
	app.run(debug=True)	
	
		
