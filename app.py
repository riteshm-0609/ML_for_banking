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
from flask_mail import Message, Mail
#import socket
#sock = socket.socket()
#sock.connect(("smtp.gmail.com", 587))
app = Flask(__name__)
UPLOAD_FOLDER = '/home/ritesh/Desktop/ocr_banking'
#SQLPART
mysql = MySQL()
app.secret_key = 'rootpasswordgiven'
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'rootpasswordgiven'
app.config['MYSQL_DATABASE_DB'] = 'test'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)
#MONGOPART
app.config["MONGO_URI"] = "mongodb://localhost:27017/test"
mongo = PyMongo(app)
conn = mysql.connect()
cursor =conn.cursor()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#EMAILPART
mail = Mail()
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 465
app.config["MAIL_USE_SSL"] = True
app.config["MAIL_USERNAME"] = 'riteshm.0609@gmail.com'
app.config["MAIL_PASSWORD"] = 'Ritesh.M@06'
mail.init_app(app)


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
	acc_no = img[403:490,120:700]
	plt.imshow(acc_no)
	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, acc_no)
	acc_no = pytesseract.image_to_string(Image.open(filename),lang='eng')
	os.remove(filename)
	
	#CUSTOMER ID
	cust_no = img[412:480,1000:1500]
	filename1 = "{}.jpg".format(os.getpid())
	cv2.imwrite(filename1, cust_no)
	cust_no = pytesseract.image_to_string(Image.open(filename1),lang='eng')
	os.remove(filename1)
	
	#NAME
	name = img[505:585,120:700]
	filename = "{}.jpg".format(os.getpid())
	cv2.imwrite(filename, name)
	name = pytesseract.image_to_string(Image.open(filename),lang='eng')
	os.remove(filename)

	#DATE OF BIRTH
	dob = img[665:730,120:555]
	filename1 = "{}.jpg".format(os.getpid())
	cv2.imwrite(filename1, dob)
	dob = pytesseract.image_to_string(Image.open(filename1),lang='eng')
	os.remove(filename1)
	
	#MOTHER'S NAME
	mother= img[940:1000,120:700]
	filename = "{}.jpg".format(os.getpid())
	cv2.imwrite(filename, mother)
	mother = pytesseract.image_to_string(Image.open(filename),lang='eng')
	
	#FATHER'S NAME
	father = img[850:920,120:700]
	filename = "{}.jpg".format(os.getpid())
	cv2.imwrite(filename, father)
	father=pytesseract.image_to_string(Image.open(filename),lang='eng')
	os.remove(filename)
		
	#EMAIL
	email = img[1320:1400,330:1000]
	filename = "{}.jpg".format(os.getpid())
	cv2.imwrite(filename, email)
	email = pytesseract.image_to_string(Image.open(filename),lang='eng')
	os.remove(filename)

	
	#INCOME
	income = img[1520:1600,120:700]
	filename = "{}.jpg".format(os.getpid())
	cv2.imwrite(filename, income)
	income = pytesseract.image_to_string(Image.open(filename),lang='eng')
	os.remove(filename)
	
	
	#CO_INCOME
	co_income = img[1670:1750,120:500]
	filename = "{}.jpg".format(os.getpid())
	cv2.imwrite(filename, co_income)
	co_income = pytesseract.image_to_string(Image.open(filename),lang='eng')
	os.remove(filename)

	#amount
	amount= img[1820:1900,120:550]
	filename = "{}.jpg".format(os.getpid())
	cv2.imwrite(filename, amount)
	amount = pytesseract.image_to_string(Image.open(filename),lang='eng')
	os.remove(filename)

	#term
	term = img[1970:2050,120:550]
	filename = "{}.jpg".format(os.getpid())
	cv2.imwrite(filename,term)
	term = pytesseract.image_to_string(Image.open(filename),lang='eng')
	os.remove(filename)

	return render_template('testing.html', acc_no=acc_no,cust_no=cust_no,name = name,dob = dob,\
	father = father,mother = mother,email = email, income = income, co_income = co_income,\
	amount = amount, term = term)
	
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
		#credit = request.form["credit"]
		df = [income,co_income,amount,term]
		df = pd.DataFrame(df).T
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
	if y_pred == 0:
		result = "reject"
	else:
		result = "approve"	
	mongo.db.customer.insert({"name":name,"customer_no":customer_no,\
	"account_no":account_no,"dob":dob,"father":father,"mother":mother,"email":email,\
	"income":income,"co_income":co_income,"amount":amount,"term":term,"dependents":dependents,"result":result })
	return render_template('uploadsuccessful.html')	
#manager

@app.route("/manager", methods = ['GET', 'POST'])
def manlogin():
	render_template('manlogin.html') 
def manager():
	error = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor.execute('SELECT * FROM managers WHERE username = %s AND password = %s', (username, password))
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account[0]
            session['username'] = account[2]
            # Redirect to home page
			customers = mongo.db.customer.find()
			return render_template('manage.html', customers = customers)
        else:
            # Account doesnt exist or username/password incorrect
            error = 'Incorrect username/password!'
    # Show the login form with message (if any)
    return render_template('manlogin.html', error=error)

	

@app.route("/approve", methods = ['GET', 'POST']) 
def approve():
	account = mongo.db.customer.find_one()
	cust_no = account['customer_no']
	email = account['email']
	msg = Message("Loan_status", sender="riteshm.0609@gmail.com",recipients = [email])
	msg.body = """
    Congratulations! Your loan has been approved
    """
	mail.send(msg)
	mongo.db.customer.delete_one({"customer_no" : cust_no })
	mongo.db.customer_finished.insert(account)
	cursor.execute('''INSERT INTO loan_status VALUES(%s,%s)''',(cust_no,"approved"))
	conn.commit()
	customers = mongo.db.customer.find()
	return render_template('manage.html', customers = customers)

@app.route("/reject", methods = ['GET', 'POST']) 
def reject():
	account = mongo.db.customer.find_one()
	cust_no = account['customer_no']
	email = account['email']
	msg = Message("Loan_status", sender="riteshm.0609@gmail.com",recipients = [email])
	msg.body = """
    Your Loan Request Has been denied. We are sorry.
    """
	mail.send(msg)
	mongo.db.customer.delete_one({"customer_no" : cust_no })
	mongo.db.customer_finished.insert(account)
	cursor.execute('INSERT INTO loan_status VALUES(%s,%s)',(cust_no,"rejected"))
	customers = mongo.db.customer.find()
	return render_template('manage.html', customers = customers)


if __name__ == '__main__':
	app.run(debug=True)	
	
		
