import socket
import sys
import threading
import time
import csv
import os
import secrets
import pandas as pd
import numpy as np
import pickle
import math
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from PIL import Image
from flask import jsonify, render_template, url_for, flash, redirect, request
from flaskApp import app, db, bcrypt, mail
from flaskApp.forms import RegisterationForm,LoginForm, UpdateAccountForm, RequestResetForm, ResetPasswordForm
from flaskApp.model import User, Post, Dustbin
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message
from datetime import datetime



day_value={"Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6,"Sunday":7}

def data_from_server(x):
    x1, y1=x.split(",")
    return x1,y1

def is_new_hour(x):
    last_update = db.session.query(Dustbin).order_by(Dustbin.id.desc()).first()
    last_update= last_update.time_in_hour
    int(last_update)
    if int(last_update)<= x:
        return False
    else:
        return True

def day_number():
    today_now = datetime.today().strftime('%A')
    today_now = int(day_value.get(today_now))
    return today_now


def get_last_status():
    newest_row = db.session.query(Dustbin).order_by(Dustbin.id.desc()).first()
    newest_row = newest_row.status
    return int(newest_row)

def get_last_row(x):
    newest_row = db.session.query(Dustbin).order_by(Dustbin.id.desc()).first()
    if x == 'previous':
        newest_row = newest_row.previous_status
    elif x == 'amount_per_day':
        newest_row = newest_row.amount_per_day
    else:
        newest_row = newest_row.full
    return newest_row



def get_amount_per_day(x):
    hour_now = datetime.now().strftime('%H')
    try:
        first_amount = db.session.query(Dustbin).filter_by(day_of_week= day_number(), time_in_hour = int(hour_now)).order_by(Dustbin.id.desc()).first()
        first_amount= int(first_amount)
        if x>= first_amount:
            return x- first_amount
        else:
            return (100-first_amount)+ x
    except:
        return x


def day_year():
    d = datetime.datetime.now().strftime('%j')
    return int(d)
def is_full():
    full_day = db.session.query(Dustbin).filter_by(day_in_year= day_year(), full= 1).order_by(Dustbin.id.desc()).first()
    full_day = full_day.full
    if  full_day==0 or full_day==1:
        return full_day
    else:
        return 0



@app.route('/')
@app.route('/home' ,methods=['POST','GET'])
def home():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return render_template('home.html')


host='192.168.1.7'
port = 65431
@app.route('/data',methods=['GET'])
def test():
    try:
        s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.connect((host, port))
        my_input='Data'
        my_inp=my_input.encode('utf-8')
        s.sendall(my_inp)
        data=s.recv(1024).decode('utf-8')
        #status
        x_tem,y_tem= data_from_server(data)
        x = datetime.now().strftime('%H')
        #time_in_hour
        x = int(x)
        if is_new_hour(x) or int(x_tem) == 100:
            #day_of_week
            day_of_week = day_number()
            #holiday
            holi=is_holiday(day_of_week)
            #last_status
            previous_status = get_last_status()
            #amount_per_day
            amount = get_amount_per_day(x_tem)
            day_in_year = day_year()
            full = is_full()
            range_k= 0
            range_r = 1
            if x_tem>= 75:
                range_k=1
                range_r=4
            if x_tem>25 and x_tem<= 50:
                range_r =2
            elif x_tem >50 and x_tem< 75:
                range_r=3

                
            d=Dustbin( day_of_week = day_number, holiday = holi, time_in_hour = x, status =x_tem , previous_status = previous_status, amount_per_day = amount, full = full, range_knn = 0, range_rf =3, day_in_year=day_in_year)
            db.session.add(d)
            db.session.commit()
        my_input='Quit'
        my_inp=my_input.encode('utf-8')
        s.sendall(my_inp)
        return jsonify(x_tem)
    except:
        pass
    finally:
        s.close()
        
        
@app.route('/about')
def about():
    return render_template('about.html', title='About')
    
@app.route("/login", methods=['GET','POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form=LoginForm()
    if form.validate_on_submit():
         user= User.query.filter_by(email=form.email.data).first()
         if user and bcrypt.check_password_hash(user.password, form.password.data):
             login_user(user, remember=form.remember.data)
             next_page= request.args.get('next')
             return redirect(next_page) if next_page else redirect(url_for('home'))
         else:
            flash('Login Unsuccessful. Please check email and password','danger')
    return render_template('login.html',title='Login', form=form)





@app.route("/register", methods=['GET','POST'])
def register():
    
    form= RegisterationForm()
    if form.validate_on_submit():
        hash_password=bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user= User(username=form.username.data, email=form.email.data, password=hash_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created!', 'success')
        return redirect(url_for('home'))
    return render_template('register.html', title='Register', form=form)




@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('login'))

def save_picture(form_picture):
    random_hex=secrets.token_hex(8)
    _, f_ext= os.path.splitext(form_picture.filename)
    picture_fn= random_hex + f_ext
    picture_path=os.path.join(app.root_path,'static/pics',picture_fn)
    output_size= (125,125)
    i= Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)
    return picture_fn

@app.route("/account", methods=['GET','POST'])
@login_required
def account():
    form= UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file=save_picture(form.picture.data)
            current_user.image_file=picture_file
        current_user.username= form.username.data
        current_user.email= form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data= current_user.username
        form.email.data= current_user.email
    image_file= url_for('static',filename='pics/'+ current_user.image_file)
    return render_template('account.html',title='Account', image_file=image_file, form=form)


@app.route("/predict",methods=['GET','POST'])
def pre():
    return render_template('predict.html')

def is_holiday(day_of_week):
    if day_of_week==6 or day_of_week==7:
        return 1
    else:
        return 0

@app.route("/result",methods=['GET','POST'])
def result():
    """
    with open('flaskApp\data\wasteManagment.csv','w') as output_file:
        output_csv=csv.writer(output_file)
        output_csv.writerow(['day_of_week','time_in_hour','status','range_knn','range_rf'   ])
        for row in db.session.query(Dustbin).all():
            output_csv.writerow([row.day_of_week, row.status])

    df = pd.read_csv("flaskApp\data\wasteManagment.csv")
    df_data=df[['day_of_week','time_in_hour','status','range_knn','range_rf']]
    df_x= df_data['day_of_week']
    df_y=df_data.status
    corpus=df_x
    cv= CountVectorizer()
    X=cv.fit_transform(corpus)
    from sklearn.model_selection import train_test_split
    X_train,X_test, y_train,y_test= train_test_split(X,df_y,test_size=0.70, random_state=42)
    from sklearn.naive_bayes import MultinomialNB
    clf=MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)
    if request.method=='POST':
       comment=request.form['comment']
       data=[comment]
       vect= cv.transform(data).toarray()
       my_prediction=clf.predict(vect)
       my_prediction=int(my_prediction)
    return render_template('result.html',prediction=my_prediction)
    """

    #knn
    df = pd.read_csv("flaskApp\data\wasteManagment.csv", header=None, skiprows=1)
    X = df.iloc[:,0:7]
    y = df.iloc[:,7]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size= 0.3)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    n=math.sqrt(len(y_test))
    m=int(n)
    if m%2 == 0:
        m=m-1
    classifier = KNeighborsClassifier(n_neighbors= m, p=2,metric='euclidean')
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    holiday=0
    if request.method=='POST':
        comment=request.form['comment']
        comment =int(day_value.get(comment))
        holiday=is_holiday(comment)
        dt = datetime.now().strftime('%H')
        previous_status='previous_status'
        data=[comment,holiday,dt,get_last_status(),get_last_row('previous'),get_last_row('amount_per_day'),get_last_row('full')]
        ww=np.array(data).reshape(1,-1)
        result_predict=classifier.predict(ww)

        #random forest
        dataset = pd.read_csv("flaskApp\data\wasteManagment.csv", header=None, skiprows=1)
        target_names = ['not recommended','slightly recommended','recommended','highly recommended']
        feature_names = ['day_of_week','holiday','time_in_hour','status','previous_status','amount_per_day','full']
        X = dataset.iloc[:, :-3].values
        y = dataset.iloc[:, 8].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
        clf = RandomForestClassifier(n_estimators = 4)   
        clf.fit(X_train, y_train) 
        y_pred = clf.predict(X_test) 
        rf_resutl=clf.predict([[comment,holiday,dt,get_last_status(),get_last_row('previous'),get_last_row('amount_per_day'),get_last_row('full')]]) 
        rf_resutl[0]=rf_resutl[0]*25
    return render_template('result.html',prediction=result_predict[0], rf= rf_resutl[0])


def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message('Password Reset Request', sender= 'noreply@demo.com',recipients=[user.email])

    msg.body = f'''To reset you password, visit the following link:
{url_for('reset_token', token=token, _external= True)}

If you did not make this request then simply ignore this email and no change will be made.
'''
    mail.send(msg)


@app.route("/reset_password", methods=['GET','POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form= RequestResetForm()
    if form.validate_on_submit():
        user= User.query.filter_by(email= form.email.data).first()
        send_reset_email(user)
        flash('AN email has been sent to reset your passeord','info')
        return redirect(url_for('login'))
    return render_template('reset_request.html', title= 'Reset Password', form=form)

@app.route("/reset_password/<token>", methods=['GET','POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token','warning')
        return redirect(url_for('reset_request'))
    form= ResetPasswordForm()
    if form.validate_on_submit():
        hash_password=bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hash_password
        db.session.commit()
        flash('Your password has been updated! You are now able to log in', 'success')
        return redirect(url_for('login')) 
    return render_template('reset_token.html', title= 'Reset Password', form = form)