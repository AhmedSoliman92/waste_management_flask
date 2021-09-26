from datetime import datetime
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from flaskApp import db,login_manager, app
from flask_login import UserMixin

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id=db.Column(db.Integer, primary_key=True)
    username= db.Column(db.String(), unique=True, nullable=False)
    email= db.Column(db.String(120),unique=True, nullable=False)
    image_file=db.Column(db.String(20),nullable=False,default='default.jpg')
    password=db.Column(db.String(60), nullable=False)
    posts=db.relationship('Post',backref='author', lazy=True)

    
    def get_reset_token(self, expires_sec=1800):
        s=Serializer(app.config['SECRET_KEY'],expires_sec)
        return s.dumps({'user_id':self.id}).decode('utf-8')
    @staticmethod
    def verify_reset_token(token):
        s = Serializer(app.config['SECRET_KEY'])
        try:
            user_id= s.loads(token)['user_id']
        except:
            return None
        return User.query.get(user_id)
    def __repr__(self):
        return f"User('{self.username}','{self.email}','{self.image_file}')"

class Post(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    title=db.Column(db.String(100), nullable=False)
    date_posted= db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    content=db.Column(db.Text,nullable=False)
    user_id= db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"Post('{self.title}','{self.date_posted}')"

class Dustbin(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    day_of_week=db.Column(db.Integer, nullable=False)
    holiday=db.Column(db.Integer,nullable=False)
    time_in_hour = db.Column(db.Integer, default = datetime.now().strftime('%H'), nullable = False)
    status=db.Column(db.Integer,nullable=False)
    previous_status = db.Column(db.Integer,nullable=False)
    amount_per_day = db.Column(db.Integer,nullable=False)
    full = db.Column(db.Integer,nullable=False)
    range_knn = db.Column(db.Integer,nullable=False)
    range_rf = db.Column(db.Integer,nullable=False)
    day_in_year= db.Column(db.Integer , nullable= False)
    
    
    def __repr__(self):
        return f"Dustbin('{self.id}','{self.day_of_week}', '{self.holiday}', '{self.time_in_hour}','{self.status}','{self.previous_status}' ,'{self.amount_per_day}','{self.full}', '{self.range_knn}' ,'{self.range_rf}', '{self.day_in_year}')"