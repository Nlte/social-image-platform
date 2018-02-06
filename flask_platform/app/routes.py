import os
import pdb
from flask import render_template, redirect, url_for, request, flash
from flask_login import current_user, login_user, logout_user, login_required
from werkzeug.utils import secure_filename
from .utils import *

from app import app, db
from app.models import User, Post
from app.forms import RegistrationForm, LoginForm, PostForm

MEDIADIR = app.config['MEDIADIR']

@app.route('/ping')
def ping():
   return 'Flask server running'

@app.route('/')
@app.route('/index')
def index():
    user = {'username': 'User'}
    posts = [{'title':'A'}, {'title':'B'}]
    return render_template('index.html', user=user, posts=posts)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user is None or not user.check_password(form.password.data):
            print('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=True)
        return redirect(url_for('index'))
    return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('User created')
        #login_user(user)
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route('/timeline')
@login_required
def userposts():
    return render_template('base.html', posts=post)

@app.route('/newpost', methods=['GET', 'POST'])
@login_required
def newpost():
    form = PostForm()
    if form.validate_on_submit():
        pdb.set_trace()
        f = form.image.data
        filename = secure_filename(f.filename)
        filename = str(gen_uuid()) + '.' + filename.split('.')[-1]
        f.save(os.path.join(
            MEDIADIR, filename
        ))
        #TODO call pred server
        relpath = os.path.relpath(filename, start=MEDIADIR)
        p = Post(title=form.title.data, image=relpath)
        db.session.add(p)
        db.session.commit()
        return redirect(url_for('index'))
    return render_template('newpost.html', form=form)

@app.route('/posts/<int:postId>')
@login_required
def show_post(postId):
   return 'Post Number %d' % postId
