import os
import pdb
from flask import render_template, redirect, url_for, request, flash, send_from_directory
from flask_login import current_user, login_user, logout_user, login_required
from werkzeug.utils import secure_filename
from jinja2 import Template
from .utils import *

from app import app, db, jinja_env
from app.models import User, Post
from app.forms import RegistrationForm, LoginForm, PostForm

ABS_MEDIA_DIR = os.path.join(app.config['SRCDIR'], app.config['MEDIADIR'])
if not os.path.isdir(ABS_MEDIA_DIR):
    os.mkdir(ABS_MEDIA_DIR)

@app.route('/ping')
def ping():
   return 'Flask server running'

@app.route('/')
@app.route('/index')
def index():
    page = request.args.get('page', 1, type=int)
    posts = Post.query.order_by(Post.dateCreated.desc())
    posts = posts.paginate(page, app.config['POST_PER_PAGE'], False)
    next_page = None
    prev_page = None
    if posts.has_next:
        next_page = url_for('index', page=posts.next_num)
    if posts.has_prev:
        prev_page = url_for('index', page=posts.prev_num)
    return render_template('grid.html', posts=posts.items,
        prev_page=prev_page, next_page=next_page)


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


@app.route('/timeline/<username>')
@login_required
def timeline(username):
    user = User.query.filter_by(username=username).first_or_404()
    page = request.args.get('page', 1, type=int)
    posts = user.posts.order_by(Post.dateCreated.desc())
    posts = posts.paginate(page, app.config['POST_PER_PAGE'], False)
    next_page = None
    prev_page = None
    if posts.has_next:
        next_page = url_for('index', page=posts.next_num)
    if posts.has_prev:
        prev_page = url_for('timeline', page=posts.prev_num)
    return render_template('grid.html', posts=posts.items,
        prev_page=prev_page, next_page=next_page)


@app.route('/newpost', methods=['GET', 'POST'])
@login_required
def newpost():
    form = PostForm()
    if form.validate_on_submit():
        f = form.image.data
        filename = secure_filename(f.filename)
        filename = str(gen_uuid()) + '.' + filename.split('.')[-1]
        abspath = os.path.join(ABS_MEDIA_DIR, filename)
        f.save(abspath)
        #TODO call pred server
        p = Post(title=form.title.data, image=filename, user_id=current_user.id)
        db.session.add(p)
        db.session.commit()
        return redirect(url_for('index'))
    return render_template('newpost.html', form=form)


@app.route('/cdn/<path:filename>')
def cdn_media(filename):
    return send_from_directory(app.config['MEDIADIR'], filename)


@app.route('/posts/<int:postId>')
@login_required
def show_post(postId):
   return 'Post Number %d' % postId
