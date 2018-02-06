import os

srcdir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(srcdir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    WTF_CSRF_CHECK_DEFAULT = False
    SECRET_KEY = 'abcd'
    MEDIADIR = os.path.join(srcdir, 'media/')
