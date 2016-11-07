from django.test import TestCase
from django.contrib import auth

class AuthTestCase(TestCase):
    def setUp(self):
        self.u = User.objects.create_user('test@dom.com', 'test@dom.com', 'pass')
        self.u.is_staff = True
        self.u.is_superuser = True
        self.u.is_active = True
        self.u.save()

    def testLogin(self):
        self.client.login(username='test@dom.com', password='pass')
