from __future__ import unicode_literals

from uuid import uuid4
from django.db import models
from authentication.models import Account
from inference import inference

def scramble_image_filename(instance, filename):
    extension = filename.split('.')[-1]
    return '{}.{}'.format(uuid4(), extension)

class Post(models.Model):
    author = models.ForeignKey(Account)
    user_caption = models.TextField(blank=True)
    auto_caption = models.TextField(blank=True)
    image = models.ImageField(upload_to=scramble_image_filename, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    #auto_caption = inference(url(image))

    #def save(self, *args, **kwargs):
    #    self.auto_caption = inference(self.image.file.name)
    #    super(Post, self).save(*args, **kwargs)

    def __unicode__(self):
        return '{} - {}'.format(self.author, self.user_caption)
