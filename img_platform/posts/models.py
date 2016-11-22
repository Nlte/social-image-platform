from __future__ import unicode_literals

from uuid import uuid4
from django.db import models
import json
from django.dispatch import receiver
from authentication.models import Account
from pred_server import PredictionServer

model_server = PredictionServer()

def scramble_image_filename(instance, filename):
    extension = filename.split('.')[-1]
    return '{}.{}'.format(uuid4(), extension)


class Post(models.Model):
    author = models.ForeignKey(Account)
    title = models.TextField(default="")
    annotation = models.TextField(blank=True)
    image = models.ImageField(upload_to=scramble_image_filename, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)



    def __unicode__(self):
        return '{} - {}'.format(self.author, self.title)


@receiver(models.signals.post_save, sender=Post)
def make_prediction(sender, instance, created, **kwargs):
    if created:
        prediction = model_server.inference(instance.image.file.name)
        instance.annotation = ' '.join(prediction)
        instance.save()
