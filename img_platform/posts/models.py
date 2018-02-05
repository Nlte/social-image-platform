from __future__ import unicode_literals

from uuid import uuid4
from django.db import models
import json
import pdb
import requests
from django.dispatch import receiver
from authentication.models import Account

URL_PREDSERVER = 'http://127.0.0.1:8080/predict'

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
        payload = {'files': open(instance.image.file.name, 'rb')}
        resp = requests.post(URL_PREDSERVER, files=payload)
        resp.raise_for_status()
        if resp.content:
            labels = resp.json()['labels']
            labels = ' '.join(['#' + l for l in labels])
            instance.annotation = labels
        instance.save()
