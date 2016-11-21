from __future__ import unicode_literals

from uuid import uuid4
from django.db import models
import json
from django.dispatch import receiver
from authentication.models import Account
from prediction import PredictionServer

#cache_key = 'model_cache'
#model_server = cache.get(cache_key)
#if model_server is None:
#    model_server = PredictionServer()
#    #save in django memory cache
#    cache.set(cache_key, model_server, None)
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

    #def save(self, *args, **kwargs):
    #    self.auto_caption = tasks.inference(self.image.file.name)
    #    super(Post, self).save(*args, **kwargs)
    def setannotation(self, x):
        self.annotation = json.dumps(x)

    def getannotation(self):
        return json.loads(self.annotation)

    def __unicode__(self):
        return '{} - {}'.format(self.author, self.title)


@receiver(models.signals.post_save, sender=Post)
def make_prediction(sender, instance, created, **kwargs):
    # without this check the save() below causes infinite post_save signals
    if created:
        #instance.some_field = complex_calculation()
        jsonlist = model_server.inference(instance.image.file.name)
        instance.setannotation(' '.join(jsonlist))
        #instance.auto_caption = 'auto generated caption'
        print(instance.image.file.name)
        instance.save()
