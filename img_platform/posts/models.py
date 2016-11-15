from __future__ import unicode_literals

from uuid import uuid4
from django.db import models
from django.dispatch import receiver
from authentication.models import Account
#from posts.tasks import PredictionServer

#pred_server = PredictionServer()

def scramble_image_filename(instance, filename):
    extension = filename.split('.')[-1]
    return '{}.{}'.format(uuid4(), extension)


class Post(models.Model):
    author = models.ForeignKey(Account)
    title = models.TextField(blank=True)
    annotation = models.TextField(blank=True)
    image = models.ImageField(upload_to=scramble_image_filename, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    #def save(self, *args, **kwargs):
    #    self.auto_caption = tasks.inference(self.image.file.name)
    #    super(Post, self).save(*args, **kwargs)

    def __unicode__(self):
        return '{} - {}'.format(self.author, self.title)


@receiver(models.signals.post_save, sender=Post)
def make_prediction(sender, instance, created, **kwargs):
    # without this check the save() below causes infinite post_save signals
    if created:
        #instance.some_field = complex_calculation()
        #instance.annotation = pred_server.inference(instance.image.file.name)
        #instance.auto_caption = 'auto generated caption'
        print(instance.image.file.name)
        instance.save()
