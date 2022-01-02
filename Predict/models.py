from django.db import models


# Create your models here.
class Predict(models.Model):
    image = models.ImageField(upload_to='uploads/', blank=True)
    modelName=models.TextField(default='')

    def delete(self, using=None, keep_parents=False):
        self.image.storage.delete(self.image.name)
        super().delete()

