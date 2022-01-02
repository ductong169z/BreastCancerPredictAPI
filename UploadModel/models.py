from django.db import models


# Create your models here.
class UploadModel(models.Model):
    file = models.FileField(upload_to='PredictModel/', blank=True)
