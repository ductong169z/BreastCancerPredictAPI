# Generated by Django 4.0 on 2021-12-22 14:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Predict', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='predict',
            name='img',
        ),
        migrations.AddField(
            model_name='predict',
            name='image',
            field=models.ImageField(blank=True, upload_to='uploads/'),
        ),
    ]
