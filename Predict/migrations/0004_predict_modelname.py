# Generated by Django 4.0 on 2021-12-29 12:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Predict', '0003_remove_predict_result'),
    ]

    operations = [
        migrations.AddField(
            model_name='predict',
            name='modelName',
            field=models.TextField(default=''),
        ),
    ]