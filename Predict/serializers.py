from rest_framework import serializers
from .models import Predict


class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Predict
        fields = ('image','modelName')
