from rest_framework import serializers
from .models import UploadModel


class UploadModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadModel
        fields = ('file',)
