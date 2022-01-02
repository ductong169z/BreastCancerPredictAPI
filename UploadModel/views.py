import shutil

from django.shortcuts import render
import cv2
import tensorflow as tf
import numpy as np
import os
from skimage import transform

import base64
from django.http import StreamingHttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import UploadModelSerializer



# Create your views here.
class UploadModelView(APIView):
    def post(self, request):

        serializer = UploadModelSerializer(data=request.data)

        if serializer.is_valid():
            file = serializer.save()

            return Response({"status": "success","file_name":os.path.basename(file.file.name)},status=status.HTTP_200_OK)
        else:
            return Response({"status": "error", "data": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)
