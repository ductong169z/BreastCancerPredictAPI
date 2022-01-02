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
from .serializers import ImageSerializer



# from mafia_api.settings import MEDIA_ROOT, BASE_DIR, DETECT_ROOT

# Create your views here.
class ImageView(APIView):
    def post(self, request):

        serializer = ImageSerializer(data=request.data)

        if serializer.is_valid():
            file = serializer.save()

            model = tf.keras.models.load_model('PredictModel/'+file.modelName+'.h5')
            img_r = cv2.imread(file.image.name)
            img1 = np.array(img_r).astype('float32') / 255
            img2 = transform.resize(img1, (128, 128, 3))
            img = np.expand_dims(img2, axis=0)
            r = model.predict(img)
            labels = ["benign", "malignant", "normal"]
            index = np.argmax(r)
            score=str(round(r[0][index] * 100, 1)) + "%"
            name = labels[index]
            with open(file.image.name, "rb") as img_file:
                b64_string = base64.b64encode(img_file.read())
            image = b64_string.decode()
            file.delete()
            return Response({"status": "success", "name": name, 'score':score,'image': image}, status=status.HTTP_200_OK)
        else:
            return Response({"status": "error", "data": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

