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
from django.contrib.auth.models import User
from keras.preprocessing                  import image
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.mobilenet         import preprocess_input
from keras.models                         import Model

# from mafia_api.settings import MEDIA_ROOT, BASE_DIR, DETECT_ROOT

# Create your views here.
class ImageView(APIView):
    @staticmethod
    def print_predicted_result(model,img):
        
        pred = model.predict(img)
        y_classes = [np.argmax(element) for element in pred]
        label_classes = ['benign', 'malignant', 'normal']
        sorted_category = np.argsort(pred[0])[:-4:-1]
        data_result = {}
        for i in range(3):
            print("{}".format(label_classes[sorted_category[i]]) + " ({:.4})".format(pred[0][sorted_category[i]]))
            data_result.update({label_classes[sorted_category[i]]:pred[0][sorted_category[i]] * 100})
        y_pred = label_classes[y_classes[0]]
    
        # print("Our proposed model is {:.2%}".format(max(pred[0])) + " sure this is {}".format(y_pred))
        # print(data_result)
        return data_result
    @staticmethod
    def process_image(img_path):
        
        pic_size = 256
        # Load the image based on the provided path with the target size
        img = image.load_img(img_path, target_size=(pic_size, pic_size))
        
        # Convert image to numpy array of shape (256, 256, 3)
        x_img = image.img_to_array(img)
    
        # Transform the image array into a matrix of shape (1, 256, 256, 3)
        x_img = np.expand_dims(x_img, axis=0)
    
        # Preprocess the batch (channel-wise color normalization)
        x_img = preprocess_input(x_img)
    
        return x_img
    def post(self, request):

        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            file = serializer.save()
            model = tf.keras.models.load_model('PredictModel/'+file.modelName)
            
            image=ImageView.process_image(file.image.name)
            data=ImageView.print_predicted_result(model,image)
            """img_r = cv2.imread(file.image.name)
            img1 = np.array(img_r).astype('float32') / 255
            img2 = transform.resize(img1, (128, 128, 3))
            img = np.expand_dims(img2, axis=0)
            r = model.predict(img)
            labels = ["benign", "malignant", "normal"]
            index = np.argmax(r)
            score=str(round(r[0][index] * 100, 1)) + "%"
            name = labels[index]
           
           
            file.delete()
            print({"status": "success", "name": name, 'score':score,'image': ''});"""
            with open(file.image.name, "rb") as img_file:
                b64_string = base64.b64encode(img_file.read())
            image = b64_string.decode()
            return Response({"status": "success", "data": data,'image': image}, status=status.HTTP_200_OK)
        else:
            return Response({"status": "error", "data": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

