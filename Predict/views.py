import shutil

from django.shortcuts import render
import cv2
import tensorflow as tf
import numpy as np
import os
import io
from skimage import transform
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.switch_backend('agg')

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
    """
    Input :  model,image
    Output : prediction result
    """ 
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
    """
    Input : image_path
    Output : image with numpy array for channel-wise color normalization
    """     
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
    """
    Input : model,image
    Output : image with gradcam
    """    
    @staticmethod    
    def gradcam(model,file):
        img_path=file.image.name
        x_img = ImageView.process_image(img_path)
        #print_predicted_result(x_img)
    
        img_sample = cv2.imread(img_path)
        #plt.rcParams['figure.figsize'] = (10.0, 10.0)
    
        # Get the output feature map from the target layer
        target_layer = model.get_layer("conv_pw_13_relu")
        # target_layer = model.get_layer("mobilenet_1.00_224")
    
        prediction = model.predict(x_img)
        prediction_idx = np.argmax(prediction)
    
        # Fix gradient error
        with tf.GradientTape() as tape:
            # Create a model with original model inputs and the last conv_layer as the output
            gradient_model = Model([model.inputs], [target_layer.output, model.output])
            # Pass the image through the base model and get the feature map  
            conv2d_out, prediction = gradient_model(x_img)
            # Prediction loss
            loss = prediction[:, prediction_idx]
    
        # gradient() computes the gradient using operations recorded in context of this tape
        gradients = tape.gradient(loss, conv2d_out)
    
        # Obtain the output from shape [1 x H x W x CHANNEL] -> [H x W x CHANNEL]
        output = conv2d_out[0]
    
        # Obtain depthwise mean
        weights = tf.reduce_mean(gradients[0], axis=(0, 1))
    
        # Create a 7x7 map for aggregation
        activation_map = np.zeros(output.shape[0:2], dtype=np.float32)
    
        # Multiply weight for every layer
        for idx, weight in enumerate(weights):
            activation_map += weight * output[:, :, idx]
    
        test_img = cv2.imread(img_path)
        original_img = np.asarray(test_img, dtype = np.float32)
    
        # Resize to image size
        activation_map = cv2.resize(activation_map.numpy(), 
                                    (original_img.shape[1], 
                                    original_img.shape[0]))
        
        # Ensure no negative number
        activation_map = np.maximum(activation_map, 0)
    
        # Convert class activation map to 0 - 255
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
        
        # Rescale and convert the type to int
        activation_map = np.uint8(255 * activation_map)
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
    
        # Superimpose heatmap onto image
        original_img = np.uint8((original_img - original_img.min()) / (original_img.max() - original_img.min()) * 255)
        cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        cvt_heatmap = img_to_array(cvt_heatmap)
        plt.rcParams["figure.dpi"] = 100
        interpolant=0.6
    
        plt.margins(x=0)
    
        plt.imshow(np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant)))
        plt.axis("off")
        plt.savefig('gradcam.png', transparent=True, bbox_inches='tight')
        plt.close()
        with open('gradcam.png', "rb") as img_file:
            b64_string = base64.b64encode(img_file.read())
            image_base64 = b64_string.decode()
            return image_base64
    def post(self, request):

        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            file = serializer.save()
            model = tf.keras.models.load_model('PredictModel/'+file.modelName)
            
            image=ImageView.process_image(file.image.name)
            data=ImageView.print_predicted_result(model,image)
            image=ImageView.gradcam(model,file)
            file.delete()
            return Response({"status": "success", "data": data,'image': image}, status=status.HTTP_200_OK)
        else:
            return Response({"status": "error", "data": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

