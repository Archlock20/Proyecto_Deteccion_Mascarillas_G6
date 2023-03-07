# accounts/views.py
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.views import generic
from django.views import View
from .forms import CustomUserCreationForm
import cv2
import os
from django.conf import settings
import numpy as np
import tensorflow as tf
from scipy.special import softmax
from django.shortcuts import render
from django.http import HttpResponse
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
from django.views.decorators.csrf import csrf_exempt



class SignUpView(generic.CreateView):
    form_class = CustomUserCreationForm
    success_url = reverse_lazy("login")
    template_name = "registration/signup.html"
    
    
# Importar el modelo de reconocimiento de mascarillas
model = tf.keras.models.load_model('static/modelo_cnn_deteccion_mascarillas_AG_MT_V3')


def detect_mask(img):
        face_detection_model = cv2.dnn.readNetFromCaffe('static/deploy.prototxt.txt',
                                                    'static/res10_300x300_ssd_iter_140000_fp16.caffemodel')
        
        etiquetas = ['Con Mascarilla', 'Sin Mascarilla']

        def getcolor(etiqueta):
            if etiqueta == "Con Mascarilla":
                color = (0,255,0)
            else: 
                color = (0,0,255)
            return color
        
        # Paso 1 - Deteccion de rostros
        image = img.copy()
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image,1,(300,300),(104,117,123),swapRB=True)
        #
        face_detection_model.setInput(blob)
        detection = face_detection_model.forward()
        for i in range(0,detection.shape[2]):
            confidence = detection[0,0,i,2]
            if confidence > 0.5:
                box = detection[0,0,i,3:7]*np.array([w,h,w,h])
                box = box.astype(int)
                pt1 = (box[0],box[1])
                pt2 = (box[2],box[3])
                cv2.rectangle(image,pt1,pt2,(0,255,0),1)    
                # Paso 2 - Preprosesamiento de los datos
                face = image[box[1]:box[3],box[0]:box[2]]
                face_blob = cv2.dnn.blobFromImage(face,1,(100,100),(104,117,123),swapRB=True)
                face_blob_squeeze = np.squeeze(face_blob).T
                face_blob_rotate = cv2.rotate(face_blob_squeeze,cv2.ROTATE_90_CLOCKWISE)
                face_blob_flip = cv2.flip(face_blob_rotate,1)
                # Paso 3 - Normalizacion
                img_norm = np.maximum(face_blob_flip,0)/face_blob_flip.max()
                # Paso 4 - Aprendizaje Profundo (CNN)
                img_input = img_norm.reshape(1,100,100,3)
                result = model.predict(img_input)
                result = softmax(result)[0]
                confidence_index = result.argmax()
                confidence_score = result[confidence_index]
                label = etiquetas[confidence_index]
                label_text = '{}: {:,.0f}%'.format(label,confidence_score*100)
                #print(label_text)
                color = getcolor(label)
                cv2.rectangle(image,pt1,pt2,color,2)
                cv2.putText(image,label_text,pt1,cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
                
        return image

def img_mask_detection(request):
    if request.method == 'POST':
        image = cv2.imdecode(np.fromstring(request.FILES['image'].read(), np.uint8), cv2.IMREAD_COLOR)
        result_image = detect_mask(image)
        _, buffer = cv2.imencode('.jpg', result_image)
        response = HttpResponse(buffer.tobytes(), content_type='image/jpeg')
        return response
    else:
        return render(request, 'img_mask_detection.html')


# vista para transmitir vídeo con la detección de mascarillas en tiempo real
@gzip.gzip_page
@csrf_exempt

def live_mask_detection(request):
    # captura el video de la cámara web
    cap = cv2.VideoCapture(0)

    def generate():
        while True:
            # lee un cuadro del video
            ret, frame = cap.read()
            if not ret:
                break

            # detecta mascarillas en el cuadro
            frame = detect_mask(frame)

            # codifica el cuadro en JPEG y lo envía
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    return StreamingHttpResponse(generate(), content_type="multipart/x-mixed-replace;boundary=frame")


def live_mask_detection_view(request):
    return render(request, 'live_mask_detection.html')
