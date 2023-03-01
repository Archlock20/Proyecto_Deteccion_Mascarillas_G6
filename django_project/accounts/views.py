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
    
    
""" def face_mask_prediction(img):
        face_detection_model = cv2.dnn.readNetFromCaffe('static/deploy.prototxt.txt',
                                                    'static/res10_300x300_ssd_iter_140000_fp16.caffemodel')
        
        # Importar el modelo de reconocimiento de mascarillas
        model = tf.keras.models.load_model('static/modelo_cnn_deteccion_mascarillas_AG_MT_V3')
        
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
                cv2.rectangle(image,pt1,pt2,color,1)
                cv2.putText(image,label_text,pt1,cv2.FONT_HERSHEY_PLAIN,1,color,1)
                
        return image    
    
    
class mask_detection_view(View):
    
    def get(self, request):
        captura = cv2.VideoCapture(0)

        while True:
            ret, frames = captura.read()
            if ret == False:
                break
                
            image = face_mask_prediction(frames)
            cv2.imshow('Deteccion de mascarillas', image)
            if cv2.waitKey(1) == 27:
                break
                
        captura.release()
        cv2.destroyAllWindows()
        
        return render(request, 'facemask_detection.html') """



""" ------------------------------NUEVO---------------------------------------  """
""" # carga el modelo de detección de mascarillas
mask_detector = tf.keras.models.load_model('static/modelo_cnn_deteccion_mascarillas_AG_MT_V3')

# función para detectar mascarillas en una imagen
def detect_mask(img):
    # convierte la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detecta los rostros en la imagen utilizando el modelo de detección de rostros
    face_detector = cv2.dnn.readNetFromCaffe('static/deploy.prototxt.txt',
                                                    'static/res10_300x300_ssd_iter_140000_fp16.caffemodel')
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    # para cada rostro detectado, determina si lleva mascarilla o no
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # filtra detecciones débiles
        if confidence < 0.5:
            continue

        # obtiene las coordenadas de la caja delimitadora y calcula las coordenadas (x, y) de la cara
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        face = img[startY:endY, startX:endX]

        # procesa la cara para detectar si lleva mascarilla o no
        face = cv2.resize(face, (100, 100))
        face = np.expand_dims(face, axis=0)
        face = face / 255.0  # normaliza la imagen
        pred = mask_detector.predict(face)
        if pred[0][0] > pred[0][1]:
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(img, "Mask", (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(img, "No Mask", (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return img

 """


""" ------------------------------NUEVO---------------------------------------  """

""" # Cargar el modelo de detección de mascarillas
model = tf.keras.models.load_model('static/modelo_cnn_deteccion_mascarillas_AG_MT_V3')

def detect_mask(img):
    # Cargar el modelo de detección de rostros
    face_detection_model = cv2.dnn.readNetFromCaffe('static/deploy.prototxt.txt', 'static/res10_300x300_ssd_iter_140000_fp16.caffemodel')

    # Etiquetas de las clases
    etiquetas = ['Con Mascarilla', 'Sin Mascarilla']

    # Función para obtener el color correspondiente a la etiqueta
    def getcolor(etiqueta):
        if etiqueta == "Con Mascarilla":
            color = (0,255,0)
        else: 
            color = (0,0,255)
        return color

    # Detección de rostros
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_detection_model.setInput(blob)
    detections = face_detection_model.forward()

    # Iterar sobre las detecciones
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Comprobar si la detección cumple con el umbral de confianza
        if confidence > 0.5:
            # Obtener las coordenadas de la caja delimitadora
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Obtener el rostro de la imagen
            face = img[startY:endY, startX:endX]

            # Preprocesamiento del rostro
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (100, 100), (104.0, 177.0, 123.0))
            face_blob_squeeze = np.squeeze(face_blob).T
            face_blob_rotate = cv2.rotate(face_blob_squeeze,cv2.ROTATE_90_CLOCKWISE)
            face_blob_flip = cv2.flip(face_blob_rotate,1)

            # Normalización del rostro
            img_norm = np.maximum(face_blob_flip,0)/face_blob_flip.max()

            # Aprendizaje Profundo (CNN)
            img_input = img_norm.reshape(1,100,100,3)
            result = model.predict(img_input)
            result = softmax(result)[0]
            confidence_index = result.argmax()
            confidence_score = result[confidence_index]
            label = etiquetas[confidence_index]
            label_text = '{}: {:,.0f}%'.format(label, confidence_score * 100)

            # Dibujar la caja delimitadora y el resultado de la detección
            color = getcolor(label)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
            cv2.putText(img, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img """


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
                cv2.rectangle(image,pt1,pt2,color,1)
                cv2.putText(image,label_text,pt1,cv2.FONT_HERSHEY_PLAIN,1,color,1)
                
        return image












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
