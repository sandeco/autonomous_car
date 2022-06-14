print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2


## SOCKET PARA COMUNICAÇÃO COM O SIMULADOR
sio = socketio.Server()

## FLASK PARA TROCA DE MENSAGENS ENTRE O SIMULADOR USANDO O SOCKET
app = Flask(__name__)

import tensorflow as tf

# SE EXISTIR GPU NA MÁQUINA RESERVAR 20%
if tf.config.list_physical_devices('GPU'):
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.2)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


## VELOCIDADE MÁXIMA DO CARRO
maxSpeed = 30


#PRÉ-PROCESSAMENTO DA IMAGEM
def preProcess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


@sio.on('telemetry')
def telemetry(sid, data):

    speed = float(data['speed']) #VELOCIDADE ATUAL NO SIMULADOR
    image = Image.open(BytesIO(base64.b64decode(data['image']))) #IMAGEM ATUAL DO SIMULADOR

    #AJUSTE DE IMAGEM
    image = np.asarray(image)
    image = preProcess(image)
    image = np.array([image])

    #PREDIÇÃO DO VOLANTE
    steering = float(model.predict(image))

    #CALCULANDO NOVA ACELERAÇÃO PARA O CARRO
    razao = abs(speed / maxSpeed)
    desc = abs(steering)
    throttle = 1.0 - desc - razao

    print("Direção : {:.6f}, Acelerador : {:.6f}, Velocidade: {:.6f}".format(steering, throttle, speed))

    #ENVIA CONTROLE PARA O SIMULADOR
    sendControl(steering, throttle)


#ENVIA COMANDO DE ACELERAÇÃO E ÂNGULO DE DIREÇÃO AO SIMULADOR
def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })


# CONEXÃO INICIAL
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)


if __name__ == '__main__':

    # CARREGANDO MODELO
    model = load_model('model.h5')

    #ABRINDO CONEXÃO COM O SIMULADOR USANDO O FLASK
    app = socketio.Middleware(sio, app)

    ## CONECTANDO NA PORTA 4567
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)