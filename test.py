from tensorflow import keras
import keras.utils as image
import numpy as np
import cv2
import gtts
import playsound
import os
import time

model = keras.models.load_model('C:/Coffee_Capsule_Classification/coffe_model.h5')

cap = cv2.VideoCapture(0)

def speak(text):
    tts = gtts.gTTS(text=text, lang='en')
    tts.save("C:/Coffee_Capsule_Classification/test.mp3")
    playsound.playsound("C:/Coffee_Capsule_Classification/test.mp3")
    os.remove("C:/Coffee_Capsule_Classification/test.mp3")

def predict(frame):
    img = cv2.resize(frame, (150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    y_pred = model.predict(img_tensor)

    if y_pred[0][0] > y_pred[0][1] and y_pred[0][0] > y_pred[0][2] and y_pred[0][0] > y_pred[0][3]:
        speak('Aarondio 150ml, grain scent, caramel scent')
        return 'Aarondio'
    elif y_pred[0][1] > y_pred[0][0] and y_pred[0][1] > y_pred[0][2] and y_pred[0][1] > y_pred[0][3]:
        speak('Diabolito 40ml, spicy scent, strong roasting scent')
        return 'Diabolito'
    elif y_pred[0][2] > y_pred[0][0] and y_pred[0][2] > y_pred[0][1] and y_pred[0][2] > y_pred[0][3]:
        speak('Intenso 230ml, caramel scent, roasting scent')
        return 'Intenso'
    elif y_pred[0][3] > y_pred[0][0] and y_pred[0][3] > y_pred[0][1] and y_pred[0][3] > y_pred[0][2]:
        speak('Portado 150ml, Strong roasting scent, woody scent')
        return 'Portado'
    else :
        return 'None'
        #speak('i can not recognize')

pt = time.time()
while True:
    ret, frame = cap.read()
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    ct = time.time()
    if ct - pt > 1:
        drinkName = predict(frame)
        print(drinkName)
        pt = ct

cap.release()
cv2.destroyAllWindows()