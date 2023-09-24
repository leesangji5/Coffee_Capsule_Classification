from tensorflow import keras
import keras.utils as image
import numpy as np
import cv2

model = keras.models.load_model('C:/Coffee_Capsule_Classification/coffe_model.h5')

img = [
    'C:/Coffee_Capsule_Classification/coffe_images/test/Aarondio/1.jpg',
    'C:/Coffee_Capsule_Classification/coffe_images/test/Aarondio/2.jpg',
    'C:/Coffee_Capsule_Classification/coffe_images/test/Aarondio/3.jpg',
    'C:/Coffee_Capsule_Classification/coffe_images/test/Aarondio/4.jpg',
    'C:/Coffee_Capsule_Classification/coffe_images/test/Diabolito/1.jpg',
    'C:/Coffee_Capsule_Classification/coffe_images/test/Diabolito/2.jpg',
    'C:/Coffee_Capsule_Classification/coffe_images/test/Diabolito/3.jpg',
    'C:/Coffee_Capsule_Classification/coffe_images/test/Diabolito/4.jpg',
    'C:/Coffee_Capsule_Classification/coffe_images/test/Intenso/1.jpg',
    'C:/Coffee_Capsule_Classification/coffe_images/test/Intenso/2.jpg',
    'C:/Coffee_Capsule_Classification/coffe_images/test/Intenso/3.jpg',
    'C:/Coffee_Capsule_Classification/coffe_images/test/Intenso/4.jpg',
    'C:/Coffee_Capsule_Classification/coffe_images/test/Portado/1.jpg',
    'C:/Coffee_Capsule_Classification/coffe_images/test/Portado/2.jpg',
    'C:/Coffee_Capsule_Classification/coffe_images/test/Portado/3.jpg',
    'C:/Coffee_Capsule_Classification/coffe_images/test/Portado/4.jpg'
]

def predict(frame):
    img = cv2.resize(frame, (150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    y_pred = model.predict(img_tensor)

    if y_pred[0][0] > y_pred[0][1] and y_pred[0][0] > y_pred[0][2] and y_pred[0][0] > y_pred[0][3]:
        return 'Aarondio'
    elif y_pred[0][1] > y_pred[0][0] and y_pred[0][1] > y_pred[0][2] and y_pred[0][1] > y_pred[0][3]:
        return 'Diabolito'
    elif y_pred[0][2] > y_pred[0][0] and y_pred[0][2] > y_pred[0][1] and y_pred[0][2] > y_pred[0][3]:
        return 'Intenso'
    elif y_pred[0][3] > y_pred[0][0] and y_pred[0][3] > y_pred[0][1] and y_pred[0][3] > y_pred[0][2]:
        return 'Portado'
    else :
        return 'None'

a = [predict(img[0]), predict(img[1]), predict(img[2]), predict(img[3])]
b = [predict(img[4]), predict(img[5]), predict(img[6]), predict(img[7])]
c = [predict(img[8]), predict(img[9]), predict(img[10]), predict(img[11])]
d = [predict(img[12]), predict(img[13]), predict(img[14]), predict(img[15])]

print(a)
print(b)
print(c)
print(d)