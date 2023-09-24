from tensorflow import keras
import keras.preprocessing.image as image
import numpy as np
import cv2
import gtts
import os

model = keras.models.load_model('C:\Coffee_Capsule_Classification\coffe_model.h5')

cap = cv2.VideoCapture(0)

def speak(text):
    tts = gtts.gTTS(text=text, lang='en')
    tts.save("C:\Coffee_Capsule_Classification\test.mp3")
    os.system("mpg123 " + "test.mp3")
    os.remove("C:\Coffee_Capsule_Classification\test.mp3")

def predict(frame):
    img = cv2.resize(frame, (150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    y_pred = model.predict(img_tensor)

    if y_pred[0][0] > y_pred[0][1] and y_pred[0][0] > y_pred[0][2]:
        speak('epro')
        return '2pro'
    elif y_pred[0][1] > y_pred[0][0] and y_pred[0][1] > y_pred[0][2]:
        speak('cider')
        return 'cider'
    elif y_pred[0][2] > y_pred[0][0] and y_pred[0][2] > y_pred[0][1]:
        speak('coke')
        return 'coke'
    else :
        speak('i can not recognize')

def speakDrinkInfo(drinkInfo, wantInfoList):
    if 0 in wantInfoList:
        speak(drinkInfo['name'])
    if 1 in wantInfoList:
        speak(drinkInfo['capacity'])
    if 2 in wantInfoList:
        speak('calorie'+drinkInfo['calorie'])
    if 3 in wantInfoList:
        speak('carbohydrate'+drinkInfo['carbohydrate'])
    if 4 in wantInfoList:
        speak('protein'+drinkInfo['protein'])
    if 5 in wantInfoList:
        speak('fat'+drinkInfo['fat'])
    if 6 in wantInfoList:
        speak('sodium'+drinkInfo['sodium'])
    if 7 in wantInfoList:
        speak('sugar'+drinkInfo['sugar'])
    if 8 in wantInfoList:
        speak('ingredient'+drinkInfo['ingredient'])

drinkName = ''

drink2pro = {
    'name': '2pro',
    'capacity': '240ml',
    'calorie': '65kcal',
    'carbohydrate': '16gram',
    'protein': '0gram',
    'fat': '0gram',
    'sodium': '5mg',
    'sugar': '15gram',
    'ingredient': 'Peach concentrate juice, purified water, high fructose liquid, synthetic flavoring, citric acid, trisodium citrate, enzyme treatment routine'
}

drinkCider = {
    'name': 'cider',
    'capacity': '250ml',
    'calorie': '110kcal',
    'carbohydrate': '28gram',
    'protein': '0gram',
    'fat': '0gram',
    'sodium': '5mg',
    'sugar': '27gram',
    'ingredient': 'Purified water, high fructose corn syrup, white sugar, carbon dioxide, citric acid, lemon lime flavor'
}

drinkCoke = {
    'name': 'coke',
    'capacity': '250ml',
    'calorie': '112kcal',
    'carbohydrate': '28gram',
    'protein': '0gram',
    'fat': '0gram',
    'sodium': '15mg',
    'sugar': '27gram',
    'ingredient': 'Purified water, sugar syrup, other fructose, sugar, carbon dioxide, caramel color, phosphoric acid, natural flavoring, caffeine'
}

while True:
    ret, frame = cap.read()
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    drinkName = predict(frame)
    if drinkName == '2pro':
        speakDrinkInfo(drink2pro, [0, 1, 2])
    elif drinkName == 'cider':
        speakDrinkInfo(drinkCider, [0, 1, 2])
    elif drinkName == 'coke':
        speakDrinkInfo(drinkCoke, [0, 1, 2])

cap.release()
cv2.destroyAllWindows()