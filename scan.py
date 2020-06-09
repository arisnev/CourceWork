import sys
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

ans=""

while(1==1):
    ans=input("Введите путь до изображения ")
    if(ans=="ex"):
        break
    model=load_model('model.h5')
    print("Изображение загружено")

    img = image.load_img(ans, target_size=(300, 300))
    img_tensor = image.img_to_array(img)                  
    img_tensor = np.expand_dims(img_tensor, axis=0)        
    img_tensor /= 255.                                      
    classes = model.predict(img_tensor)
    print("Изображение распознанно")  
    skin_type= ['Чистая кожа', 'Псориаз']
    print(skin_type[0]," - ",classes[0][0]*100,"%")
    print(skin_type[1]," - ",classes[0][1]*100,"%")

    prediction = np.argmax(classes)
    print("Номер класса:", skin_type[prediction])
    plt.imshow(img.convert('RGBA'))

    plt.show()
