import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()
 
train_filter = y_train < 5
test_filter = y_test < 5
X_train, y_train = X_train[train_filter], y_train[train_filter]
X_test, y_test = X_test[test_filter], y_test[test_filter]

 
X_train = X_train / 255.0
X_test = X_test / 255.0
 
y_train_cat = to_categorical(y_train, 5)
y_test_cat = to_categorical(y_test, 5)


model = Sequential([
    Flatten(input_shape=(28, 28)),    
    Dense(10, activation='relu'),     
    Dense(5, activation='softmax')     
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train_cat, epochs=5, batch_size=32, validation_split=0.1)


test_loss, test_accuracy = model.evaluate(X_test, y_test_cat)
print(f"\n Précision du modèle sur les chiffres 0 à 4 : {test_accuracy:.2f}")


print("\n Voici 5 prédictions du modèle :\n")

for i in range(5):
    image = X_test[i]
    vrai_chiffre = y_test[i]

  
    plt.imshow(image, cmap='gray')
    plt.title(f"Image n°{i+1} - Chiffre Réel : {vrai_chiffre}")
    plt.axis('off')
    plt.show()

    prediction = model.predict(np.expand_dims(image, axis=0))
    chiffre_predit = np.argmax(prediction)

    print(f" Prédiction {i+1} : {chiffre_predit}")
    print(" Bonne réponse" if chiffre_predit == vrai_chiffre else " Mauvaise réponse")
    print("-" * 30)




    https://github.com/miari123/DevoirIA.git
CC01-F037



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