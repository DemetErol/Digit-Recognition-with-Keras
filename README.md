# Digit-Recognition-with-Keras


Loading minst data

```
from keras.datasets import mnist
(tr_img,tr_label),(test_img,test_label)=mnist.load_data()
```

Data example;

```
plt.imshow(tr_img[2])
```

![image](https://user-images.githubusercontent.com/45537416/118357269-e73ee700-b581-11eb-82a9-cb1196e2088e.png)

Data normalization and reshape

```
def preparation(df):
    df=df/255 #normalization
    df=df.reshape(-1,28,28,1)
    return df
```

Split data as train and validation

```
X_train, X_val, y_train, y_val = train_test_split(tr_img, tr_label, test_size = 0.2, random_state = 0)
```

Data augmentation

```
datagen= ImageDataGenerator(rotation_range=10, zoom_range = 0.2, width_shift_range=0.1, height_shift_range=0.1)  
data=datagen.flow(X_train, y_train)
```

![Ekran Alıntısı](https://user-images.githubusercontent.com/45537416/118545833-d055eb80-b75f-11eb-8405-6d96265f94bd.PNG)

![image](https://user-images.githubusercontent.com/45537416/118545856-d8ae2680-b75f-11eb-9379-00d71717757c.png)


Create Model and fit

```
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28,28,1), activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add((layers.Flatten()))
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dropout(.5))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(.5))
model.add(layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()
```

![image](https://user-images.githubusercontent.com/45537416/118357387-8ebc1980-b582-11eb-88e4-a405cc21169c.png)

```
model.fit(data, validation_data=(X_val, y_val), epochs=5)
```
![image](https://user-images.githubusercontent.com/45537416/118545936-f3809b00-b75f-11eb-9fff-97670bb1bfed.png)

Prediction result map

![image](https://user-images.githubusercontent.com/45537416/118546027-11e69680-b760-11eb-9ed2-c1c1c290e78a.png)
