# VIRTUAL-ASSISTANT-FOR-VISUALLY-IMPAIRED
#This is my internship repository on GITHUB

import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

from google.colab import drive 
drive.mount('/content/drive')

local_zip = '/content/drive/My Drive/Thai and Indian Currency Dataset256x256 (1).zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()
!rm -rf '/tmp/Thai and Indian Currency Dataset256x256/Thai Currencies'
!rm -rf '/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/2000/INDIA2000_16.jpg'

train_10_New = os.path.join('/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/10 New')
train_10_Old = os.path.join('/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/10 Old')
train_100_New = os.path.join('/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/100 New')
train_100_Old = os.path.join('/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/100 Old')
train_20 = os.path.join('/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/20')
train_200 = os.path.join('/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/200')
train_2000 = os.path.join('/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/2000')
train_50_New = os.path.join('/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/50 New')
train_50_Old = os.path.join('/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/50 Old')
train_500 = os.path.join('/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/500')

print('total training images of 10 New: ', len(os.listdir(train_10_New)))
print('total training images of 10 Old: ', len(os.listdir(train_10_Old)))
print('total training images of 100 New:', len(os.listdir(train_100_New)))
print('total training images of 100 Old:', len(os.listdir(train_100_Old)))
print('total training images of 20:     ', len(os.listdir(train_20)))
print('total training images of 200:    ', len(os.listdir(train_200)))
print('total training images of 2000:   ', len(os.listdir(train_2000)))
print('total training images of 50 New: ', len(os.listdir(train_50_New)))
print('total training images of 50 Old: ', len(os.listdir(train_50_Old)))
print('total training images of 500:    ', len(os.listdir(train_500)))

to_create = [
    '/tmp/indiancurrency',
    '/tmp/indiancurrency/training',
    '/tmp/indiancurrency/testing',
    '/tmp/indiancurrency/training/10New',
    '/tmp/indiancurrency/testing/10New',
    '/tmp/indiancurrency/training/10Old',
    '/tmp/indiancurrency/testing/10Old',
    '/tmp/indiancurrency/training/100New',
    '/tmp/indiancurrency/testing/100New',
    '/tmp/indiancurrency/training/100Old',
    '/tmp/indiancurrency/testing/100Old',
    '/tmp/indiancurrency/training/20',
    '/tmp/indiancurrency/testing/20',
    '/tmp/indiancurrency/training/200',
    '/tmp/indiancurrency/testing/200',
    '/tmp/indiancurrency/training/2000',
    '/tmp/indiancurrency/testing/2000',
    '/tmp/indiancurrency/training/50New',
    '/tmp/indiancurrency/testing/50New',
    '/tmp/indiancurrency/training/50Old',
    '/tmp/indiancurrency/testing/50Old',
    '/tmp/indiancurrency/training/500',
    '/tmp/indiancurrency/testing/500'
]

for directory in to_create:
    try:
        os.mkdir(directory)
        print(directory, 'created')
    except:
        print(directory, 'failed')
        
        
        def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    all_files = []
    
    for file_name in os.listdir(SOURCE):
        file_path = SOURCE + file_name

        if os.path.getsize(file_path):
            all_files.append(file_name)
        else:
            print('{} is zero length, so ignoring'.format(file_name))
    
    n_files = len(all_files)
    split_point = int(n_files * SPLIT_SIZE)
    
    shuffled = random.sample(all_files, n_files)
    
    train_set = shuffled[:split_point]
    test_set = shuffled[split_point:]
    
    for file_name in train_set:
        copyfile(SOURCE + file_name, TRAINING + file_name)
        
    for file_name in test_set:
        copyfile(SOURCE + file_name, TESTING + file_name)

split_size = .9

NEW10_SOURCE_DIR = r"/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/10 New/"
TRAINING_NEW10_DIR = r"/tmp/indiancurrency/training/10New/"
TESTING_NEW10_DIR = r"/tmp/indiancurrency/testing/10New/"

split_data(NEW10_SOURCE_DIR, TRAINING_NEW10_DIR, TESTING_NEW10_DIR, split_size)

Old10_SOURCE_DIR = r"/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/10 Old/"
TRAINING_OLD10_DIR = r"/tmp/indiancurrency/training/10New/"
TESTING_OLD10_DIR = r"/tmp/indiancurrency/testing/10New/"

split_data(Old10_SOURCE_DIR, TRAINING_OLD10_DIR, TESTING_OLD10_DIR, split_size)

NEW100_SOURCE_DIR = r"/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/100 New/"
TRAINING_NEW100_DIR = r"/tmp/indiancurrency/training/100New/"
TESTING_NEW100_DIR = r"/tmp/indiancurrency/testing/100New/"

split_data(NEW100_SOURCE_DIR, TRAINING_NEW100_DIR, TESTING_NEW100_DIR, split_size)

OLD100_SOURCE_DIR = r"/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/100 Old/"
TRAINING_OLD100_DIR = r"/tmp/indiancurrency/training/100Old/"
TESTING_OLD100_DIR = r"/tmp/indiancurrency/testing/100Old/"

split_data(OLD100_SOURCE_DIR, TRAINING_OLD100_DIR, TESTING_OLD100_DIR, split_size)

NEW20_SOURCE_DIR = r"/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/20/"
TRAINING_NEW20_DIR = r"/tmp/indiancurrency/training/20/"
TESTING_NEW20_DIR = r"/tmp/indiancurrency/testing/20/"

split_data(NEW20_SOURCE_DIR, TRAINING_NEW20_DIR, TESTING_NEW20_DIR, split_size)

NEW200_SOURCE_DIR = r"/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/200/"
TRAINING_NEW200_DIR = r"/tmp/indiancurrency/training/200/"
TESTING_NEW200_DIR = r"/tmp/indiancurrency/testing/200/"

split_data(NEW200_SOURCE_DIR, TRAINING_NEW200_DIR, TESTING_NEW200_DIR, split_size)

NEW2000_SOURCE_DIR = r"/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/2000/"
TRAINING_NEW2000_DIR = r"/tmp/indiancurrency/training/2000/"
TESTING_NEW2000_DIR = r"/tmp/indiancurrency/testing/2000/"

split_data(NEW2000_SOURCE_DIR, TRAINING_NEW2000_DIR, TESTING_NEW2000_DIR, split_size)

NEW50_SOURCE_DIR = r"/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/50 New/"
TRAINING_NEW50_DIR = r"/tmp/indiancurrency/training/50New/"
TESTING_NEW50_DIR = r"/tmp/indiancurrency/testing/50New/"

split_data(NEW50_SOURCE_DIR, TRAINING_NEW50_DIR, TESTING_NEW50_DIR, split_size)

OLD50_SOURCE_DIR = r"/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/50 Old/"
TRAINING_OLD50_DIR = r"/tmp/indiancurrency/training/50Old/"
TESTING_OLD50_DIR = r"/tmp/indiancurrency/testing/50Old/"

split_data(OLD50_SOURCE_DIR, TRAINING_OLD50_DIR, TESTING_OLD50_DIR, split_size)

NEW500_SOURCE_DIR = r"/tmp/Thai and Indian Currency Dataset256x256/Indian Currencies/500/"
TRAINING_NEW500_DIR = r"/tmp/indiancurrency/training/500/"
TESTING_NEW500_DIR = r"/tmp/indiancurrency/testing/500/"

split_data(NEW500_SOURCE_DIR, TRAINING_NEW500_DIR, TESTING_NEW500_DIR, split_size)

print(len(os.listdir(TRAINING_NEW10_DIR)))
print(len(os.listdir(TESTING_NEW10_DIR)))

print(len(os.listdir(TRAINING_OLD10_DIR)))
print(len(os.listdir(TESTING_OLD10_DIR)))

print(len(os.listdir(TRAINING_NEW100_DIR)))
print(len(os.listdir(TESTING_NEW100_DIR)))

print(len(os.listdir(TRAINING_OLD100_DIR)))
print(len(os.listdir(TESTING_OLD100_DIR)))

print(len(os.listdir(TRAINING_NEW20_DIR)))
print(len(os.listdir(TESTING_NEW20_DIR)))

print(len(os.listdir(TRAINING_NEW200_DIR)))
print(len(os.listdir(TESTING_NEW200_DIR)))

print(len(os.listdir(TRAINING_NEW2000_DIR)))
print(len(os.listdir(TESTING_NEW2000_DIR)))

print(len(os.listdir(TRAINING_NEW50_DIR)))
print(len(os.listdir(TESTING_NEW50_DIR)))

print(len(os.listdir(TRAINING_OLD50_DIR)))
print(len(os.listdir(TESTING_OLD50_DIR)))

print(len(os.listdir(TRAINING_NEW500_DIR)))
print(len(os.listdir(TESTING_NEW500_DIR)))


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.90):
      print("\nReached 85% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

model = tf.keras.models.Sequential([

    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

model.compile( optimizer='rmsprop',loss = 'categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=90,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
      
      TRAINING_DIR = '/tmp/indiancurrency/training/'
#train_datagen = ImageDataGenerator(rescale= 1/255)
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size=32,
    class_mode='categorical',
    target_size=(256, 256)
)

VALIDATION_DIR = '/tmp/indiancurrency/testing/'
validation_datagen = ImageDataGenerator(rescale= 1/255)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=32,
    class_mode='categorical',
    target_size=(256, 256)
)

history = model.fit(
      train_generator,
      epochs=15,  
      verbose=1,
      validation_data=validation_generator,
      callbacks=[callbacks])
      
      import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

import numpy as np
from google.colab import files
from tensorflow.keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path = fn
  img = image.load_img(path, target_size=(256, 256))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])

  #predictions = model.predict(images,batch_size=32)
  #class_names = ['10RsNote', '10RsNote', '100RsNote', '100RsNote','20RsNote','200RsNote', '2000RsNote','50RsNote','50RsNote','500RsNote']
  #print(model.predict_classes())
  print(np.argmax(model.predict(images), axis=-1))
  
  INDIAN_CURRENCY_SAVED_MODEL = "exp_saved_model"
tf.saved_model.save(model, INDIAN_CURRENCY_SAVED_MODEL)
loaded = tf.saved_model.load(INDIAN_CURRENCY_SAVED_MODEL)
print(list(loaded.signatures.keys()))
infer = loaded.signatures["serving_default"]
print(infer.structured_input_signature)
print(infer.structured_outputs)

converter = tf.lite.TFLiteConverter.from_saved_model(INDIAN_CURRENCY_SAVED_MODEL)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
tflite_model_file = 'converted_model.tflite'

with open(tflite_model_file, "wb") as f:
    f.write(tflite_model)
    
    labels = ['10 Rupees', 'Rupees 10','100 Rupees','Rupees 100','20 Rupees','200 Rupees','2000 Rupees','50 Rupees','Rupees 50','500 Rupees']

with open('labels.txt', 'w') as f:
    f.write('\n'.join(labels))
    
    
try:
    from google.colab import files
    files.download('converted_model.tflite')
    files.download('labels.txt')
except:
    pass
