import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import EfficientNetB0
from keras import layers

# Especifica la ruta a la carpeta de imágenes
img_folder = 'trainedimages'

# Preprocesa las imágenes para el entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normaliza los valores de píxeles a [0,1]
    validation_split=0.2 # Separa un 20% de las imágenes para validación
)

# Carga las imágenes desde la carpeta especificada
train_generator = train_datagen.flow_from_directory(
    img_folder,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training' # Selecciona el conjunto de entrenamiento
)

# Carga las imágenes de validación
val_generator = train_datagen.flow_from_directory(
    img_folder,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation' # Selecciona el conjunto de validación
)

# Carga el modelo preentrenado EfficientNetB0 sin la capa final
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Congela los pesos del modelo preentrenado
base_model.trainable = False

# Añade las capas finales al modelo
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(2, activation='softmax')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

# Compila el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Entrena el modelo con el conjunto de entrenamiento y valida con el conjunto de validación
model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Guarda el modelo entrenado
model.save('modelstrained')
