from PIL import Image
import numpy as np
import keras

data=np.random.normal(0,255,size=(32,32,3))
image=keras.preprocessing.image.array_to_img(data)
image.show()
