import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
model = tf.keras.models.load_model(" ") #file-path of saved model.
map_dict = {
    0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: '10',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}
uploaded_file = st.file_uploader("Choose an image file", type="png")
if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(opencv_image, (64, 64))
    st.image(opencv_image, channels="RGB", caption="Uploaded Image"
    if st.button("Guess The Sign"):
        img_reshape = resized_image[np.newaxis, ...]
        prediction = model.predict(img_reshape)
        st.title("Predicted Sign: {}".format(map_dict[prediction]))
