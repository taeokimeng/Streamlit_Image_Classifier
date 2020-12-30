import numpy as np
import streamlit as st
from keras.preprocessing import image
from src.resource import CLASSIFICATION_MODEL, TARGET_SIZE_224, TARGET_SIZE_299, TARGET_SIZE_331

@st.cache
def vgg16_classifier():
    from keras.applications.vgg16 import VGG16
    return VGG16(), TARGET_SIZE_224

@st.cache
def resnet50_classifier():
    from keras.applications.resnet50 import ResNet50
    return ResNet50(), TARGET_SIZE_224

@st.cache
def resnet101v2_classifier():
    from keras.applications.resnet_v2 import ResNet101V2
    return ResNet101V2(), TARGET_SIZE_224

@st.cache
def xception_classifier():
    from keras.applications.xception import Xception
    return Xception(), TARGET_SIZE_299

@st.cache
def nasnetlarge_classifier():
    from keras.applications.nasnet import NASNetLarge
    return NASNetLarge(), TARGET_SIZE_331

def image_classifier(selected_model, loaded_image):
    # Which model did you choose?
    if selected_model == CLASSIFICATION_MODEL[0]: # VGG16
        from keras.applications.vgg16 import preprocess_input, decode_predictions
        classifier, target_size = vgg16_classifier()
    if selected_model == CLASSIFICATION_MODEL[1]: # ResNet50
        from keras.applications.resnet50 import preprocess_input, decode_predictions
        classifier, target_size = resnet50_classifier()
    if selected_model == CLASSIFICATION_MODEL[2]: # ResNet101V2
        from keras.applications.resnet_v2 import preprocess_input, decode_predictions
        classifier, target_size = resnet101v2_classifier()
    if selected_model == CLASSIFICATION_MODEL[3]: # Xception
        from keras.applications.xception import preprocess_input, decode_predictions
        classifier, target_size = xception_classifier()
    if selected_model == CLASSIFICATION_MODEL[4]: # NASNetLarge
        from keras.applications.nasnet import preprocess_input, decode_predictions
        classifier, target_size = nasnetlarge_classifier()

    # You should make the size to the expected size
    new_image = loaded_image.resize(target_size)

    transformed_image = image.img_to_array(new_image)
    # 4D (batch_size, width, height, channels)
    transformed_image = np.expand_dims(transformed_image, axis=0)
    transformed_image = preprocess_input(transformed_image)

    y_pred = classifier.predict(transformed_image)
    decode_predictions(y_pred, top=5)
    label = decode_predictions(y_pred)
    # Retrieve the most likely result, e.g. highest probability
    decoded_label = label[0][0]
    # The result (prediction) from the classifier
    st.write("%s: ''%s'' (%.2f%%)" % (selected_model, decoded_label[1], decoded_label[2]*100))
    # Return "Top 1 result" and "The probability"
    return decoded_label[1], float(decoded_label[2]*100)