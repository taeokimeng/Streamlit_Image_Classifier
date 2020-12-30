import streamlit as st
from PIL import Image
from src.resource import IMAGE_WIDTH

def upload_image():
    uploaded_image = st.file_uploader("Please choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
        except Exception: # Check invalid image
            st.error("Error: Invalid image")
        else:
            st.image(image, caption=f"Uploaded image", width=IMAGE_WIDTH, use_column_width=False)
            return image



    """
    # This is for multiple files upload (on going...)
    uploaded_images = st.file_uploader("Please choose a image file", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    for uploaded_image in uploaded_images:
        #bytes_data = uploaded_image.read()
        #image = np.array(Image.open(uploaded_image))
        image = Image.open(uploaded_image)
        st.write("filename:", uploaded_image.name)
        st.write(f"{type(image)}")
        #st.write("%s" % ("pizza"))
        #st.write(bytes_data)
        st.image(image, caption=f"Uploaded image", width=196, use_column_width=False)
        return image
    """
