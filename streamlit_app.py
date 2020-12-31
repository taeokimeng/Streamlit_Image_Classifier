"""Main module for the streamlit app"""
import streamlit as st
import src.image_upload
import src.image_classification
import src.data_plot
from src.resource import APP_MODE, CLASSIFICATION_MODEL, IMAGE_WIDTH

def main():
    """Main function of the App"""
    st.title("Image Classifier")
    st.sidebar.title("Menu")
    mode_selection = st.sidebar.radio("Please choose a mode", APP_MODE)
    # Upload an image
    image_pil = src.image_upload.upload_image()

    col1, col2 = st.beta_columns(2)
    # Use left column
    with col1:
        # Image has been selected
        if image_pil is not None:
            st.image(image_pil, caption=f"Uploaded image", width=IMAGE_WIDTH, use_column_width=False)

    # Classification mode
    if mode_selection == APP_MODE[0]:
        model_selection = st.sidebar.selectbox("Please choose a model", CLASSIFICATION_MODEL)
        # Image has been selected
        if image_pil is not None:
            answer, probability = src.image_classification.image_classifier(model_selection, image_pil)
            # Use right column
            with col2:
                # The result (prediction) from the classifier
                st.write("%s: ''%s'' (%.2f%%)" % (model_selection, answer, probability))

    # Comparison mode
    if mode_selection == APP_MODE[1]:
        model_selection = st.sidebar.multiselect("Please choose models", CLASSIFICATION_MODEL)
        # At least one model selected
        if len(model_selection) >= 1:
            # Image has been selected
            if image_pil is not None:
                answers, probabilities = [], []
                for model in model_selection:
                    answer, probability = src.image_classification.image_classifier(model, image_pil)
                    # Use right column
                    with col2:
                        # The result (prediction) from the classifier
                        st.write("%s: ''%s'' (%.2f%%)" % (model, answer, probability))
                    answers.append(answer)
                    probabilities.append(probability)
                # Plot bar chart for visualization
                else:
                    src.data_plot.plot_bar(model_selection, probabilities, answers)



if __name__ == '__main__':
    main()

