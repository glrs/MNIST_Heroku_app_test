import streamlit as st
import tensorflow as tf
from skimage.transform import rescale
from skimage.color import rgb2gray

from streamlit_drawable_canvas import st_canvas


def main():
    st.title("MNIST Number Prediction")
    left_column, right_column = st.columns(2)

    # Load model
    model = tf.keras.models.load_model('tf_MNIST_app')

    # Create a canvas component
    with left_column:
        st.header("Draw a number")
        st.subheader("[0-9]")
        canvas_result = st_canvas(
                fill_color="rgb(0,0,0)",
                stroke_width=10,
                stroke_color="#FFFFFF",
                background_color="#000000",
                update_streamlit=True,
                width=280,
                height=280,
                drawing_mode="freedraw",
                key="canvas",
        )

    # Use image data to predict a number
    img = canvas_result.image_data
    if img is not None:
        with right_column:
            st.header("Predicted Result")
            # Result
            user_img = rescale(rgb2gray(img[...,0:3]), 0.1, anti_aliasing=False)
            st.image(user_img)
            user_img = tf.reshape(user_img, [-1, 784])
            pred = model.predict(user_img)
            st.write(pred.argmax())

if __name__ == "__main__":
    main()
