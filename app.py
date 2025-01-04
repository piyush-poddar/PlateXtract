import streamlit as st
import os
from main import extract_text_from_image

st.title("PlateXtract")

st.header("Extract text from number plates with ease.")
st.subheader("Just upload an image of a vehicle with number plate and see the magic.")

File = st.file_uploader(label = "Upload Image")

if File:
    image_path = os.path.join('.', 'images', File.name)

    with open(image_path, mode='wb') as w:
        w.write(File.getvalue())

    if os.path.exists(image_path):
        st.success("Image uploaded successfully")
    
    st.image(image_path, width=500)

    if (st.button("Extract Text")):
        st.subheader(f"Number Plate Text: {extract_text_from_image(image_path)}")