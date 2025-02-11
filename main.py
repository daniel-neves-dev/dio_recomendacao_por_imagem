# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from annoy import AnnoyIndex
import pickle
from pathlib import Path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

############################
# 1) Load Pretrained Artifacts
############################

module_2 = hub.load("https://www.kaggle.com/models/google/bit/TensorFlow2/s-r50x3-ilsvrc2012-classification/1")

dims = 1000
annoy_index_path = "content/img_vectors/indexer.ann"
t = AnnoyIndex(dims, metric='angular')
t.load(annoy_index_path)

file_index_to_file_name = pickle.load(open("content/img_vectors/file_index_to_file_name.p", "rb"))
file_index_to_product_id = pickle.load(open("content/img_vectors/file_index_to_file_vector.p", "rb"))
# or whichever dictionary you need

styles_path = "content/styles.csv"
styles = pd.read_csv(styles_path, on_bad_lines="skip")
styles["id"] = styles["id"].astype(str)

data_dir = Path("content/images/categories")
path_dict = {}
for p in data_dir.rglob('*.jpg'):
    path_dict[p.stem] = str(p)


############################
# 2) Helper: Preprocess a Single Image
############################

def load_img_for_module(img_bytes):
    img_decoded = tf.image.decode_jpeg(img_bytes, channels=3)
    img_resized = tf.image.resize_with_pad(img_decoded, 224, 224)
    img_float = tf.image.convert_image_dtype(img_resized, tf.float32)
    return tf.expand_dims(img_float, axis=0)


############################
# 3) Streamlit Interface
############################

st.title("Image Similarity Demo")

# Let user upload an image
uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the query image
    st.image(uploaded_file, caption="Query Image", use_column_width=True)

    # Preprocess & get embedding
    file_bytes = uploaded_file.read()
    input_tensor = load_img_for_module(file_bytes)
    test_vec = module_2(input_tensor)
    test_vec = np.squeeze(test_vec)

    # Let the user select topK
    topK = st.slider("Number of similar images to retrieve:", min_value=1, max_value=10, value=4)
    nn_indices = t.get_nns_by_vector(test_vec, n=topK)

    # Show the results in a horizontal layout
    cols = st.columns(topK)
    for i, idx in enumerate(nn_indices):
        file_stem = file_index_to_file_name[idx]
        product_id = file_index_to_product_id[idx]

        # if we have the row in styles
        if product_id != -1:
            row = styles.loc[product_id]
            info = f"""
            **ID**: {row['id']}  
            **Category**: {row.get('masterCategory', '')}  
            **SubCat**: {row.get('subCategory', '')}  
            """
        else:
            info = f"ID: {file_stem}\n(No CSV metadata)"

        neighbor_img_path = path_dict.get(file_stem, None)
        if neighbor_img_path:
            with cols[i]:
                st.image(neighbor_img_path, caption=info, use_column_width=True)
