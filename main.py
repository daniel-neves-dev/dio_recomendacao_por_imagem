import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from shutil import move
import tensorflow as tf
import tensorflow_hub as hub

import json
from annoy import AnnoyIndex
from scipy import spatial
import pickle
import glob

from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

tqdm.pandas()

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


# 1) Move Images to Subfolders by masterCategory (from styles.csv)
base_dir = Path("content")
images_dir = base_dir / "images"
csv_path = base_dir / "styles.csv"
categories_dir = images_dir / "categories"

# Read CSV, keep only relevant columns
df = pd.read_csv(csv_path, usecols=["id", "masterCategory"], on_bad_lines="skip").reset_index(drop=True)
df["id"] = df["id"].astype(str)

all_images = os.listdir(images_dir)

co = 0
for image in tqdm(all_images, desc="Moving images"):
    image_id = image.split(".")[0]  # e.g., "1525" from "1525.jpg"
    category = df.loc[df["id"] == image_id, "masterCategory"]
    if not category.empty:
        category_name = category.values[0]

        target_folder = categories_dir / category_name
        target_folder.mkdir(parents=True, exist_ok=True)

        # Move file
        path_from = images_dir / image
        path_to = target_folder / image
        move(path_from, path_to)
        co += 1

print(f"Moved {co} images into category subfolders.")

# 2) (Optional) Keras Classification Model with TF Hub (BiT)

MODULE_HANDLE = 'https://tfhub.dev/google/bit/m-r50x3/1'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
N_FEATURES = 256

print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

data_dir = str(categories_dir)  # "content/images/categories"

# Prepare data generators (Validation split: 20%)
datagen_kwargs = dict(rescale=1. / 255, validation_split=0.20)
dataflow_kwargs = dict(target_size=IMAGE_SIZE,
                       batch_size=BATCH_SIZE,
                       interpolation="bilinear")

# Validation generator
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False, **dataflow_kwargs
)

# Training generator (can enable augmentation if desired)
do_data_augmentation = False
if do_data_augmentation:
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        **datagen_kwargs
    )
else:
    train_datagen = valid_datagen  # reuse same config

train_generator = train_datagen.flow_from_directory(
    data_dir, subset="training", shuffle=True, **dataflow_kwargs
)

#Build a model that uses the TF Hub BiT layer
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=False),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(N_FEATURES, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(train_generator.num_classes, kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])

model.build((None,) + IMAGE_SIZE + (3,))
model.summary()

# 3)Feature Extraction on a Small Fraction of Images

module_handle_2 = "https://www.kaggle.com/models/google/bit/TensorFlow2/s-r50x3-ilsvrc2012-classification/1"
print(f"\nLoading raw feature-extraction module:\n{module_handle_2}")
module_2 = hub.load(module_handle_2)

# Build a list of *all* images in categories_dir, then take a fraction for testing
img_paths = [str(p) for p in Path(data_dir).rglob('*.jpg')]
np.random.shuffle(img_paths)

# e.g., use 0.1% of images
SAMPLE_FRACTION = 0.001
num_samples = int(len(img_paths) * SAMPLE_FRACTION)
img_paths = img_paths[:num_samples]

print(f"Using {len(img_paths)} images out of {len(np.asarray(list(Path(data_dir).rglob('*.jpg'))))} total for feature extraction.")

def load_img_tf(path):
    """Loads and preprocesses an image (for batch dataset usage)."""
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_pad(img, 224, 224)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

dataset = tf.data.Dataset.from_tensor_slices(img_paths)
dataset = dataset.map(load_img_tf, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Directory where we'll save the extracted embeddings
imgvec_path = 'content/img_vectors/'
Path(imgvec_path).mkdir(parents=True, exist_ok=True)

batches = []
for i in range(0, len(img_paths), BATCH_SIZE):
    batches.append(img_paths[i : i + BATCH_SIZE])

idx = 0
for batch_images, file_batch in zip(dataset, batches):
    # Extract embeddings
    features_batch = module_2(batch_images)
    for i, features in enumerate(features_batch):
        feature_set = features.numpy()
        filename = os.path.basename(file_batch[i])
        outfile_name = filename.split('.')[0] + ".npy"
        out_path_file = os.path.join(imgvec_path, outfile_name)
        # Save as .npy
        np.save(out_path_file, feature_set)
    idx += len(file_batch)
    print(f"Extracted features for {idx} / {len(img_paths)} images.")

# 4) Quick Test: Display One Image
test_img = 'content/images/categories/Home/40826.jpg'
print("\nTesting display of:", test_img)
exists = os.path.exists(test_img)
print("File exists?", exists)
if exists:
    img = Image.open(test_img)
    img.show()
else:
    print("File not found or path is incorrect.")


# 5) Annoy Index Creation (Nearest Neighbor Retrieval)
styles_path = 'content/styles.csv'
styles = pd.read_csv(styles_path, on_bad_lines="skip")
styles['id'] = styles['id'].astype('str')

def match_id(fname):
    """Return the row index from the styles DF or -1 if not found."""
    matches = styles.index[styles.id == fname].values
    if len(matches) > 0:
        return matches[0]
    else:
        return -1

file_index_to_file_name = {}
file_index_to_file_vector = {}
file_index_to_product_id = {}

# Make sure dims matches the model's embedding dimension (often 1000 for R50x3)
dims = 1000
trees = 10000
n_nearest_neighbors = 20

file_path = imgvec_path
allfiles = glob.glob(os.path.join(file_path, '*.npy'))

t = AnnoyIndex(dims, metric='angular')

print("\nBuilding Annoy index...")
for findex, fname in tqdm(enumerate(allfiles), desc="Annoy Add Items"):
    # load embedding from .npy
    file_vector = np.load(fname)
    file_name = os.path.basename(fname).split('.')[0]

    file_index_to_file_name[findex] = file_name
    file_index_to_file_vector[findex] = file_vector

    pid = match_id(file_name)
    file_index_to_product_id[findex] = pid

    t.add_item(findex, file_vector)

# After adding all items, build once
t.build(trees)

ann_index_file = os.path.join(file_path, "indexer.ann")
t.save(ann_index_file)

pickle.dump(file_index_to_file_name, open(os.path.join(file_path, "file_index_to_file_name.p"), "wb"))
pickle.dump(file_index_to_file_vector, open(os.path.join(file_path, "file_index_to_file_vector.p"), "wb"))
pickle.dump(file_index_to_product_id, open(os.path.join(file_path, "file_index_to_product_id.p"), "wb"))

print(f"\nAnnoy index built with {len(allfiles)} items. Saved to:\n  {ann_index_file}")
print("Pickle files saved with name-to-vector and ID mappings.")


# 6) Example Similarity Search

# We'll define a helper to load and preprocess a single image for "module_2".
def load_img_for_module(path):
    """Load a single image (off-disk) into a tf.Tensor for module_2 inference."""
    img_raw = tf.io.read_file(path)
    img_decoded = tf.io.decode_jpeg(img_raw, channels=3)
    img_resized = tf.image.resize_with_pad(img_decoded, 224, 224)
    img_float = tf.image.convert_image_dtype(img_resized, tf.float32)
    # Expand batch dimension: shape [1, 224, 224, 3]
    return tf.expand_dims(img_float, axis=0)


local_test_imgs = ["shoe.jpg", "wristwatch.jpg", "cap.jpg"]
topK = 4

# Loop over each test image
for test_img_path in local_test_imgs:

    input_tensor = load_img_for_module(test_img_path)
    test_vec = np.squeeze(module_2(input_tensor))
    nns = t.get_nns_by_vector(test_vec, n=topK)

    path_dict = {}
    for path in Path(data_dir).rglob('*.jpg'):
        path_dict[path.stem] = path

    plt.figure(figsize=(20, 10))
    for i in range(topK):
        x = file_index_to_file_name[nns[i]]
        neighbor_img_path = path_dict.get(x, None)

        # Title from the CSV row if we have a product ID
        y = file_index_to_product_id[nns[i]]
        if y != -1:
            row = styles.loc[y]
            title = '\n'.join([str(val) for val in row[-5:].values])
        else:
            title = "No matching CSV row"

        plt.subplot(1, topK, i + 1)
        if neighbor_img_path and neighbor_img_path.exists():
            plt.imshow(mpimg.imread(neighbor_img_path))
            plt.title(title)
            plt.axis('off')
        else:
            plt.imshow(np.zeros((224, 224, 3)))  # blank
            plt.title("Not found")
    plt.tight_layout()
    plt.show()
else:
    print(f"Local test image not found: {local_test_imgs}")
