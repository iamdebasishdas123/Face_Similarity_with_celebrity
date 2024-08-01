import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from mtcnn import MTCNN

# Load feature list and filenames
feature_list = np.array(pickle.load(open(r'C:\Users\Debasish Das\Documents\01_ML Project\Deep learning Projects\Bollywood Celebratity\embedding_vgg16.pkl', 'rb')))
filenames = pickle.load(open(r'C:\Users\Debasish Das\Documents\01_ML Project\Deep learning Projects\Bollywood Celebratity\filenames.pkl', 'rb'))

# Load VGG16 model
model = VGG16(
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='avg'
)

# Initialize MTCNN for face detection
detector = MTCNN()

# Load and detect faces in the sample image
sample_image_path = r"C:\Users\Debasish Das\Pictures\friend\20231021_181125.jpggit "
sample_image = cv2.imread(sample_image_path)
results = detector.detect_faces(sample_image)
x, y, width, height = results[0]['box']
face = sample_image[y:y + height, x:x + width]

# Display the detected face
cv2.imshow('Detected Face', face)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract features from the detected face
image = Image.fromarray(face)
image = image.resize((224, 224))
face_array = np.asarray(image)
face_array = face_array.astype('float32')
expanded_img = np.expand_dims(face_array, axis=0)
preprocessed_img = preprocess_input(expanded_img)
result = model.predict(preprocessed_img).flatten()

print("Extracted Feature Vector:", result)
print("Feature Vector Shape:", result.shape)

# Calculate cosine similarity
similarity = []
for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]

# Debugging output
print("Index Position:", index_pos)
print("Filename:", filenames[index_pos])

# Load and display the similar image
temp_img = cv2.imread(filenames[index_pos])
if temp_img is None:
    print("Error loading image. Check the file path.")
else:
    cv2.imshow('Similar Image', temp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
