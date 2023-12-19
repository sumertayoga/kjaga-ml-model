import time
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
from detection_helpers import sliding_window
from detection_helpers import image_pyramid
from detection_helpers import decode_predictions
import numpy as np
import argparse
import cv2
import imutils

# Argumen Parser
# Contoh cara menjalankan kode
# python kjaga-detection.py --image nasi-tempe.jpg
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# Nilai konstanta
WIDTH = 900
PYR_SCALE = 1.5
WIN_STEP = 24
ROI_SIZE = (250, 250)
INPUT_SIZE = (224, 224)

# Load model yang telah dibuat
print("[INFO] Loading model...")
model = load_model("./modelv1-12.h5")

# Load gambar dan mendapatkan dimensinya
original_image = cv2.imread(args["image"])
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image = imutils.resize(original_image, width=WIDTH)
(H, W) = original_image.shape[:2]

# Melakukan inisialisasi image pyramid generator
pyramid = image_pyramid(original_image, scale=PYR_SCALE, minSize=ROI_SIZE)

# Menyimpan nilai ROI pada list
# Menyimpan lokasi ROI pada original image
rois = []
locations = []

# Melakukan looping pada setiap gambar di pyramid
# Mengaplikasikan sliding windows
for image in pyramid:
	scale = W / float(image.shape[1])
	for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
		x = int(x * scale)
		y = int(y * scale)
		w = int(ROI_SIZE[0] * scale)
		h = int(ROI_SIZE[1] * scale)
		roi = cv2.resize(roiOrig, INPUT_SIZE)
		roi = img_to_array(roi)
		roi = preprocess_input(roi)
		roi = roi/255.
		rois.append(roi)
		locations.append((x, y, x + w, y + h))

rois = np.array(rois, dtype="float32")
# Melakukan prediksi dan mapping label
# sesuai dengan nilai prediksi
preds = model.predict(rois)
preds = decode_predictions(preds)


# Membuat set dan menambahkan nilai label
# dengan nilai probabilitas > 85%
labels = set()
for (i, p) in enumerate(preds):
	(label, prob) = p
	if prob >= 0.9:
		labels.add(label)

print(labels)
