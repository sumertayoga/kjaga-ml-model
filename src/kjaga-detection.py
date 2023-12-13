from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
from detection_helpers import sliding_window
from detection_helpers import image_pyramid
from detection_helpers import decode_predictions
import numpy as np
import argparse
import cv2

# Argumen Parser
# Contoh cara menjalankan kode
# python kjaga-detection.py --image nasi-tempe.jpg --size "(250, 250)"
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-s", "--size", type=str, default="(200, 150)",
	help="ROI size (in pixels)")
args = vars(ap.parse_args())

# Nilai konstanta
WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = eval(args["size"])
INPUT_SIZE = (224, 224)

# Load model yang telah dibuat
print("[INFO] Loading model...")
model = load_model("./modelv1-4.h5", custom_objects={'KerasLayer':hub.KerasLayer})

# Load gambar dan mendapatkan dimensinya
original_image = cv2.imread(args["image"])
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
		print(roi)
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
for (label, prob) in preds:
	if prob >= 0.85:
		labels.add(label)

print(labels)