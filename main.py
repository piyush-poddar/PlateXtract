import google.generativeai as genai
import os
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import cv2
import numpy as np
import util
from streamlit import secrets
import gdown

#os.environ['api_key']
#API_KEY = os.environ['api_key']
genai.configure(api_key=secrets["GEMINI_API_KEY"])

model_cfg_path = os.path.join('.', 'model', 'cfg', 'darknet-yolov3.cfg')
model_weights_path = os.path.join('.', 'model', 'weights', 'model.weights')
class_names_path = os.path.join('.', 'model', 'class.names')

input_dir = os.path.join('.', 'images')

# You can yourself download the weights from the link below and place it in the weights folder
# Otherwise, the weights will be downloaded automatically but first time it will take some time
if not os.path.exists(model_weights_path):
    model_weights_url = "https://drive.google.com/uc?id=1Qlcv7vcyWn9UsKsjqHat4V_CuVh5Lggs"
    gdown.download(model_weights_url, model_weights_path)
else:
    print("model.weights already exists")


def extract_number_plate(img_path):

    with open(class_names_path, 'r') as f:
        class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
        f.close()

    # load model
    net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

    # load image

    img = cv2.imread(img_path)

    H, W, _ = img.shape

    # convert image
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

    # get detections
    net.setInput(blob)

    detections = util.get_outputs(net)

    # bboxes, class_ids, confidences
    bboxes = []
    class_ids = []
    scores = []

    for detection in detections:
        # [x1, x2, x3, x4, x5, x6, ..., x85]
        bbox = detection[:4]

        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

        bbox_confidence = detection[4]

        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)

    # apply nms
    bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)
    #print(bboxes)
    # plot
    license_plate_gray = None
    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox
        #print("hi",xc,yc,w,h)

        img = cv2.rectangle(img,
                            (int(xc - (w / 2)), int(yc - (h / 2))),
                            (int(xc + (w / 2)), int(yc + (h / 2))),
                            (0, 255, 0),
                            10)

        license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :]

        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    
    return bboxes, license_plate_gray
#_, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY)

# Prepare and upload image
def prep_image(image_path):
    sample_file = genai.upload_file(path=image_path, display_name='Diagram')
    #print(f"{sample_file.display_name}:{sample_file.uri}")
    
    file = genai.get_file(name=sample_file.name)
    #print(f"{file.display_name}:{sample_file.uri}")
    return sample_file

def delete_files(img_path):
    # if os.path.exists(img_path):
    #     os.remove(img_path)

    processed_img_path = f".{img_path.split('.')[1]}_processed.png"

    if os.path.exists(processed_img_path):
        os.remove(processed_img_path)

    d = genai.list_files()
    for i in d:
        #print(i.name)
        genai.delete_file(i.name)

def extract_text_from_image(img_path):
    bboxes, license_plate_gray = extract_number_plate(img_path)

    if len(bboxes)>0:
        processed_img_path = f".{img_path.split('.')[1]}_processed.png"
        cv2.imwrite(processed_img_path, license_plate_gray)
        #image = ImageOps.exif_transpose(processed_img_path)

        sample_file = prep_image(processed_img_path)
    else:
        #image = ImageOps.exif_transpose(img_path)

        sample_file = prep_image(img_path)

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    prompt = "Extract the 'Indian' vehicle number plate from the image and isolate the numerical and alphabetic characters. Remove any extraneous text, symbols, or logos that may be present on the number plate, and give the output in a single line."
    response = model.generate_content([sample_file, prompt],
	safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH:HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT:HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:HarmBlockThreshold.BLOCK_NONE
	})
    text = response.text

    if text:
        print(img_path.split('\\')[-1])
        delete_files(img_path)
        return text
    else:
        delete_files(img_path)
        return "Failed to extract the vehicle number plate"