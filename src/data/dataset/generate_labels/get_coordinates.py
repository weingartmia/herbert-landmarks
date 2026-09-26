import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path



BASE_PATH= Path(__file__).resolve().parent
model_path = str(BASE_PATH/"hand_landmarker.task")

failed_landmark=[]

print(model_path)

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)


def read_image(path):

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img= cv2.resize(img,(224,224))
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:

        image = mp.Image(data=img, image_format=mp.ImageFormat.SRGB)

        result = landmarker.detect(image)

        print(f"result:{result}")

    return result


def get_coordinates(path):
    result= read_image(path)

   
    if result.hand_landmarks:
        landmarks= result.hand_landmarks[0]


        thumb =landmarks[4].x,landmarks[4].y
        index =landmarks[8].x, landmarks[8].y
        palm = landmarks[0].x, landmarks[0].y
        middle= landmarks[12].x, landmarks[12].y

        print(f"thumb landmark: {landmarks[4].x,landmarks[4].y}")
    else:
        failed_landmark.append(path)
        return 0

    return  thumb, index,palm,middle

def show_failed_landmarks():
    print(failed_landmark)
    
    for path in failed_landmark:
        im=cv2.imread(str(path))
        img= cv2.resize(im,(224,224))
        cv2.imshow(str(path),img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    




