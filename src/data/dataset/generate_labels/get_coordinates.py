import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import numpy as np


BASE_PATH= Path(__file__).resolve().parent
model_path = str(BASE_PATH/"hand_landmarker.task")

failed_landmark=[]

wrong_landmarks=[]

print(model_path)

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.1,
    min_hand_presence_confidence=0.1,
    min_tracking_confidence=0.1,        
)

landmarker = vision.HandLandmarker.create_from_options(options)

# def read_image(path):

#     img = cv2.imread(path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img= cv2.resize(img,(224,224))
 
#     result = detect(img)

#     print(f"result:{result}")

#     return result


# def return_best_coordinates(padded,H,W,p,rotations=(0, 1, 2, 3)):
#     Hp, Wp = padded.shape[:2]
#     best = None
#     for k in rotations:
#         rot = np.ascontiguousarray(np.rot90(padded, k))
#         # rgb = np.ascontiguousarray(cv2.cvtColor(rot, cv2.COLOR_BGR2RGB))
#         res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rot))

#         if not res.hand_landmarks:
#             continue

#         lm = np.array([[l.x, l.y] for l in res.hand_landmarks[0]])
#         for _ in range(k):                                # undo the rotation
#             lm = np.stack([1 - lm[:, 1], lm[:, 0]], 1)
#             pts = np.stack([(lm[:, 0] * Wp - p) / W, (lm[:, 1] * Hp - p) / H], 1)
#             if auto_valid(pts):
#                 score = res.handedness[0][0].score
#                 print(f"score: {score}")
#                 if best is None or score > best[0]:
#                     best = (score, pts)
#     return best


# def detect(img_rgb, pad_fracs=(0.0, 0.5, 1.0)):
#     H, W = img_rgb.shape[:2]
    
#     for pf in pad_fracs:
#         p = int(pf * max(H, W))
#         padded = cv2.copyMakeBorder(img_rgb, p, p, p, p, cv2.BORDER_REFLECT) if p else img_rgb

#         best = return_best_coordinates(padded,H,W,p)
        
#     return best
# 
# 
#  
def read_image(path):
    img = cv2.imread(path)
    return detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),path)        

def return_best_coordinates(padded,path, H, W, p, rotations=(0, 1, 2, 3)):
    Hp, Wp = padded.shape[:2]
    best = None
    for k in rotations:
        rot = np.ascontiguousarray(np.rot90(padded, k))
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rot))
        if not res.hand_landmarks:
            print("continue")
            continue

        lm = np.array([[l.x, l.y] for l in res.hand_landmarks[0]], dtype=np.float32)
        for _ in range(k):                                # undo rotation
            lm = np.stack([1 - lm[:, 1], lm[:, 0]], 1)
        pts = np.stack([(lm[:, 0] * Wp - p) / W, (lm[:, 1] * Hp - p) / H], 1,dtype=np.float32)
      
       

        if auto_valid(pts):          
            print("/////////////////////////Good landmark ///////////////////////")                   
            score = res.handedness[0][0].score
            if best is None or score > best[0]:
                best = (score, pts)
        else:
            print("///////////////////////// Wrong landmark ///////////////////////")
            wrong_landmarks.append([path,[pts[4], pts[8],get_angle(pts),pts[12]] ])

      

    return best

def detect(img_rgb,path, pad_fracs=(0.0, 0.5, 1.0)):
    H, W = img_rgb.shape[:2]
    best = None
    for pf in pad_fracs:
        p = int(pf * max(H, W))
        padded = cv2.copyMakeBorder(img_rgb, p, p, p, p, cv2.BORDER_REFLECT) if p else img_rgb
        cand = return_best_coordinates(padded,path, H, W, p)
        if cand is not None and (best is None or cand[0] > best[0]):
            best = cand

    print(f"best : {best}")
    return best


def get_angle(landmarks):
    palm_center = np.mean([[landmarks[1][0], landmarks[1][1]],[landmarks[5][0], landmarks[5][1]]],axis=0)
    

    finger_tip_center=  np.mean([[landmarks[4][0], landmarks[4][1]],[landmarks[8][0], landmarks[8][1]]],axis=0)
   

    diff= np.mean([finger_tip_center,palm_center], axis=0)
    # diff = finger_tip_center -palm_center

    

    # angle= np.degrees(np.arctan2(diff[0],diff[1]))
    # print(f"**************************{angle}")

    return diff



def get_coordinates(path):
    result= read_image(path)
      
    if result is not None:
        print("result is not none")
        result=result[1]
     
        thumb =result[4][0],result[4][1]
        index =result[8][0], result[8][1]
        palm = get_angle(result)
        middle= result[12][0], result[12][1]
   
  
        return  thumb, index,palm,middle
 

    else: return 0


def auto_valid(pts21,min_hand_size=0.08):                    
    tips = np.asarray(pts21[[4, 8, 12]], dtype=np.float32)          # [3, 2]
    knuckles = np.asarray(pts21[[5, 9, 13, 17]], dtype=np.float32)  # [4, 2]

    if np.any(tips < 0.0) or np.any(tips > 1.0):
        return False

    if np.any(knuckles < -0.05) or np.any(knuckles > 1.05):
        print(f"knuckles out of frame: {knuckles.ravel()}")
        return False

    hand_size = np.linalg.norm(tips.mean(axis=0) - knuckles.mean(axis=0))
    if hand_size < min_hand_size:
        print(f"hand too small / bunched: {hand_size:.3f}")
        return False

    if np.any(tips[:,1]) < tips[0][1]:
        print(f" thumb is higher than tips {tips[1,:][1]} {tips[0][1]}")
        return False
    return True

def show_failed_landmarks():
    print(failed_landmark)
    
    for path in failed_landmark:
        im=cv2.imread(str(path))
        img= cv2.resize(im,(224,224))
        cv2.imshow(str(path),img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_wrong_landmarks():

    for path,coordinate in wrong_landmarks:
        im=cv2.imread(str(path))
        img= cv2.resize(im,(224,224))
        for i,(x,y) in enumerate(coordinate):
        
                print(f"pixels x,y : {int(x *224), int(y*224)}")
                cv2.circle(img, (int(x *224), int(y*224)), 6, (0, 0, 255), 1)
        
                if i==0: text="thumb"
                elif i==1: text="index"
                elif i==2: text="palm"
                elif i==3: text="middle"
                
                cv2.putText(img, f"{text}", (int(x*224),int(y*224)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        cv2.imshow(str(path),img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    




