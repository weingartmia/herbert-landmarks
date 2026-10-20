import cv2
import numpy as np

def show_absolute_coordinates(coordinates,center,resized_thumb,resized_index,img):

    print(coordinates)
    for i,(x,y) in enumerate(coordinates):

        print(f"pixels x,y : {int(x *224), int(y*224)}")
        cv2.circle(img, (int(x *224), int(y*224)), 6, (0, 0, 255), 1)

        if i==0: text="thumb"
        elif i==1: text="index"
        elif i==2: text="palm"
        elif i==3: text="middle"
        
        cv2.putText(img, f"{text}", (int(x*224),int(y*224)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    cv2.circle(img, (int(center[0] *224), int(center[1]*224)), 6, (0, 255, 0), 1)

    cv2.circle(img, (int(resized_index[0] *224), int(resized_index[1] *224)), 6, (255, 0, 0), 1)
    cv2.circle(img, (int(resized_thumb[0] *224), int(resized_thumb[1]*224)), 6, (255, 0, 0), 1)

    



def get_absolute_distance(coordinates) -> np.float32:
    middle = coordinates[3]
    palm = coordinates[2]

    resing = 1/np.abs((middle - palm))

    resized_thumb = coordinates[0][0] * resing[0],coordinates[0][1] * resing[1]
    resized_index = coordinates[1][0] * resing[0],coordinates[1][1] * resing[1]


    # distance = np.sqrt(( resized_index[0]-resized_thumb[0])**2 + (resized_thumb[1]- resized_index[1])**2)
    distance=  np.sqrt((coordinates[0][0]-coordinates[1][0]) **2 + (coordinates[0][1]-coordinates[1][1]) **2)
    print(distance * 224)
    return distance,resized_thumb,resized_index


def get_rotation(coordinates):
    thumb= coordinates[0]
    index= coordinates[1]

    palm= coordinates[2]

    tip_center=  np.mean([[thumb[0], thumb[1]],[index[0], index[1]]],axis=0)
    diff = tip_center -palm
    
    angle= np.degrees(np.arctan2(diff[1],diff[0]))
    print(angle)
    return angle, tip_center









