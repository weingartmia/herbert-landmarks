import cv2

def show_absolute_coordinates(coordinates,img):

   
    # middle = coordinates[3]
    # palm = coordinates[2]
    print(coordinates)
    for i,(x,y) in enumerate(coordinates):

        print(f"pixels x,y : {int(x *224), int(y*224)}")
        cv2.circle(img, (int(x *224), int(y*224)), 6, (0, 0, 255), 1)

        if i==0: text="thumb"
        elif i==1: text="index"
        elif i==2: text="palm"
        elif i==3: text="middle"
        
        cv2.putText(img, f"{text}", (int(x*224),int(y*224)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)



def get_absolute_distance(coordinates, img):
    middle = coordinates[3]
    palm = coordinates[2]

