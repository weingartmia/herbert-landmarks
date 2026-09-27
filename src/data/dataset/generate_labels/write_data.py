from pathlib import Path
import csv
from get_coordinates import get_coordinates,show_failed_landmarks

import random as r

def write(dir,samples,file_path):
    with open(dir,"a",newline="") as file:
        writer = csv.writer(file)       
        writer.writerow([file_path, str(samples)])

def delete(dir):
    with open(dir,"w", newline="") as file:
        writer= csv.writer(file)
        writer.writerow("")

def read_data(dir):
    with open(dir, "r", newline="") as file:
        reader = csv.reader(file)
        print(reader)



def write_distances():
    distance_dir= Path("../images/landmarks")
    labels=[]
    for file in distance_dir.iterdir():

        cordinates = get_coordinates(file)

        if cordinates==0: continue
           
        print(f"final coordinates: {cordinates}")
        print(f"File: {distance_dir.name+"/"+file.name}, distance: {distance_dir.name}")
        labels.append([distance_dir.name+"/"+file.name,distance_dir.name])
        
        write("../labels/distance.csv",cordinates,distance_dir.name+"/"+file.name )
    show_failed_landmarks()


def write_binary_data():
    no_hands_dir= Path("../images/no-hand")
    hands_dir= Path("../images/landmarks")


    count = list(hands_dir.iterdir())

    for file in no_hands_dir.iterdir():

        write("../labels/hand.csv", 0,no_hands_dir.name+"/"+file.name )
        index = r.randrange(0,len(count) -1)
        hands_file= count[index]
        count.pop(index)
        write("../labels/hand.csv",1, hands_dir.name +"/"+ hands_file.name)

if __name__ == "__main__":

    write_binary_data()