from pathlib import Path
import csv
from get_coordinates import get_coordinates,show_failed_landmarks

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



if __name__ == "__main__":

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