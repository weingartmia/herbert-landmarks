from pathlib import Path
import csv

def write(dir,samples):
    with open(dir,"a",newline="") as file:
        writer = csv.writer(file)       
        writer.writerows(samples)

def delete(dir):
    with open(dir,"w", newline="") as file:
        writer= csv.writer(file)
        writer.writerow("")

def read_data(dir):
    with open(dir, "r", newline="") as file:
        reader = csv.reader(file)
        print(reader)


if __name__ == "__main__":

    distance_dir= Path("images/8")
    files=[]
    for file in distance_dir.iterdir():
        print(f"File: {distance_dir.name+"/"+file.name}, distance: {distance_dir.name}")
        files.append([distance_dir.name+"/"+file.name,distance_dir.name])
        
    write("labels/distance.csv",files)