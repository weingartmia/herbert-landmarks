from pathlib import Path
import csv

def write(samples):
    with open("labels.csv","w",newline="") as file:
        writer = csv.writer(file)       
        writer.writerows(samples)

def read_data():
    with open("labels.csv", "r", newline="") as file:
        reader = csv.reader(file)
        print(reader)

if __name__ == "__main__":

    distance_dir= Path("images/0")
    files=[]
    for file in distance_dir.iterdir():
        print(f"File: {distance_dir.name+"/"+file.name}, distance: {distance_dir.name}")
        files.append([distance_dir.name+"/"+file.name,distance_dir.name])
        
    write(files)