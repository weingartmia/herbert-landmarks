

import csv


def write_values(rows):
    with open("bad_values.csv","w", newline="") as file:
        writer= csv.writer(file)
        writer.writerows(rows)

