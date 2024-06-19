
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--score_file', type=str, help='score file')
args = parser.parse_args()

score_file = args.score_file
csv_file = "scores"+score_file+"_MM.csv"

def count_high_scores(csv_file, threshold=3):
    total=0
    morph = 0
    bonafide=0
    bonafide_files=[]
    morph_files=[]
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            total+=1
            if float(row['Score']) > threshold:
                morph += 1
                morph_files.append(row['File Name'])
                
            else:
                bonafide+=1
                bonafide_files.append(row['File Name'])
    print(f'Bonafide: {bonafide}')
    print(f'Morph:    {morph}')
    print(f'Total:   {total}')
    print(f'Morphs Percentage: ',(morph/total)*100)
    print(f'Bonafide Percentage: ',(bonafide/total)*100)
    print(f'Morph Files As per MMFUSION:\n',morph_files)
    print(f'Bonafide Files As per MMFUSION:\n',bonafide_files)

count_high_scores(csv_file)

