from shutil import copyfile
import os
from xml.dom.minidom import parse, parseString
import xml.etree.ElementTree as ET

# Count the number of verdict documentations that have labels (Normchain)
labels = 0
# Count the total number of verdict documentations
total = 0

# Filepath which includes all verdict documents as xml files
filepath = "/Users/jieyizhang/Desktop/Master_Thesis/Rechtsprechung_komplett_20161110"

# Filefolders for saving the files with/without lables
target_labelled = '/Users/jieyizhang/Desktop/Master_Thesis/labelled_files/'
target_unlabelled = '/Users/jieyizhang/Desktop/Master_Thesis/unlabelled_files/'

for root, directories, files in os.walk(filepath, topdown=True):
    for xml_filename in files:
        if xml_filename[-4:].lower() == ".xml":
            total = total + 1
            filename = root + '/' + xml_filename
            print(filename)
            doc = parse(filename)
            # In some cases, only one of the following element has value
            if (len(doc.getElementsByTagName('NORM'))>0|len(doc.getElementsByTagName('NORMENKETTE'))>0):
                labels = labels + 1
                copyfile(filename, target_labelled + xml_filename)
            else:
                copyfile(filename, target_unlabelled + xml_filename)
    for directory in directories:
        continue
