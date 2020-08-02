from xml.dom.minidom import parse, parseString
import xml.etree.ElementTree as ET
import re
import numpy as np
import os
import pandas as pd
from xml.dom.minidom import parse, parseString
import xml.etree.ElementTree as ET
import re
from bs4 import BeautifulSoup
import urllib.request as http
import string

import numpy as np
from sklearn.model_selection import train_test_split

all_tags = {}
path = 'https://www.gesetze-im-internet.de/Teilliste_'
for i in range(26):
    try:
        file = path + string.ascii_uppercase[i] + '.html'
        content = http.urlopen(file).read()
        parsed_html = BeautifulSoup(content)
        for tag in parsed_html.find_all('abbr'):
            if tag.text!='PDF':
                abbr = tag.text.lstrip().rstrip()
                all_tags[abbr]=[]
    except:
        continue
        
file_names = []

count = 0
filepath = "/Users/jieyizhang/Desktop/Master_Thesis/labelled_files"
# Constant value
for root, directories, files in os.walk(filepath, topdown=True):
    for xml_filename in files:
        if xml_filename[-4:].lower() == ".xml":
            count = count +1
            print(count/32705)
            filename = root + '/' + xml_filename
            file_names.append(filename)
            
def preprocess_extraction(ext):
    processed=[]
    for e in ext:
        e = e[0]
        if ',' in e:
            print(e)
            words = e.split()
            indices = [i-1 for i, x in enumerate(words) if x == "Abs."]
            if len(indices)==1:
                words = re.split(' |, ', e)
                for j in range(2,len(words)-1):
                    l=' '.join(list( words[i] for i in [0, 1, j, -1] ))
                    processed.append(l)
            if len(indices)>1:
                for i in range(len(indices)):
                    if i!=len(indices)-1:
                        string = ' '.join(words[indices[i]:indices[i+1]])
                        string = string[:-1]+' '+words[-1]
                        processed.append(string)
                    else:
                        processed.append(' '.join(words[indices[i]:len(words)]))
        else:
            processed.append(e)
    return processed

def get_normsAll(files_list):
# Return all norms annotated in relevant sections (except NORM and NORMENKETTE) in 
# each verdict documentation
    for file in files_list:
        tree = ET.parse(file)
        root = tree.getroot()

        sections = ['SCHLAGWORT', 'ANMERKUNG',
                    'UNTERSCHLAGWORT', 'ENTSCHEIDUNGSFORMEL',
                    'LEITSATZ', 'TATBESTAND', 'NACHGEHEND', 'GRUENDE']

        section_weights = {'SCHLAGWORT':1, 'ANMERKUNG':1,
                'UNTERSCHLAGWORT':1, 'ENTSCHEIDUNGSFORMEL':1,
                'LEITSATZ':2, 'TATBESTAND':1, 'NACHGEHEND':1, 'GRUENDE':1}


        pattern = '§\s(\d+[a-zA-ZäüöÄÜÖß]*[\sFall|Abs.|Satz|Nr.\s\d+[,\s\d+]*]*\s([a-z]+ )*[A-Z][a-zA-ZäüöÄÜÖß]+)'    
    #     pattern = '§\s(\d+[a-zA-ZäüöÄÜÖß]*[\sAbs.|Satz|Nr.\s\d+[,\s\d+]*]*\s[a-zA-ZäüöÄÜÖß]+)'
        tags = all_tags
        for section in sections:
            for node in root.iter(section):
                for i in node.iter():
                    if i.tag != 'VERWEIS-GS':
                        string = i.text
                        try:
                            ext = re.findall(pattern, string)
                            processed = preprocess_extraction(ext)
                            for e in processed:
                                norm = re.findall('(\w+)',e)[-1]
                                if norm in tags:
                                    tags[norm].append(section_weights[section])

                        except:
                            pass
                    else:
                        attr = i.attrib
                        string = i.text
                        book = re.findall('(\w+)',string)[-1]
                        norm = attr['PUBKUERZEL']
                        if norm in tags:
                            tags[norm].append(section_weights[section])
                    if i.tail is not None:
                        string = i.tail
                        try:
                            ext = re.findall(pattern, string)
                            processed=preprocess_extraction(ext)
                            for e in processed:
                                norm = re.findall('(\w+)',e)[-1]
                                if norm in tags:
                                    tags[norm].append(section_weights[section])                       
                        except:
                            pass
                    
    return tags

tags = get_normsAll(file_names)
input_tags = [k for k, v in tags.items() if len(v) >= 1]

tags_index = {}
for i in range(len(input_tags)):
    tags_index[input_tags[i]] = i

# get_normchain(doc)
def get_normchain(doc):
# Return all legal norms in the norm chain in node "NORM" in each verdict documentation
    norm_chain = []
    for node in doc.getElementsByTagName('NORM'):
        for t in node.childNodes:
            string = t.nodeValue
            norm = string.split()[0]
            if norm in all_tags_set:
                norm_chain.append(norm)
    norm_chain = list(set(norm_chain))
    return norm_chain

all_tags_set = set()
path = 'https://www.gesetze-im-internet.de/Teilliste_'
for i in range(26):
    try:
        file = path + string.ascii_uppercase[i] + '.html'
        content = http.urlopen(file).read()
        parsed_html = BeautifulSoup(content)
        for tag in parsed_html.find_all('abbr'):
            if tag.text!='PDF':
                abbr = tag.text.lstrip().rstrip()
                all_tags_set.add(abbr) 
    except:
        continue

count = 0

all_labels = []

# Constant value
for root, directories, files in os.walk("/Users/jieyizhang/Desktop/Master_Thesis/labelled_files", topdown=True):
    for xml_filename in files:
        if xml_filename[-4:].lower() == ".xml":
            count = count +1
            print(count/32893)
            filename = root + '/' + xml_filename
            print(filename)
            doc = parse(filename)
            tree = ET.parse(filename)
            label = get_normchain(doc)
            all_labels = all_labels + label
            
all_labels = set(all_labels)


tags_set = all_tags_set.intersection(all_labels)

def get_normchains(filenames):
# Return all legal norms in the norm chain in node "NORM" in each verdict documentation
    norm_chains = []
    for file in filenames:
        doc = parse(file)
        norm_chain = []
        for node in doc.getElementsByTagName('NORM'):
            for t in node.childNodes:
                string = t.nodeValue
                norm = string.split()[0]
                abbr = string.split()[0]
                if len(string.split())>1:
                    if string.split()[1][0].isdigit():
                        norm = norm + ' ' + string.split()[1]
                if abbr in tags_set:
                    norm_chain.append(norm)
        norm_chain = list(set(norm_chain))
        norm_chains.append(norm_chain)
    return norm_chains

norm_chains = get_normchains(file_names)

def get_normsVec(file_names, sections):
# Return all norms annotated in relevant sections (except NORM and NORMENKETTE) in 
# each verdict documentation
    norm_vecs = []
    
    for file in file_names:
        norm_vec = [0] * 642
        
        tree = ET.parse(file)
        root = tree.getroot()

        
        section_weights = {'SCHLAGWORT':1, 'ANMERKUNG':1,
                'UNTERSCHLAGWORT':1, 'ENTSCHEIDUNGSFORMEL':1,
                'LEITSATZ':1, 'TATBESTAND':1, 'NACHGEHEND':1, 'GRUENDE':1}


        pattern = '§\s(\d+[a-zA-ZäüöÄÜÖß]*[\sFall|Abs.|Satz|Nr.\s\d+[,\s\d+]*]*\s([a-z]+ )*[A-Z][a-zA-ZäüöÄÜÖß]+)'

        for section in sections:
            for node in root.iter(section):
                for i in node.iter():
                    if i.tag != 'VERWEIS-GS':
                        string = i.text
                        try:
                            ext = re.findall(pattern, string)
                            processed = preprocess_extraction(ext)
                            for e in processed:
                                norm = re.findall('(\w+)',e)[-1]
                                try:
                                    norm_vec[tags_index[norm]] += section_weights[section]
                                except:
                                    print(norm)
                                    pass

                        except:
                            pass
                    else:
                        attr = i.attrib
                        string = i.text
                        book = re.findall('(\w+)',string)[-1]
                        norm = attr['PUBKUERZEL']
                        try:
                            norm_vec[tags_index[norm]] += section_weights[section]
                        except:
                            print(norm)
                            pass
                    if i.tail is not None:
                        string = i.tail
                        try:
                            ext = re.findall(pattern, string)
                            processed=preprocess_extraction(ext)
                            for e in processed:
                                norm = re.findall('(\w+)',e)[-1]
                                try:
                                    norm_vec[tags_index[norm]] += section_weights[section]     
                                except:
                                    print(norm)
                                    pass
                        except:
                            pass
        norm_vecs.append(norm_vec)
                    
    return norm_vecs

sections_2 = ['TATBESTAND']

sections_3 = ['GRUENDE']

norm_vecs_2 = get_normsVec(file_names, sections_2)
norm_vecs_3 = get_normsVec(file_names, sections_3)

dictionary = {'file_names': file_names, 'norm_vecs_2': norm_vecs_2,
              'norm_vecs_3': norm_vecs_3,'norm_chains': norm_chains}  
    
data_df = pd.DataFrame(dictionary) 

data_df['norm_vecs'] =  data_df['norm_vecs_2'] + data_df['norm_vecs_3']

data_df = data_df.drop(columns=[  'norm_vecs_2', 'norm_vecs_3'])

data_df.head()

from sklearn.preprocessing import MultiLabelBinarizer

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(data_df.norm_chains)

# transform target variable
y = multilabel_binarizer.transform(data_df.norm_chains)

xtrain, xval, ytrain, yval = train_test_split(np.array(data_df.norm_vecs.to_list()), y, test_size=0.3, random_state=9)

clf = MLPClassifier(hidden_layer_sizes=(2000,))
# fit model on train data
clf.fit(xtrain, ytrain)

y_pred = clf.predict(xval)

f1_score(yval, y_pred,average='weighted')

print('Accuracy score: ')
print(metrics.accuracy_score(yval, y_pred))
print('Precision score: ')
print(metrics.precision_score(yval, y_pred,average='weighted'))
print('Recall score: ')
print(metrics.recall_score(yval, y_pred,average='weighted'))

y_pred_label = multilabel_binarizer.inverse_transform(y_pred)
y_val_label = multilabel_binarizer.inverse_transform(yval)


diction = {'y_val_label': y_val_label, 'y_pred_label': y_pred_label}  
assess_df = pd.DataFrame(diction)
assess_df.y_val_label = assess_df.y_val_label.apply(np.array)
assess_df.y_pred_label = assess_df.y_pred_label.apply(np.array)
assess_df['precision_score'] = assess_df.apply(lambda x: compute_precision_score(x['y_val_label'], 
                                                                                 x['y_pred_label']), axis=1)

assess_df['recall_score'] = assess_df.apply(lambda x: compute_recall_score(x['y_val_label'], 
                                                                                 x['y_pred_label']), axis=1)

assess_df = assess_df[assess_df.precision_score!=-1]
print('Precision score: ')
assess_df.precision_score.mean()
print('Recall score: ')
assess_df.recall_score.mean()
