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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics

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
filepath = '/Users/jieyizhang/Desktop/Master_Thesis/labelled_files'

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
                    'UNTERSCHLAGWORT', 'ENTSCHEIDUNGSFORMEL','TATBESTAND', 'NACHGEHEND', 'GRUENDE']

        section_weights = {'SCHLAGWORT':1, 'ANMERKUNG':1,
                'UNTERSCHLAGWORT':1, 'ENTSCHEIDUNGSFORMEL':1,'TATBESTAND':1, 'NACHGEHEND':1, 'GRUENDE':2}


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



# Calculate the matching score
def compute_score(norm_mat, generate_mat):
# Computed a matching score based on the number of TRUE and FALSE matching flags
    flags =[]
    for e in norm_mat:
        print(e, 'is one legal norm in the label.')
        if len(e[0])>len(e[1])+1:
            # There are some norms have name consist of several words
            # Skip calculating scores for these ones
            continue
        if (len(e[1])==1) & (len(e[0])>1):
            # There are some norms in annotation with Absatz number but without artikle number
            continue
        book = e[0][0]
        art = e[1][0]
        book_flag = False
        for i in generate_mat:
            if i[0][0]==book:
                book_flag = True
        flags.append([book_flag])
    flags=np.array(flags)
    if len(np.argwhere(flags==True))+len(np.argwhere(flags==False))>0:
        score = len(np.argwhere(flags==True))/(len(np.argwhere(flags==True))+len(np.argwhere(flags==False)))
    else:
        score = -1
    return score

# Calculate the matching score
def compute_precision_score(norm_mat, generate_mat):
# Computed a matching score based on the number of TRUE and FALSE matching flags
    flags =[]
    for e in generate_mat:
        print(e, 'is one legal norm in the label.')
        if len(e[0])>len(e[1])+1:
            # There are some norms have name consist of several words
            # Skip calculating scores for these ones
            continue
        if (len(e[1])==1) & (len(e[0])>1):
            # There are some norms in annotation with Absatz number but without artikle number
            continue
        book = e[0][0]
        art = e[1][0]
        book_flag = False
        for i in norm_mat:
            if i[0][0]==book:
                book_flag = True
        flags.append([book_flag])
    flags=np.array(flags)
    if len(np.argwhere(flags==True))+len(np.argwhere(flags==False))>0:
        score = len(np.argwhere(flags==True))/(len(np.argwhere(flags==True))+len(np.argwhere(flags==False)))
    else:
        score = -1
    return score


tags = get_normsAll(file_names)
input_tags = [k for k, v in tags.items() if len(v) >= 1]
top_tags = [k for k, v in tags.items() if len(v) >= 300]
normal_tags = list(set(input_tags).difference(set(top_tags)))

path = 'https://www.gesetze-im-internet.de/Teilliste_'
Top_Abbr_URL = {}
for i in range(26):
    try:
        file = path + string.ascii_uppercase[i] + '.html'
        content = http.urlopen(file).read()
        parsed_html = BeautifulSoup(content)
        for tag in parsed_html.find_all('a', href=True):
                abbr = tag.text.lstrip().rstrip()
                if abbr in top_tags:
                    if abbr in Top_Abbr_URL:
                        Top_Abbr_URL[abbr].append('https://www.gesetze-im-internet.de' + tag['href'][1:])
                    else:
                        Top_Abbr_URL[abbr] = ['https://www.gesetze-im-internet.de' + tag['href'][1:]]
    except:
        continue
        
norms_withNum = []
for abbr in Top_Abbr_URL:
    for link in Top_Abbr_URL[abbr]:
        content_test = http.urlopen(link).read()
        parsed_html = BeautifulSoup(content_test)
        
        for tag in parsed_html.find_all('a', href=True):
            if tag['href'].endswith('.html'):
                if tag['href'].startswith('___'):
                    string = tag['href'].replace('.html','').replace('___','')
                    if string[0].isdigit():
                        if 'bis' in string:
                            nums = string.split("_bis_")
                            try:
                                if string[-1].isdigit():
                                    nums = list(map(int, nums))
                                    for num in range(nums[0],nums[1]+1):
                                        norm = abbr + ' ' + str(num)
                                        norms_withNum.append(norm)
                                else:
                                    start = nums[0][-1]
                                    end = nums[1][-1]
                                    num = nums[0][:-1]
                                    for c in range(ord(start), ord(end)+1):
                                        norm = abbr + ' ' + num + chr(c)
                                        norms_withNum.append(norm)
                            except:
                                print(string)
                                nums = string.split("_bis_")
                                for num in nums:
                                    norm = abbr + ' ' + num
                                    norms_withNum.append(norm)
                        elif 'und' in string:
                            nums = string.split("_und_")
                            for num in nums:
                                norm = abbr + ' ' + num
                                norms_withNum.append(norm)
                else:
                    string = tag['href'].replace('.html','').replace('__','')
                    if string[0].isdigit():
                        norm = abbr + ' ' + string
                        norms_withNum.append(norm)
                        
tags_wo_norms = norms_withNum + input_tags
tags_index = {}
for i in range(len(tags_wo_norms)):
    tags_index[tags_wo_norms[i]] = i
    
def get_normchain(doc):
# Return all legal norms in the norm chain in node "NORM" in each verdict documentation
    norm_chain = []
    for node in doc.getElementsByTagName('NORM'):
        for t in node.childNodes:
            string = t.nodeValue
            norm = string.split()[0]
            if len(string.split())>1 & string.split()[1][0].isdigit():
                norm = norm + ' ' + string.split()[1]
            if norm in tags_wo_norms:
                norm_chain.append(norm)
    norm_chain = list(set(norm_chain))
    return norm_chain


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
                if len(string.split())>1:
                    if string.split()[1][0].isdigit():
                        norm = norm + ' ' + string.split()[1]
                if norm in tags_wo_norms:
                    norm_chain.append(norm)
        norm_chain = list(set(norm_chain))
        norm_chains.append(norm_chain)
    return norm_chains


def get_normsVec(file_names):
# Return all norms annotated in relevant sections (except NORM and NORMENKETTE) in 
# each verdict documentation
    norm_vecs = []
    
    for file in file_names:
        norm_vec = [0] * 22123
        
        tree = ET.parse(file)
        root = tree.getroot()

        sections = ['SCHLAGWORT', 'ANMERKUNG',
                    'UNTERSCHLAGWORT', 'ENTSCHEIDUNGSFORMEL', 'TATBESTAND', 'NACHGEHEND', 'GRUENDE']

        section_weights = {'SCHLAGWORT':1, 'ANMERKUNG':1,
                'UNTERSCHLAGWORT':1, 'ENTSCHEIDUNGSFORMEL':1,'TATBESTAND':1, 'NACHGEHEND':1, 'GRUENDE':2}


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
                                if len(re.findall('(\d+[a-z]*)',e))>0:
                                    art = re.findall('(\d+[a-z]*)',e)[0]
                                    norm = norm + " " + art
                                try:
                                    norm_vec[tags_index[norm]] += section_weights[section]
                                except:
                                    print(norm)
                                    pass

                        except:
                            pass
                    else:
                        attr = i.attrib
                        if 'NORM' in attr:
                            # Check attribute NORM
                            norm = attr['PUBKUERZEL']+' ' + attr['NORM']
                        else:
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
                                if len(re.findall('(\d+[a-z]*)',e))>0:
                                    art = re.findall('(\d+[a-z]*)',e)[0]
                                    norm = norm + " " + art
                                try:
                                    norm_vec[tags_index[norm]] += section_weights[section]     
                                except:
                                    print(norm)
                                    pass
                        except:
                            pass
        norm_vecs.append(norm_vec)
                    
    return norm_vecs


def compute_precision_score(label_array, generate_array):
    
#     label_array = np.array(label_tuple)
#     generate_array = np.array(generate_tuple)
    
    matches = 0
    total = 0
    
    for label in label_array: 
        for pred in generate_array:
            if pred == label:
                matches = matches + 1
        total = total + 1
    
    if total>0:
        score = matches/total
    else:
        score = -1.0

    return score

def compute_recall_score(label_array, generate_array):
    
#     label_array = np.array(label_tuple)
#     generate_array = np.array(generate_tuple)
    
    matches = 0
    total = 0
    
    for pred in generate_array: 
        for label in label_array:
            if pred == label:
                matches = matches + 1
        total = total + 1
    
    if total>0:
        score = matches/total
    else:
        score = 0.0

    return score

norm_chains = get_normchains(file_names)

norm_vecs = get_normsVec(file_names)
# dictionary of lists  
dictionary = {'file_names': file_names, 'norm_vecs': norm_vecs, 'norm_chains': norm_chains}  
    
data_df = pd.DataFrame(dictionary) 
# data_df.to_pickle('normsinput_MLP.pkl')

data_df['len'] = data_df['norm_chains'].apply(len)
data_df = data_df[data_df['len']!=0]

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(data_df.norm_chains)

# transform target variable
y = multilabel_binarizer.transform(data_df.norm_chains)

xtrain, xval, ytrain, yval = train_test_split(np.array(data_df.norm_vecs.to_list()), y, test_size=0.2, random_state=9)

clf = MLPClassifier(hidden_layer_sizes=(3000,))
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
