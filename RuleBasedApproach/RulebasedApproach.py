from xml.dom.minidom import parse, parseString
import xml.etree.ElementTree as ET
import re
import numpy as np
import os

def get_normchain(doc):
# Return all legal norms in the norm chain in node "NORM" in each verdict documentation
    norm_chain = []
    for node in doc.getElementsByTagName('NORM'):
        for t in node.childNodes:
            print(t.nodeValue)
            norm_chain.append(t.nodeValue)
    return norm_chain

def preprocess_extraction(ext):
# Clean the extracted norms and append them into a list in the one of following forms:
#     1. Law code abbreviation
#     2. Law code abbreviation + Article number
#     3. Law code abbreviation + Article number + Abs. paragraph number
# The form of the appended norm depends on the granularity of the extracted norm
    processed=[]
    for e in ext:
        e = e[0]
        if ',' in e:
            # print(e)
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


def get_normsAll(tree):
# Return all norms annotated in relevant sections (except NORM and NORMENKETTE) in 
# each verdict documentation
    root = tree.getroot()
    
    sections = ['SCHLAGWORT', 'ANMERKUNG',
                'UNTERSCHLAGWORT', 'ENTSCHEIDUNGSFORMEL',
                'LEITSATZ', 'TATBESTAND', 'NACHGEHEND', 'GRUENDE']
    
    section_weights = {'SCHLAGWORT':1, 'ANMERKUNG':1,
            'UNTERSCHLAGWORT':1, 'ENTSCHEIDUNGSFORMEL':1,
            'LEITSATZ':2, 'TATBESTAND':1, 'NACHGEHEND':1, 'GRUENDE':1}
    
    # output dictionary
    norms = {}
    
    # pattern for norm extraction
    pattern = '§\s(\d+[a-zA-ZäüöÄÜÖß]*[\sFall|Abs.|Satz|Nr.\s\d+[,\s\d+]*]*\s([a-z]+ )*[A-Z][a-zA-ZäüöÄÜÖß]+)'    
    
    for section in sections:
        for node in root.iter(section):
            for i in node.iter():
                # Match from plain text
                if i.tag != 'VERWEIS-GS':
                    string = i.text
                    try:
                        ext = re.findall(pattern, string)
                        processed = preprocess_extraction(ext)
                        for e in processed:
                            book = re.findall('(\w+)',e)[-1]
                            art = re.findall('(\d+)',e)[0]
                            norm = book + ' ' + art
                            if norm not in norms:
                                norms[norm] = {}
                            art_abs = re.findall('(\d+\w+\sAbs.\s\d+)',e)
                            if art_abs==[]:
                                if norm in norms[norm]:
                                    norms[norm][norm] += section_weights[section]
                                else:
                                    norms[norm][norm] = section_weights[section]
                            else:
                                for s in art_abs:
                                    name = book + ' '  + s
                                    if name in norms[norm]:
                                        norms[norm][name] += section_weights[section]
                                    else:
                                        norms[norm][name] = section_weights[section]                                
                    except:
                        pass
                else:
                # Extract from node
                    attr = i.attrib
                    string = i.text
                    book = re.findall('(\w+)',string)[-1]
                    norm = attr['PUBKUERZEL']+' ' + attr['NORM']
                    if norm not in norms:
                        norms[norm] = {}
                    art_abs = re.findall('(\d+\w+\sAbs.\s\d+)',string)                     
                    if art_abs==[]:
                        if norm in norms[norm]:
                            norms[norm][norm] += section_weights[section] 
                        else:
                            norms[norm][norm] = section_weights[section] 
                    else:
                        for s in art_abs:
                            name = attr['PUBKUERZEL'] + ' '  + s
                            if name in norms[norm]:
                                norms[norm][name] += section_weights[section] 
                            else:
                                norms[norm][name] = section_weights[section] 
                if i.tail is not None:
                    string = i.tail
                    try:
                        ext = re.findall(pattern, string)
                        processed=preprocess_extraction(ext)
                        for e in processed:
                            book = re.findall('(\w+)',e)[-1]
                            art = re.findall('(\d+)',e)[0]
                            norm = book + ' ' + art
                            if norm not in norms:
                                norms[norm] = {}
                            art_abs = re.findall('(\d+\w+\sAbs.\s\d+)',e)
                            if art_abs==[]:
                                if norm in norms[norm]:
                                    norms[norm][norm] += section_weights[section]
                                else:
                                    norms[norm][norm] = section_weights[section]
                            else:
                                for s in art_abs:
                                    name = book + ' '  + s
                                    if name in norms[norm]:
                                        norms[norm][name] += section_weights[section]
                                    else:
                                        norms[norm][name] = section_weights[section]                              
                    except:
                        pass
                    
    return norms


def generate_normchain(norms, threshold = 2):
# Select legal norms for norm chain output from the candidates
# The details are described in the thesis
    chain = []
    
    for (key, value) in norms.items():
        if len(value) == 1:
            for (key_2, value_2) in value.items():
                if value_2 > threshold:
                    key_2 = key_2.replace(u'\xa0', u' ')
                    chain.append(key_2)
        else:
            append_key = True
            candidates = {}
            for (key_2, value_2) in value.items():
                if (value_2 > threshold) & (key_2 != key):
                    key_2 = key_2.replace(u'\xa0', u' ')
                    candidates[key_2] = value_2
                    append_key = False
            if append_key:
                key = key.replace(u'\xa0', u' ')
                chain.append(key)
            else:
                name = max(candidates, key=candidates.get)
                name = name.replace(u'\xa0', u' ')
                chain.append(name)
    return chain

def extr2mat(arr):
# This function extract the [info, value] pairs from the extracted norms. 
# E.g. norm string 'ZPO 139 Abs. 1' and 'ZPO 522' for a docuumentation is extracted as 
# a 3-D matrix [[[ZPO, Abs],[139, 1]],[['ZPO'],[522]]]
    extraction = []
    for n in arr:
        s = re.findall('(\D[a-zA-ZäüöÄÜÖß.]+\d*[a-zA-ZäüöÄÜÖß.]*[-]*[a-zA-ZäüöÄÜÖß.]*[\\]*[a-zA-ZäüöÄÜÖß.]*)\s',n)
        s = [x.strip(' ') for x in s]
        s = [x for x in s if len(x) > 1]       
        remove = ['Art.', 'a.F.', 'i.V.m.', 'Anh.', 'zu', 'BW', 'i.d.F.','und', 'n.F.',
                  'BEL', 'THA', 'BRA', 'FRA', 'CHE', 'Baugewerbe', 'Bund', 'Abschn.',
                  'BT-K', 'NRW', 'Berlin', 'iVm.', 'NW', 'XII', 'II', 'Schl.-H.', 'MV',
                  'Bln', 'RhPf', 'RP', '-H','Tageszeitungen',  'Flugsicherung', 'ATZ', 'DTAG',
                  'DTTS', 'DRK', 'SH', 'UmBw', 'BKK', '(Zusatzabkommen',  'Einzelhandel','Sicherheit',
                 'Bayern',  'dänischer', 'ATZ-TgRV', 'Bühne', 'BT-BBiG', 'Brandenburg', 'AWO','PV',
                 '/Lackierer', 'Th','Art','nF','Baden-Württemberg','(jetzt', 'III','u.', 'analog','VI',
                 'des','StVergAbG','JStG', 'AG','BeitrRLUmsG', 'd.', 'Rheinland-Pfalz','InfoV','Anhang','f.',
                 'Hamburger', 'StSenkG','DDR', 'BR','IV','Abkommen', 'mehr', 'Bbg','a.2']
        if len(s)>0:
            s = [x for x in s if x not in remove]
            if len(s)>1:
                if s[0]==s[1]:
                    s.pop(1)
            d = re.findall('\s(\d+\w*)',n)
            if (len(s)>0) & (len(d)>0):
                print(d)
                extraction.append([s,d])
    return extraction


# Calculate the matching score
# The scores are computed based on three different granularities:
#     1. Law code abbreviation
#     2. Law code abbreviation + Article number
#     3. Law code abbreviation + Article number + Abs. paragraph number

def compute_recall_score_AbbrArtPara(norm_mat, generate_mat):
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
        art_flag = False
        abs_flag = None
        for i in generate_mat:
            if i[0][0]==book:
                book_flag = True
                if i[1][0]==art:
                    art_flag = True
                    for j in range(1,len(e[0])):
                        if e[0][j]=='Abs.':
                            absch = e[1][j]
                            abs_flag = False
                    for j in range(1,len(i[0])):
                        if i[0][j]=='Abs.':
                            absch_2 = i[1][j]
                    try:
                        if absch == absch_2:
                            abs_flag = True
                    except:
                        pass
        flags.append([book_flag, art_flag, abs_flag])
    flags=np.array(flags)
    if len(np.argwhere(flags==True))+len(np.argwhere(flags==False))>0:
        score = len(np.argwhere(flags==True))/(len(np.argwhere(flags==True))+len(np.argwhere(flags==False)))
    else:
        score = -1
    return score


def compute_recall_score_AbbrArt(norm_mat, generate_mat):
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
        art_flag = False
        for i in generate_mat:
            if i[0][0]==book:
                book_flag = True
                if i[1][0]==art:
                    art_flag = True
        flags.append([book_flag, art_flag])
    flags=np.array(flags)
    if len(np.argwhere(flags==True))+len(np.argwhere(flags==False))>0:
        score = len(np.argwhere(flags==True))/(len(np.argwhere(flags==True))+len(np.argwhere(flags==False)))
    else:
        score = -1
    return score


# Calculate the matching score
def compute_recall_score_Abbr(norm_mat, generate_mat):
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
def compute_precision_score_AbbrArtPara(norm_mat, generate_mat):
# Computed a matching score based on the number of TRUE and FALSE matching flags
    flags =[]
    for e in generate_mat:
        print(e, 'is one legal norm in the generated chain.')
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
        art_flag = False
        abs_flag = None
        for i in norm_mat:
            if len(i[0])>len(i[1])+1:
                continue
            if (len(i[1])==1) & (len(i[0])>1):
                continue
            if i[0][0]==book:
                book_flag = True
                if i[1][0]==art:
                    art_flag = True
                    for j in range(1,len(e[0])):
                        if e[0][j]=='Abs.':
                            absch = e[1][j]
                            abs_flag = False
                    for j in range(1,len(i[0])):
                        if i[0][j]=='Abs.':
                            print(i)
                            absch_2 = i[1][j]
                    try:
                        if absch == absch_2:
                            abs_flag = True
                    except:
                        pass
        flags.append([book_flag, art_flag, abs_flag])
    flags=np.array(flags)
    if len(np.argwhere(flags==True))+len(np.argwhere(flags==False))>0:
        score = len(np.argwhere(flags==True))/(len(np.argwhere(flags==True))+len(np.argwhere(flags==False)))
    else:
        score = -1
    return score


# Calculate the matching score
def compute_precision_score_AbbrArt(norm_mat, generate_mat):
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
        art_flag = False
        for i in norm_mat:
            if i[0][0]==book:
                book_flag = True
                if i[1][0]==art:
                    art_flag = True
        flags.append([book_flag, art_flag])
    flags=np.array(flags)
    if len(np.argwhere(flags==True))+len(np.argwhere(flags==False))>0:
        score = len(np.argwhere(flags==True))/(len(np.argwhere(flags==True))+len(np.argwhere(flags==False)))
    else:
        score = -1
    return score


# Calculate the matching score
def compute_precision_score_Abbr(norm_mat, generate_mat):
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

def compare_length(norm_mat, generate_mat):
# Compare the length of label normchain and generated normchain
    if len(norm_mat)>0:
        score = abs(len(norm_mat)-len(generate_mat))/len(norm_mat)
    else:
        score = -1
    return score
