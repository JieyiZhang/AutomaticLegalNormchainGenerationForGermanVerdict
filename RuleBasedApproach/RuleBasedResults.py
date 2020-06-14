from RulebasedApproach import get_normchain, preprocess_extraction, get_normsAll, generate_normchain, extr2mat, compute_recall_score_AbbrArtPara, compute_recall_score_AbbrArt, compute_recall_score_Abbr, compute_precision_score_AbbrArtPara, compute_precision_score_AbbrArt, compute_precision_score_Abbr

# total number of labelled files
file_num = 32893
# file count
count = 0

file_names = []
file_recall_scores_AbbrArtPara = []
file_recall_scores_AbbrArt = []
file_recall_scores_Abbr = []
file_precision_scores_AbbrArtPara = []
file_precision_scores_AbbrArt = []
file_precision_scores_Abbr = []
length_score = []

labelledData_path = "/Users/jieyizhang/Desktop/Master_Thesis/labelled_files"

# Constant value
for root, directories, files in os.walk(labelledData_path, topdown=True):
    for xml_filename in files:
        if xml_filename[-4:].lower() == ".xml":
            count = count +1
            print(count/file_num)
            filename = root + '/' + xml_filename
            print(filename)
            doc = parse(filename)
            tree = ET.parse(filename)
            label = get_normchain(doc)
            if len(label)>0:
                label_mat = extr2mat(label)
                try:
                    norms = get_normsAll(tree)
                    generated_chain = generate_normchain(norms)
                except:
                    print("This file has label ", label)
                    print("But something is missing or wrong within the text.")
                    continue
                generate_mat = extr2mat(generated_chain)
                
                recall_score_AbbrArtPara = compute_recall_score_AbbrArtPara(label_mat, generate_mat)
                precision_score_AbbrArtPara = compute_precision_score_AbbrArtPara(label_mat, generate_mat)
                recall_score_AbbrArt = compute_recall_score_AbbrArt(label_mat, generate_mat)
                precision_score_AbbrArt = compute_precision_score_AbbrArt(label_mat, generate_mat)
                recall_score_Abbr = compute_recall_score_Abbr(label_mat, generate_mat)
                precision_score_Abbr = compute_precision_score_Abbr(label_mat, generate_mat)
                
                length_score.append(compare_length(label_mat, generate_mat))
                file_names.append(filename)
                file_recall_scores_AbbrArtPara.append(recall_score_AbbrArtPara)
                file_recall_scores_AbbrArt.append(recall_score_AbbrArt)
                file_recall_scores_Abbr.append(recall_score_Abbr)
                file_precision_scores_AbbrArtPara.append(precision_score_AbbrArtPara)
                file_precision_scores_AbbrArt.append(precision_score_AbbrArt)
                file_precision_scores_Abbr.append(precision_score_Abbr)

print('Number of files that has been evaluated on recall: ' + str(len(file_recall_scores_AbbrArtPara)))

print('Average recall score of Abbr Article Paragraph prediction: ' + str(sum(file_recall_scores_AbbrArtPara)/len(file_recall_scores_AbbrArtPara)))
print('Average recall score of Abbr Article prediction: ' + str(sum(file_recall_scores_AbbrArt)/len(file_recall_scores_AbbrArt)))
print('Average recall score of Abbr prediction: ' + str(sum(file_recall_scores_Abbr)/len(file_recall_scores_Abbr)))

print('Number of files that has been evaluated on precision: ' + str(len(file_precision_scores_AbbrArtPara)))

print('Average precision score of Abbr Article Paragraph prediction: ' + str(sum(file_precision_scores_AbbrArtPara)/len(file_precision_scores_AbbrArtPara)))
print('Average precision score of Abbr Article prediction: ' + str(sum(file_precision_scores_AbbrArt)/len(file_precision_scores_AbbrArt)))
print('Average precision score of Abbr prediction: ' + str(sum(file_precision_scores_Abbr)/len(file_precision_scores_Abbr)))
