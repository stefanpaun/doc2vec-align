#%%
from massalign.core import *
from nltk.corpus import stopwords
import string
from time import time
import gensim

file_name1 = 'berkeley1709_orig_pars.txt'
file_name2 = 'berkeley1709_simp_pars.txt'

#Get files to align:
path =  '/home/stefan/projects/nlp_backup/texts/post/'

file1 = path + file_name1
file2 = path + file_name2
stop_words = stopwords.words('english') + list(string.punctuation)


hard_treshold = 0.5
soft_threshold = 0.2
certain_threshold = 0.9

hard_treshold_s = 0.5
soft_threshold_s = 0.25
certain_threshold_s = 0.95

vector_size = 300
window = 15
min_count = 3
epochs = 150
infer_epochs = 150


#Get paragraph aligner:
m = MASSAligner()
model = D2VModel(vector_size=vector_size, window_size = window, min_count = min_count, epochs   = epochs, infer_epochs = infer_epochs, input_files=[file1, file2], stop_list_file =stop_words)
paragraph_aligner = VicinityDrivenParagraphAlignerAlternate(similarity_model=model, certain_threshold=certain_threshold, hard_threshold=hard_treshold, soft_threshold=soft_threshold, slack=0.001)
sentence_aligner = VicinityDrivenSentenceAlignerAlternate(similarity_model=model, certain_threshold=certain_threshold_s, hard_threshold=hard_treshold_s, soft_threshold=soft_threshold_s, slack = 0.01)



p1s = m.getParagraphsFromDocument(file1)
p2s = m.getParagraphsFromDocument(file2)


t = time()
alignments_par, aligned_paragraphs = m.getParagraphAlignments(p1s, p2s, paragraph_aligner)



sentence_alignments = []
i=0
for a in aligned_paragraphs:
    p1 = a[0]
    p2 = a[1]
    alignments, aligned_sentences = m.getSentenceAlignments(p1, p2, sentence_aligner)
    sentence_alignments.append((alignments, aligned_sentences))

par_tuple_list = zip(alignments_par, sentence_alignments)
print('Time to align: {} mins'.format(round((time() - t) / 60, 2)))

#%%

ground_sent = dict()
with open("/home/stefan/projects/nlp_backup/groundtruth_sentences.txt", "r") as groundtruth:
        for line in groundtruth:
                if line != '\r\n':
                        par_id = re.findall(r"(?<=P ).*(?=\r)", line)
                        if par_id:
                                ids_left_pre = re.findall(r".+(?=\-)", line)
                                ids_left_str = re.findall(r"\d+", ids_left_pre[0])
                                ids_right_pre= re.findall(r"(?<=\-).*(?=\r)",line)
                                ids_right_str = re.findall(r"\d+",ids_right_pre[0])

                                ids_left = []
                                ids_right = []

                                for element in ids_left_str:
                                        ids_left.append(int(element))

                                for element in ids_right_str:
                                        ids_right.append(int(element))

                                last_par = (tuple(ids_left), tuple(ids_right))
                                ground_sent[last_par] = []
                        else:
                                s_left = re.findall(r".+(?=\-)", line)
                                s_right = re.findall(r"(?<=\-).+(?=\r)", line)

                                s_left_digits = re.findall(r"\d+", s_left[0])
                                s_right_digits = re.findall(r"\d+", s_right[0])

                                sent_left = [] 
                                sent_right = [] 

                                for element in s_left_digits:
                                        sent_left.append(int(element))
                                
                                for element in s_right_digits:
                                        sent_right.append(int(element))

                                ground_sent[last_par].append((sent_left, sent_right))

results = dict()
for alignment in par_tuple_list:
        paragraphs_alignment = alignment[0]
        sentence_alignments = alignment[1][0]
        sentences = alignment[1][1]
        par_key = (tuple(paragraphs_alignment[0]), tuple(paragraphs_alignment[1]))
        results[par_key] = []
        for i, element in enumerate(sentence_alignments):
                if(element):
                        original_sentences = element[0]
                        simple_sentences = element[1]
                        results[par_key].append((original_sentences, simple_sentences))

#%%

total_paragraph_correct_alignments = 0
total_paragraphs = len(results.keys())
total_correct_paragraphs = 0
total_sentences = 0
total_correct_sentences_alig = 0
partial_correct_sentences_alig = 0
wrong = 0
for key in results.keys():
        total_sentences += len(results[key])
        if(key in ground_sent.keys()):
                total_paragraph_correct_alignments+=1
                for result in results[key]:
                        ok = True
                        for ground in ground_sent[key]:
                                left_g, right_g = ground
                                left_r, right_r = result
                                if(ground==result):
                                        total_correct_sentences_alig+=1
                                        ok = False
                                        break
                                if(set(left_r)==(set(left_g))):
                                        ok = False
                                        if(len(set(right_r).intersection(set(right_g))) != 0):
                                                partial_correct_sentences_alig+=1
                                                break
                                        else:
                                                wrong+=1
                                                break
                                elif (set(right_r)==(set(right_g))):
                                        ok = False
                                        if(len(set(left_r).intersection(set(left_g))) != 0):
                                                partial_correct_sentences_alig+=1  
                                                break   
                                        else:
                                                wrong+=1
                                                break
                                elif (len(set(left_r).intersection(set(left_g))) != 0 and len(set(right_r).intersection(set(right_g))) !=0 ):
                                        partial_correct_sentences_alig+=1
                                        ok = False
                                        break
                        if ok:
                                wrong+= len(result)
        else:
                wrong+= len(results[key])        



total_sent_alignemnts = 0 
for key in ground_sent.keys():
        total_sent_alignemnts+=len(ground_sent[key])


#sys.stdout = open(file_name1[:-14] + "_stats.txt", "w+")

# print("Number of paragraphs (original, simplified), detected alignments", len(p1s), len(p2s), total_paragraphs, float(total_paragraphs)/len(p1s))
# print("Number of sentences (original, simplified), detected alignments", sum([len(element) for element in p1s]), sum([len(element) for element in p2s]), total_sentences, float(total_sentences)/sum([len(element) for element in p1s]))

print("Par stats:", len(ground_sent.keys()), total_paragraphs, total_paragraph_correct_alignments)
print("Sentence stats:", total_sent_alignemnts, total_sentences, total_correct_sentences_alig, partial_correct_sentences_alig, wrong)


precision_par = float(total_paragraph_correct_alignments)/total_paragraphs
recall_par = float(total_paragraph_correct_alignments)/len(ground_sent.keys())
f_par =  2 * precision_par * recall_par / (precision_par + recall_par)

precision_sent = float(total_correct_sentences_alig) / total_sentences
recall_sent = float(total_correct_sentences_alig) / total_sent_alignemnts
f_sent = 2 * precision_sent * recall_sent / (precision_sent + precision_par)

print("Paragraph (P,R,F1)", precision_par, recall_par, f_par)
print("Sentence (P,R,F1)", precision_sent, recall_sent, f_sent)

#sys.stdout.close()

#%%
name_par = file_name1[:-14]+"_par_alig.txt"
name_sent = file_name1[:-14]+"_sent_alig.txt"
aligned_file_par = open(name_par, "w+")
aligned_file_sent = open(name_sent, "w+")


for entry in aligned_paragraphs:
        original_par = ''
        for item in entry[0]:
                original_par += str(item) + ' '
        simple_par = ''
        for item in entry[1]:
                simple_par += str(item) + ' '
        original_par = original_par.strip()
        simple_par = simple_par.strip()
        aligned_file_par.write(original_par)
        aligned_file_par.write('\n')
        aligned_file_par.write(simple_par)
        aligned_file_par.write('\n\n')
aligned_file_par.close()
     
sims = []
for element in par_tuple_list:
        s_aligs = element[1][1]
        for alignment in s_aligs:
                vec1 = str(alignment[0])
                vec2 = str(alignment[1])
                aligned_file_sent.write(vec1)
                aligned_file_sent.write('\n')
                aligned_file_sent.write(vec2)
                aligned_file_sent.write('\n\n')
                sim = model.getTextSimilarity(alignment[0], alignment[1])
                sims.append(sim)
        aligned_file_sent.write('\n')
aligned_file_sent.close()       
                
#%%

import numpy as np 
import matplotlib.pyplot as plt

 
a = np.hstack(sims)
_ = plt.hist(a, bins='auto')  # arguments are passed to np.histogram
plt.title("Sentence alignment similarity score distribution")
plt.show()
plt.savefig(file_name1[:-14]+'_eval.png')