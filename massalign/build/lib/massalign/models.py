from abc import ABCMeta, abstractmethod
import numpy as np
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from massalign.util import FileReader
from time import time
from scipy import spatial
from nltk.corpus import stopwords
from nltk import word_tokenize
import string

class SimilarityModel:

	__metaclass__ = ABCMeta

	@abstractmethod
	def getSimilarityMapBetweenParagraphsOfDocuments(self, ps1, ps2):
		pass

	@abstractmethod
	def getSimilarityMapBetweenSentencesOfParagraphs(self, p1, p2):
		pass
		
class TFIDFModel(SimilarityModel):
	"""
	Implements a typical gensim TFIDF model for MASSAlign.
			
	* *Parameters*:
		* **input_files**: A set of file paths containing text from which to extract TFIDF weight values.
		* **stop_list_file**: A path to a file containing a list of stop-words.
	"""

	def __init__(self, input_files=[], stop_list_file=None):
		reader = FileReader(stop_list_file)
		self.stoplist = set([line.strip() for line in reader.getRawText().split('\n')])
		self.tfidf, self.dictionary = self.getTFIDFmodel(input_files)
		
	def getTFIDFmodel(self, input_files=[]):
		"""
		Trains a gensim TFIDF model.
				
		* *Parameters*:
			* **input_files**: A set of file paths containing text from which to extract TFIDF weight values.
		* *Output*:
			* **tfidf**: A trained gensim models.TfidfModel instance.
			* **dictionary**: A trained gensim.corpora.Dictionary instance.
		"""
		#Create text sentence set for training:
		sentences = []
		for file in input_files:
			reader = FileReader(file, self.stoplist)
			sentences.extend(reader.getSplitSentences())
				
		#Train TFIDF model:
		dictionary = gensim.corpora.Dictionary(sentences)
		corpus = [dictionary.doc2bow(sentence) for sentence in sentences]
		tfidf = gensim.models.TfidfModel(corpus)
		
		#Return tfidf model:
		return tfidf, dictionary
	
	def getSimilarityMapBetweenSentencesOfParagraphs(self, p1, p2):
		"""
		Produces a matrix containing similarity scores between all sentences in a pair of paragraphs.
				
		* *Parameters*:
			* **p1**: A source paragraph. A paragraph is a list of sentences.
			* **p2**: A target paragraph. A paragraph is a list of sentences.
		* *Output*:
			* **sentence_similarities**: A matrix containing a similarity score between all possible pairs of sentences in the union of p1 and p2. The matrix's height and width are equal and equivalent to the number of distinct sentences present in the union of p1 and p2.
			* **sentence_indexes**: A map connecting each sentence to its numerical index in the sentence_similarities matrix.
		"""
		#Get distinct sentences from paragraphs:
		sentences = list(self.getSentencesFromParagraph(p1).union(self.getSentencesFromParagraph(p2)))
		
		#Get TFIDF model controllers:
		sentence_similarities, sentence_indexes = self.getTFIDFControllers(sentences)
		
		#Return similarity matrix:
		return sentence_similarities, sentence_indexes
		
	def getSimilarityMapBetweenParagraphsOfDocuments(self, p1s=[], p2s=[]):
		"""
		Produces a matrix containing similarity scores between all paragraphs in a pair of paragraph lists.
				
		* *Parameters*:
			* **p1s**: A list of source paragraphs. Each paragraph is a list of sentences.
			* **p2s**: A list of target paragraphs. Each paragraph is a list of sentences.
		* *Output*:
			* **paragraph_similarities**: A matrix containing a similarity score between all possible pairs of paragraphs in the union of p1 and p2. The matrix's height and width are equal and equivalent to the number of distinct paragraphs present in the union of p1s and p2s.
		"""
		#Get distinct sentences from paragraph sets:
		sentences = list(self.getSentencesFromParagraphs(p1s).union(self.getSentencesFromParagraphs(p2s)))

		#Get TFIDF model controllers:
		sentence_similarities, sentence_indexes = self.getTFIDFControllers(sentences)
	
		#Calculate paragraph similarities:
		paragraph_similarities = list(np.zeros((len(p1s), len(p2s))))
		for i, p1 in enumerate(p1s):
			for j, p2 in enumerate(p2s):
				values = []
				for sent1 in p1:
					for sent2 in p2:
						values.append(sentence_similarities[sentence_indexes[sent1]][sentence_indexes[sent2]])
				paragraph_similarities[i][j] = np.max(values)
				
		#Return similarity matrix:
		return paragraph_similarities
				
	def getTFIDFControllers(self, sentences):
		"""
		Produces TFIDF similarity scores between all possible pairs of sentences in a list.
				
		* *Parameters*:
			* **sentences**: A list of sentences.
		* *Output*:
			* **sentence_similarities**: A matrix containing a similarity score between all possible sentence pairs in the input sentence list. The matrix's height and width are equal and equivalent to the number of distinct sentences in the input sentence list.
			* **sentence_indexes**: A map connecting each sentence to its numerical index in the sentence_similarities matrix.
		"""
		#Create data structures for similarity calculation:
		sent_indexes = {}
		for i, s in enumerate(sentences):
			sent_indexes[s] = i
			
		#Get similarity querying framework:
		texts = [[word for word in sentence.split(' ') if word not in self.stoplist] for sentence in sentences]
		corpus = [self.dictionary.doc2bow(text) for text in texts]
		index = gensim.similarities.MatrixSimilarity(self.tfidf[corpus])
		
		#Create similarity matrix:
		sentence_similarities = []
		for j in range(0, len(sentences)):
			sims = index[self.tfidf[corpus[j]]]
			sentence_similarities.append(sims)
		
		#Return controllers:
		return sentence_similarities, sent_indexes
	
	def getTextSimilarity(self, buffer1, buffer2):
		"""
		Calculates the TFIDF similarity between two buffers containing text.
				
		* *Parameters*:
			* **buffer1**: A source buffer containing a block of text.
			* **buffer2**: A target buffer containing a block of text.
		* *Output*:
			* **similarity**: The TFIDF similarity between the two buffers of text.
		"""
		#Get bag-of-words vectors:
		vec1 = self.dictionary.doc2bow(buffer1.split())
		vec2 = self.dictionary.doc2bow(buffer2.split())
		corpus = [vec1, vec2]
		
		#Get similarity matrix from bag-of-words model:
		index = gensim.similarities.MatrixSimilarity(self.tfidf[corpus])
		
		#Return the similarity between the vectors:
		sims = index[self.tfidf[vec1]]
		similarity = sims[1]
		return similarity
	
	def getSentencesFromParagraphs(self, ps):
		"""
		Extracts a set containing all unique sentences in a list of paragraphs.
				
		* *Parameters*:
			* **ps**: A list of paragraphs. A paragraph is a list of sentences.
		* *Output*:
			* **sentences**: The set containing all unique sentences in the input paragraph list.
		"""
		#Get all distinct sentences from a set of paragraphs:
		sentences = set([])
		for p in ps:
			psents = self.getSentencesFromParagraph(p)
			sentences.update(psents)
		
		#Return sentences found:
		return sentences
	
	def getSentencesFromParagraph(self, p):
		"""
		Extracts a set containing all unique sentences in a paragraph.
				
		* *Parameters*:
			* **p**: A paragraph. A paragraph is a list of sentences.
		* *Output*:
			* **sentences**: The set containing all unique sentences in the input paragraph.
		"""
		#Return all distinct sentences from a paragraph:
		sentences = set(p)
		return sentences
			

class W2VModel(SimilarityModel):
	"""
	Implements a typical gensim Word2Vec model for MASSAlign.
			
	* *Parameters*:
		* **input_files**: A set of file paths containing text from which to extract TFIDF weight values.
		* **stop_list_file**: A path to a file containing a list of stop-words.
	"""

	def __init__(self, input_files=[], stop_list_file=None):
		self.stoplist = stopwords.words('english') + list(string.punctuation)
		self.w2v, self.dictionary = self.getW2Vmodel(input_files)
		self.index2word_set = set(self.w2v.wv.index2word)
		
	def getW2Vmodel(self, input_files=[]):
		"""
		Trains a gensim Word2Vec model.
				
		* *Parameters*:
			* **input_files**: A set of file paths containing text from which to extract TFIDF weight values.
		* *Output*:
			* **tfidf**: A trained gensim models.TfidfModel instance.
			* **dictionary**: A trained gensim.corpora.Dictionary instance.
		"""
		#Create text sentence set for training:
		sentences = []
		for file in input_files:
			reader = FileReader(file, self.stoplist)
			sentences.extend(reader.getSplitSentences())
		
		#Train Word2Vec model:
		
		dictionary = gensim.corpora.Dictionary(sentences)
		#corpus = [dictionary.doc2bow(sentence) for sentence in sentences]
		w2v = gensim.models.Word2Vec()
		w2v.build_vocab(sentences, progress_per=10000)
		w2v.train(sentences, total_examples=w2v.corpus_count, epochs=30, report_delay=1)
		w2v.init_sims(replace=True)
		#Return tfidf model:
		return w2v, dictionary
	
	def getSimilarityMapBetweenSentencesOfParagraphs(self, p1, p2):
		"""
		Produces a matrix containing similarity scores between all sentences in a pair of paragraphs.
				
		* *Parameters*:
			* **p1**: A source paragraph. A paragraph is a list of sentences.
			* **p2**: A target paragraph. A paragraph is a list of sentences.
		* *Output*:
			* **sentence_similarities**: A matrix containing a similarity score between all possible pairs of sentences in the union of p1 and p2. The matrix's height and width are equal and equivalent to the number of distinct sentences present in the union of p1 and p2.
			* **sentence_indexes**: A map connecting each sentence to its numerical index in the sentence_similarities matrix.
		"""
		#Get distinct sentences from paragraphs:
		sentences = list(self.getSentencesFromParagraph(p1).union(self.getSentencesFromParagraph(p2)))
		
		#Get TFIDF model controllers:
		sentence_similarities, sentence_indexes = self.getW2VControllers(sentences)
		
		#Return similarity matrix:
		return sentence_similarities, sentence_indexes
		
	def getSimilarityMapBetweenParagraphsOfDocuments(self, p1s=[], p2s=[]):
		"""
		Produces a matrix containing similarity scores between all paragraphs in a pair of paragraph lists.
				
		* *Parameters*:
			* **p1s**: A list of source paragraphs. Each paragraph is a list of sentences.
			* **p2s**: A list of target paragraphs. Each paragraph is a list of sentences.
		* *Output*:
			* **paragraph_similarities**: A matrix containing a similarity score between all possible pairs of paragraphs in the union of p1 and p2. The matrix's height and width are equal and equivalent to the number of distinct paragraphs present in the union of p1s and p2s.
		"""
		#Get distinct sentences from paragraph sets:
		sentences = list(self.getSentencesFromParagraphs(p1s).union(self.getSentencesFromParagraphs(p2s)))

		#Get TFIDF model controllers:
		sentence_similarities, sentence_indexes = self.getW2VControllers(sentences)
	
		#Calculate paragraph similarities:
		paragraph_similarities = list(np.zeros((len(p1s), len(p2s))))
		for i, p1 in enumerate(p1s):
			for j, p2 in enumerate(p2s):
				values = []
				for sent1 in p1:
					for sent2 in p2:
						values.append(sentence_similarities[sentence_indexes[sent1]][sentence_indexes[sent2]])
				paragraph_similarities[i][j] = np.max(values)
				
		#Return similarity matrix:
		return paragraph_similarities

	def avg_sentence_vector(self, words, num_features):
		featureVec = np.zeros((num_features,), dtype="float32")
		nwords = 0

		for word in words:
			if word in self.index2word_set:
				nwords = nwords+1
				featureVec = np.add(featureVec, self.w2v[word])

		if nwords>0:
			featureVec = np.divide(featureVec, nwords)
		return featureVec

	def getW2VControllers(self, sentences):
		"""
		Produces TFIDF similarity scores between all possible pairs of sentences in a list.
				
		* *Parameters*:
			* **sentences**: A list of sentences.
		* *Output*:
			* **sentence_similarities**: A matrix containing a similarity score between all possible sentence pairs in the input sentence list. The matrix's height and width are equal and equivalent to the number of distinct sentences in the input sentence list.
			* **sentence_indexes**: A map connecting each sentence to its numerical index in the sentence_similarities matrix.
		"""
		#Create data structures for similarity calculation:
		sent_indexes = {}
		for i, s in enumerate(sentences):
			sent_indexes[s] = i
			
		#Get similarity querying framework:
		texts = [[word for word in list(gensim.utils.tokenize(sentence)) if word not in self.stoplist] for sentence in sentences]
		#corpus = [self.dictionary.doc2bow(text) for text in texts]
		#index = gensim.similarities.MatrixSimilarity(self.w2v[corpus])
		
		#Create similarity matrix:
		sentence_similarities = []
		for j in range(0, len(sentences)):
			sims = []
			s1_afv = self.avg_sentence_vector(words=texts[j], num_features=100)
			for k in range(0, len(sentences)):
				s2_afv = self.avg_sentence_vector(words=texts[k], num_features=100)
				sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
				sims.append(sim)
			sentence_similarities.append(sims)
		
		#Return controllers:
		return sentence_similarities, sent_indexes
	
	def getTextSimilarity(self, buffer1, buffer2):
		"""
		Calculates the TFIDF similarity between two buffers containing text.
				
		* *Parameters*:
			* **buffer1**: A source buffer containing a block of text.
			* **buffer2**: A target buffer containing a block of text.
		* *Output*:
			* **similarity**: The TFIDF similarity between the two buffers of text.
		"""
		#Get bag-of-words vectors:
		vec1 = buffer1.split()
		vec2 = buffer2.split()
		

		s1_afv = self.avg_sentence_vector(words=vec1, num_features=100)
		s2_afv = self.avg_sentence_vector(words=vec2, num_features=100)

		#Return the similarity between the vectors:
		similarity = 1 - spatial.distance.cosine(s1_afv, s2_afv)
		return similarity
	
	def getSentencesFromParagraphs(self, ps):
		"""
		Extracts a set containing all unique sentences in a list of paragraphs.
				
		* *Parameters*:
			* **ps**: A list of paragraphs. A paragraph is a list of sentences.
		* *Output*:
			* **sentences**: The set containing all unique sentences in the input paragraph list.
		"""
		#Get all distinct sentences from a set of paragraphs:
		sentences = set([])
		for p in ps:
			psents = self.getSentencesFromParagraph(p)
			sentences.update(psents)
		
		#Return sentences found:
		return sentences
	
	def getSentencesFromParagraph(self, p):
		"""
		Extracts a set containing all unique sentences in a paragraph.
				
		* *Parameters*:
			* **p**: A paragraph. A paragraph is a list of sentences.
		* *Output*:
			* **sentences**: The set containing all unique sentences in the input paragraph.
		"""
		#Return all distinct sentences from a paragraph:
		sentences = set(p)
		return sentences		


class D2VModel(SimilarityModel):
	"""
	Implements a typical gensim Word2Vec model for MASSAlign.
			
	* *Parameters*:
		* **input_files**: A set of file paths containing text from which to extract TFIDF weight values.
		* **stop_list_file**: A path to a file containing a list of stop-words.
	"""

	def __init__(self, vector_size=100, window_size=10, min_count=2, epochs=200, infer_epochs=200, input_files=[], stop_list_file=None, dm=0):
		#reader = FileReader(stop_list_file)
		self.stoplist = stopwords.words('english') + list(string.punctuation)
		self.vector_size = vector_size
		self.window_size = window_size
		self.min_count = min_count
		self.epochs = epochs
		if(dm in [0, 1]):
			self.dm = dm
		else:
			self.dm = 0
		self.infer_epochs = infer_epochs
		self.d2v, self.paragraphs = self.getD2Vmodel(input_files)

	def getD2Vmodel(self, input_files=[]):
		"""
		Trains a gensim Word2Vec model.
				
		* *Parameters*:
			* **input_files**: A set of file paths containing text from which to extract TFIDF weight values.
		* *Output*:
			* **tfidf**: A trained gensim models.TfidfModel instance.
			* **dictionary**: A trained gensim.corpora.Dictionary instance.
		"""
		#Create text sentence set for training:
		paragraphs = []
		for file in input_files:
			reader = FileReader(file, self.stoplist)
			paragraphs.extend(reader.getSplitParagraphs())
		
		#Train Word2Vec model:
		documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(paragraphs)]


		d2v = Doc2Vec(vector_size=self.vector_size, window=self.window_size, min_count=self.min_count, epochs=self.epochs, dm=self.dm)
		d2v.build_vocab(documents)
		d2v.train(documents, total_examples=d2v.corpus_count, epochs=d2v.epochs)
		d2v.init_sims(replace=True)
		#Return tfidf model:
		return d2v, paragraphs
	
	def getSimilarityMapBetweenSentencesOfParagraphs(self, p1, p2):
		"""
		Produces a matrix containing similarity scores between all sentences in a pair of paragraphs.
				
		* *Parameters*:
			* **p1**: A source paragraph. A paragraph is a list of sentences.
			* **p2**: A target paragraph. A paragraph is a list of sentences.
		* *Output*:
			* **sentence_similarities**: A matrix containing a similarity score between all possible pairs of sentences in the union of p1 and p2. The matrix's height and width are equal and equivalent to the number of distinct sentences present in the union of p1 and p2.
			* **sentence_indexes**: A map connecting each sentence to its numerical index in the sentence_similarities matrix.
		"""
		#Get distinct sentences from paragraphs:
		sentences = p1 + p2
		sentences_list = [item for sublist in sentences for item in sublist]
		sentence_words = self.tokenize_sentences(sentences)
		
		sent_indexes = {}
		for i, s in enumerate(sentences_list):
			sent_indexes[s] = i

		sentence_similarities = []
		for j in range(0,  len(p1)):
			sims = []
			s1 = self.d2v.infer_vector(sentence_words[j], epochs=self.infer_epochs, alpha=0.025, min_alpha=0.0001)
			for k in range(len(p1), len(p1)+len(p2)):
				s2 = self.d2v.infer_vector(sentence_words[k], epochs=self.infer_epochs, alpha=0.025, min_alpha=0.0001)
				sim = 1 - spatial.distance.cosine(s1, s2)
				sims.append(sim)
			sentence_similarities.append(sims)

		#Return similarity matrix:
		return sentence_similarities, sent_indexes
	
	def tokenize_paragraphs(self, paragraph_list):
		word_list_par = []
		for paragraph in paragraph_list:
			par = []
			for sentence in paragraph:
				for token in word_tokenize(sentence.lower()):
					if token not in self.stoplist:
						par.append(token)
			word_list_par.append(par)
		return word_list_par

	def tokenize_sentences(self, sentence_list):
		word_list_sent = []
		for sentence in sentence_list:
			word_list_sent.append([i for i in word_tokenize(sentence.lower()) if i not in self.stoplist])
		return word_list_sent

	def getSimilarityMapBetweenParagraphsOfDocuments(self, p1s=[], p2s=[]):
		"""
		Produces a matrix containing similarity scores between all paragraphs in a pair of paragraph lists.
				
		* *Parameters*:
			* **p1s**: A list of source paragraphs. Each paragraph is a list of sentences.
			* **p2s**: A list of target paragraphs. Each paragraph is a list of sentences.
		* *Output*:
			* **paragraph_similarities**: A matrix containing a similarity score between all possible pairs of paragraphs in the union of p1 and p2. The matrix's height and width are equal and equivalent to the number of distinct paragraphs present in the union of p1s and p2s.
		"""
		#Get distinct sentences from paragraph sets:
		
		# print(p1s)
		# print(p2s)
		#Get TFIDF model controllers:
		paragraph_words = self.tokenize_paragraphs(p1s)+self.tokenize_paragraphs(p2s)
		paragraph_similarities = []
		for j in range(0,  len(p1s)):
			sims = []
			s1 = self.d2v.infer_vector(paragraph_words[j], epochs=self.infer_epochs, alpha=0.025, min_alpha=0.0001)
			for k in range(len(p1s), len(p1s)+len(p2s)):
				s2 = self.d2v.infer_vector(paragraph_words[k], epochs=self.infer_epochs, alpha=0.025, min_alpha=0.0001)
				sim = 1 - spatial.distance.cosine(s1, s2)
				sims.append(sim)
			paragraph_similarities.append(sims)
	
		# #Calculate paragraph similarities:
		# paragraph_similarities = list(np.zeros((len(p1s), len(p2s))))
		# for i, p1 in enumerate(p1s):
		# 	for j, p2 in enumerate(p2s):
		# 		values = []
				
		# 		paragraph_similarities[i][j] = np.max(values)
				
		#Return similarity matrix:
		return paragraph_similarities

	# def getD2VControllers(self, p1s=[], p2s=[]):
	# 	"""
	# 	Produces TFIDF similarity scores between all possible pairs of sentences in a list.
				
	# 	* *Parameters*:
	# 		* **sentences**: A list of sentences.
	# 	* *Output*:
	# 		* **sentence_similarities**: A matrix containing a similarity score between all possible sentence pairs in the input sentence list. The matrix's height and width are equal and equivalent to the number of distinct sentences in the input sentence list.
	# 		* **sentence_indexes**: A map connecting each sentence to its numerical index in the sentence_similarities matrix.
	# 	"""
	# 	#Create data structures for similarity calculation:
	# 	sent_indexes = {}
	# 	for i, s in enumerate(paragraph_mapping):
	# 		sent_indexes[s] = i
			
	# 	#Get similarity querying framework:
	# 	#texts = [[word for word in list(gensim.utils.tokenize(sentence)) if word not in self.stoplist] for sentence in paragraphs]
	# 	#corpus = [self.dictionary.doc2bow(text) for text in texts]
	# 	#index = gensim.similarities.MatrixSimilarity(self.w2v[corpus])
	# 	#print(sentences[0:3])
	# 	#Create similarity matrix:
	# 	paragraph_words = self.tokenize_paragraphs(p1s+p2s)
	# 	paragraph_similarities = []
	# 	for j in range(0, len(paragraph_words)):
	# 		sims = []
	# 		s1 = self.d2v.infer_vector(paragraph_words[j], epochs=self.infer_epochs, alpha=0.025, min_alpha=0.0001)
	# 		for k in range(0, len(paragraph_words)):
	# 			s2 = self.d2v.infer_vector(paragraph_words[k], epochs=self.infer_epochs, alpha=0.025, min_alpha=0.0001)
	# 			sim = 1 - spatial.distance.cosine(s1, s2)
	# 			sims.append(sim)
	# 		paragraph_similarities.append(sims)
		
	# 	#Return controllers:
	# 	return paragraph_similarities
	
	def getTextSimilarity(self, buffer1, buffer2):
		"""
		Calculates the TFIDF similarity between two buffers containing text.
				
		* *Parameters*:
			* **buffer1**: A source buffer containing a block of text.
			* **buffer2**: A target buffer containing a block of text.
		* *Output*:
			* **similarity**: The TFIDF similarity between the two buffers of text.
		"""
		#Get bag-of-words vectors:
		vec1 = buffer1.split()
		vec2 = buffer2.split()
		

		s1 = self.d2v.infer_vector(vec1, epochs=self.infer_epochs, alpha=0.025, min_alpha=0.0001)
		s2 = self.d2v.infer_vector(vec2, epochs=self.infer_epochs, alpha=0.025, min_alpha=0.0001)

		#Return the similarity between the vectors:
		similarity = 1 - spatial.distance.cosine(s1, s2)
		return similarity
	
