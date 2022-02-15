# Importing the Tf-idf vectorizer from sklearn
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from underthesea import word_tokenize as tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--input', help = "put the input txt file here", type = str)
parser.add_argument('--result', help = "put the output txt file here", type = str)
args = parser.parse_args()
stopwords = []
with open('vnstop.txt', 'r', encoding='utf-8') as fstop:
        for c in fstop:
                #stopwords.append(c.strip())
                temp = c.strip().split(' ')
                stopwords.append("_".join(temp))
texts = []
with open(args.input, 'r', encoding='utf-8') as fin:
        for c in fin:
                temp = tokenizer(c.strip(), format = 'text')
                #print(temp)
                temp = temp.split(' ')
                text = []
                for token in temp:
                        if token not in stopwords:
                                text.append(token)
                #print(' '.join(text))
                texts.append(' '.join(text))
# Defining the vectorizer
vectorizer = TfidfVectorizer(stop_words = stopwords, max_features= 1000,  max_df = 0.5, smooth_idf=True)



# Transforming the tokens into the matrix form through .fit_transform()
matrix= vectorizer.fit_transform(texts)

# SVD represent documents and terms in vectors
from sklearn.decomposition import TruncatedSVD
SVD_model = TruncatedSVD(n_components=4, algorithm='randomized', n_iter=100, random_state=122)
SVD_model.fit(matrix)

# Getting the terms 
terms = vectorizer.get_feature_names_out()

topic = []
# Iterating through each topic
for i, comp in enumerate(SVD_model.components_):
        terms_comp = zip(terms, comp)
        # sorting the 7 most important terms
        sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:12]
        temp = []
        for t in sorted_terms:
                temp.append(t[0])
        topic.append(temp)


with open(args.result, 'w', encoding='utf-8') as fout:
        for i in range(0,4):
                fout.write("Topic "+str(i)+": ")
                # printing the terms of a topic
                for j in topic[i]:
                        fout.write(j)
                        fout.write(' ')
                fout.write('\n')

