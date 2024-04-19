import spacy
import random
from spacy.tokens import Span
from spacy.tokens import DocBin

# create blank nlp object
nlp = spacy.blank("en")

# create a Doc with entity spans
doc1 = nlp("iPhone X is coming")

# identify entity
doc1.ents = [Span(doc1, 0, 2, label="GADGET")]

# create another doc without entity spans
doc2 = nlp("I need a new phone! Any tips?")

# form dataset of docs
docs = [doc1, doc2]


# from here follow practices of machine learning
# shuffle dataset and divide into train/test
random.shuffle(docs)
train_docs = docs[:len(docs) // 2]  # training set
dev_docs = docs[len(docs) // 2:]  # testing set

# create and save a collection of training docs
train_docbin = DocBin(docs=train_docs)
train_docbin.to_disk("./datasets/train.spacy")

# create and save a collection of evaluation docs
dev_docbin = DocBin(docs=dev_docs)
dev_docbin.to_disk("./datasets/dev.spacy")




















