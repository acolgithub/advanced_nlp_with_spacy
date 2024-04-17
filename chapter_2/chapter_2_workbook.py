import spacy

# create a blank English nlp object
nlp = spacy.blank("en")

# add coffee to vocab
nlp.vocab.strings.add("coffee")  # adds coffee to vocabulary, encoded as hash

# get coffee has
coffee_hash = nlp.vocab.strings["coffee"]
print(f"coffee hash: {coffee_hash}")  # pritn hash

# get string
coffee_string = nlp.vocab.strings[coffee_hash]
print(f"coffee string: {coffee_string}")

# create document
doc = nlp("I love coffee")

# get hash and string values for coffee
print(f"hash value: {nlp.vocab.strings['coffee']}")  # get hash value for coffee
print(f"string value: {nlp.vocab.strings[nlp.vocab.strings['coffee']]}")  # get string value

# doc also exposes the vocab and strings
print(f"hash value and string: {doc.vocab.strings['coffee']}, {doc.vocab.strings[doc.vocab.strings['coffee']]}")

# get coffee lexeme
lexeme = nlp.vocab["coffee"]

# print the lexical attributes
print(f"lexeme attributes (text, hash, is alphabetical): {lexeme.text}, {lexeme.orth}, {lexeme.is_alpha}")





# print new line to separate
print("\n")




# create a blank English nlp object
nlp = spacy.blank("en")

# create document
doc = nlp("I have a cat")

# look up the hash for the word 'cat'
cat_hash = nlp.vocab.strings["cat"]
print(cat_hash)

# look up the cat_hash to get the string
cat_string = nlp.vocab.strings[cat_hash]
print(cat_string)


# overwrite document
doc = nlp("David Bowie is a PERSON")

# look up the hash for the string label 'PERSON'
person_hash = nlp.vocab.strings["PERSON"]
print(person_hash)

# look up the person_hash to get the string
person_string = nlp.vocab.strings[person_hash]
print(person_string)




# print new line to separate
print("\n")




# overwriting nlp object
nlp = spacy.blank("en")

# import the Doc class
from spacy.tokens import Doc

# create a list of words and 'spaces' to create the doc from
words = ["Hello", "world", "!"]  # list of words to be used
spaces = [True, False, False]  # indicates  whether word is followed by space

# create a doc manually
doc = Doc(nlp.vocab, words=words, spaces=spaces)  # form doc using vocabulary, list of words, and indication of space



# import the Span class
from spacy.tokens import Span

# create a span manually
span = Span(doc, 0, 2)  # create span with document, start index, end index

# create a span with a label
span_with_label = Span(doc, 0, 2, label="GREETING")  # same as before but now added label

# add span with label to the document entities
doc.ents = [span_with_label]




# print new line to separate
print("\n")




# overwriting nlp object
nlp = spacy.blank("en")

# create desired text and indicate if spaces follow words
words = ["spaCy", "is", "cool", "!"]  # intended sentence is 'spaCy is cool!'
spaces = [True, True, False, False]

# create a doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)


# overwrite words and spaces
words = ["Go", ",", "get", "started", "!"]  # intended sentence is 'Go, get started!'
spaces = [False, True, True, False, False]

# create a doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)



# overwrite words and spaces
words = ["Oh", ",", "really", "?", "!"]
spaces = [False, True, False, False, False]

# create a doc crom the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)




# print new line to separate
print("\n")



# overwriting nlp object
nlp = spacy.blank("en")

# create desired words and indicate if spaces follow words
words = ["I", "like", "David", "Bowie"]  # intended sentence is 'I like David Bowie'
spaces = [True, True, True, False]

# create a doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)

# create a span for 'David Bowie' from the doc and assign it the label 'PERSON'
span = Span(doc, 2, 4, label="PERSON")
print(span.text, span.label_)

# add span to the doc's entities
doc.ents = [span]

# print doc entities' text and labels
print(f"Entities text and labels: {[(ent.text, ent.label_) for ent in doc.ents]}")




# print new line to separate
print("\n")




# overwrite nlp object
nlp = spacy.load("en_core_web_sm")

# overwrite doc
doc = nlp("Berlin looks like a nice city")

# iterate over tokens in doc
for token in doc:

    # Check if the current token is a proper noun, index + 1 is in bounds, and next token is a verb
    if (token.pos_ == "PROPN") and (token.i + 1 < len(doc)) and (doc[token.i + 1].pos_ == "VERB"):
        print("Found proper noun before a verb:", token.text)




# print new line to separate
print("\n")















