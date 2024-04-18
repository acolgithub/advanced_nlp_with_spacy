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




# overwrite nlp object with medium sized pipeline
nlp = spacy.load("en_core_web_md")  # need medium or large sized pipeline for similarity scoring

# compare two documents
doc1 = nlp("I like fast food")
doc2 = nlp("I like pizza")
print(f"Similarity score between documents doc1 and doc2: {doc1.similarity(doc2)}")


# compare two tokens
doc = nlp("I like pizza and pasta")
token1 = doc[2]
token2 = doc[4]
print(f"Similarity score between tokens token1 and token2: {token1.similarity(token2)}")



# compare a document with a token
doc = nlp("I like pizza")
token = nlp("soap")[0]

# print similarity of token to doc
print(f"Similarity of doc and token: {doc.similarity(token)}")



# compare a span with a document
span = nlp("I like pizza and pasta")[2:5]
doc = nlp("Wendy's sells burgers")

# print similarity of span and doc
print(f"Similarity of span and doc: {span.similarity(doc)}")



# overwrite doc
doc = nlp("I have a banana")

# access the vector via the token.vector attribute
print(doc[3].vector)




# print new line to separate
print("\n")




# overwrite nlp object with medium sized pipeline
nlp = spacy.load("en_core_web_md")

# overwrite doc
doc = nlp("Two bananas in pyjamas")

# get the vector for the token 'bananas'
bananas_vector = doc[1].vector
print(bananas_vector)




# print new line to separate
print("\n")




# overwrite nlp object with medium sized pipeline
nlp = spacy.load("en_core_web_md")

# get docs
doc1 = nlp("It's a warm summer day")
doc2 = nlp("It's sunny outside")

# get the similarity of doc1 and doc2
similarity = doc1.similarity(doc2)
print(f"Similarity of the documents doc1 and doc2: {similarity}")


# get tokens from doc
doc = nlp("TV and books")
token1, token2 = doc[0], doc[2]  # get 'TV' and 'books'

# get the similarity of the tokens 'TV' and 'books'
similarity = token1.similarity(token2)
print(f"Similarity of the tokens token1 and token2: {similarity}")



# create new document
doc = nlp("This was a great restaurant. Afterwards, we went to a really nice bar.")

# create span for 'great restaurant' and 'really nice bar'
span1 = Span(doc, 3, 5)
span2 = Span(doc, 12, 15)

# get the similarity of the spans
similarity = span1.similarity(span2)
print(f"Similarity of the spans span1 and span2: {similarity}")




# print new line to separate
print("\n")



# import matcher
from spacy.matcher import Matcher

# initialize matcher with the shared vocab
matcher = Matcher(nlp.vocab)

# patterns are lists of dictionaries describing the tokens
pattern = [{"LEMMA": "love", "POS": "VERB"}, {"LOWER": "dogs"}]  # find lemma 'love' when part of speech is verb followed by words whose lowercase form result in 'cats'
matcher.add("LOVE_DOGS", [pattern])  # add pattern to matcher

# operators can specify how often a token should be matched
pattern = [{"TEXT": "very", "OP": "+"}, {"TEXT": "happy"}]  # find 'very' as text 1 or more times followed by find 'happy' as text
matcher.add("VERY_HAPPY", [pattern])

# calling matcher on doc returns list of (match_id, start, end) tuples
doc = nlp("I love dogs and I'm very very happy")
matches = matcher(doc)

# iterate over matches
for matchid, start, end in matches:
    print(doc[start:end])




# overwrite matcher
matcher = Matcher(nlp.vocab)

# add pattern to matcher
matcher.add("DOG", [[{"LOWER": "golden"}, {"LOWER": "retriever"}]])

# create document
doc = nlp("I have a Golden Retriever")

# iterate over matches
for match_id, start, end in matcher(doc):
    # create span
    span = doc[start:end]

    # print text of span
    print(f"Matched span: {span.text}")

    # get the span's root token and root head token
    print(f"Root token: {span.root.text}")  # root token decides category of phrase
    print(f"Root head token: {span.root.head.text}")  # head token of root is syntactic parent that governs phrase

    # get the previous token and its POS tag
    print(f"Previous token: {doc[start - 1].text, doc[start - 1].pos_}")





# import phrase matcher
from spacy.matcher import PhraseMatcher

# create phrase matcher
phrase_matcher = PhraseMatcher(nlp.vocab)

# get pattern
pattern = nlp("Golden Retriever")

# add pattern to phrase matcher
phrase_matcher.add("DOG", [pattern])  # pattern from before about dogs

# get document
doc = nlp("I have a Golden Retriever")

# iterate over the matches
for match_id, start, end in matcher(doc):
    # get the matched span
    span = doc[start:end]

    print(f"Matched span: {span.text}")




# print new line to separate
print("\n")




# overwrite nlp object
nlp = spacy.load("en_core_web_sm")

# get document
doc = nlp(
    "Twitch Prime, the perks program for Amazon Prime members offering free "
    "loot, games and other benefits, is ditching one of its best features: "
    "ad-free viewing. According to an email sent out to Amazon Prime members "
    "today, ad-free viewing will no longer be included as a part of Twitch "
    "Prime for new members, beginning on September 14. However, members with "
    "existing annual subscriptions will be able to continue to enjoy ad-free "
    "viewing until their subscription comes up for renewal. Those with "
    "monthly subscriptions will have access to ad-free viewing until October 15."
)

# create the match patterns
pattern1 = [{"LOWER": "amazon"}, {"IS_TITLE": True, "POS": "PROPN"}]  # matches words whose lowercases is 'amazon' followed by prosition
pattern2 = [{"LOWER": "ad"}, {"TEXT": "-"}, {"LOWER": "free"}, {"POS": "NOUN"}]  # matches words whose lowercase letters correspond to 'ad-free' followed by noun

# initialize the matcher and add the patterns
matcher = Matcher(nlp.vocab)
matcher.add("PATTERN1", [pattern1])
matcher.add("PATTERN2", [pattern2])

# iterate over the matches
for match_id, start, end in matcher(doc):
    # print pattern string name (the match id) and text of matched span
    print(doc.vocab.strings[match_id], doc[start:end].text)









