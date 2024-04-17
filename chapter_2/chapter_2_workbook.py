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














