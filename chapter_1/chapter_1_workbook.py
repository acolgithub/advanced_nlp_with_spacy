import spacy



# create a blank English nlp object
nlp = spacy.blank("en")

# process a string of text with nlp object
doc = nlp("Hello world!")

# iterate over tokens in doc
for token in doc:
    print(token.text)

# index into the doc to get a single token
token = doc[1]

# get the token text via the .text attribute
print(token.text)

# get a slice from the doc, referred to as a Span object
span = doc[1:3]

# get the span text via the .text attribute
print(span.text)

# get new doc
doc = nlp("It costs $5.")

# print token indices and token text
print(f"Index: {[token.i for token in doc]}")
print(f"Text: {[token.text for token in doc]}")

# print bools indicating alphabetical, punctuation, or numeric
print(f"is_alpha: {[token.is_alpha for token in doc]}")
print(f"is_punct: {[token.is_punct for token in doc]}")
print(f"like_num: {[token.like_num for token in doc]}")





# separate with new line
print("\n")





# process some text
doc = nlp("This is a sentence.")

# print the text of the document consisting of processed text
print(doc.text)




# now create a blank german nlp object
nlp_german = spacy.blank("de")

# porcess some German text (this is German for: "Kind regards!")
doc_german = nlp_german("Liebe Grüße!")

# print the German document text
print(doc_german.text)




# next create a blank spanish nlp object
nlp_spanish = spacy.blank("es")

# process some Spanish text (this is Spanish for: "How are you?")
doc_spanish = nlp_spanish("¿Cómo estás?")

# print the Spanish text
print(doc_spanish.text)



# separate with new line
print("\n")




# process text
new_doc = nlp("I like tree kangaroos and narwhals.")

# select first token by indexing
first_token = new_doc[0]

# print the first token's text
print(f"The first token is: {first_token.text}")



# separate with new line
print("\n")





# process text
another_new_doc = nlp("I like tree kangaroos and narwhals.")

# a slice of the doc for 'tree kangaroos'
tree_kangaroos = another_new_doc[2:4]
print(f"A slice of the doc for: {tree_kangaroos.text}")

# a slice of the doc for 'tree kangaroos and narwhals' (without the '.')
tree_kangaroos_and_narwhals = another_new_doc[2:-1]
print(f"A slice of the doc for: {tree_kangaroos_and_narwhals.text}")



# separate with new line
print("\n")





# process some text
doc_text = nlp(
    "In 1990, more than 60% of people in East Asia were in extreme poverty. "
    "Now less than 4% are."
)

# iterate over the tokens in the doc
for token in doc_text:
    # first check if the token resembles a number
    if token.like_num:
        # get the next token in the document
        next_token = doc_text[token.i + 1]  # token.i gets index position of token
        
        # check if the next token's text equals '%'
        if next_token.text == '%':
            print(f"Percentage found: {token.text}")



# separate with new line
print("\n")




# load the small English pipeline package
nlp_pipeline = spacy.load("en_core_web_sm")

# process some text,  we will try to determine part of speech of each word
doc_pipeline = nlp_pipeline("She ate the pizza")

# iterate over the tokens
for token in doc_pipeline:
    # print the text and the predicted part-pf-speech tag
    print(token.text, token.pos_)  # print token text, as well as predicted part of speech


# predict syntactic dependencies
for token in doc_pipeline:
    print(token.text, token.pos_, token.dep_, token.head.text)  # print text, predicted part of speech, predicted dependency label, and syntactic head token


# process some different text
doc_pipeline = nlp_pipeline("Apple is looking at buying U.K. startup for $1 billion")

# iterate over the predicted entities
for ent in doc_pipeline.ents:  # gets predicted doc entities of Span type
    # print the entity text and its label
    print(ent.text, ent.label_)  # print predicted named entities as well as its classification




# separate with new line
print("\n")




# load the small English pipeline package
nlp_pipeline = spacy.load("en_core_web_sm")

# get text
text = "It's official: Apple is the first U.S. public company to reach a $1 trillion market value"

# process text
doc_pipeline = nlp_pipeline(text)

# print the document text
print(doc_pipeline.text)

# iterate over tokens and get text, part of speech, and syntactic dependency
for token in doc_pipeline:
    token_text = token.text  # get token text
    token_pos = token.pos_  # get token part of speech
    token_dep = token.dep_  # get syntactic dependency

    print(f"{token_text:<12}{token_pos:<10}{token_dep:<10}")





# separate with new line
print("\n")



# iterate over the predicted entities
for ent in doc_pipeline.ents:
    # print the entity text and its label
    print(ent.text, ent.label_)




# separate with new line
print("\n")




# get new text
new_text = "Upcoming iPhone X release date leaked as Apple reveals pre-orders"

# process the new text
doc_pipeline = nlp_pipeline(new_text)

# iterate over the entities
for ent in doc.ents:
    # print the entity text and label
    print(ent.text, ent.label_)

# get the span for 'iPhone X'
iphone_x = doc_pipeline[1:3]

# print the span text
print(f"Missing entity: {iphone_x.text}")





