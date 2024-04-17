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




# separate with new line
print("\n")




# import the matcher
from spacy.matcher import Matcher

# load a pipeline and create the nlp object
nlp_matcher = spacy.load("en_core_web_sm")

# initialize the matcher with the shared vocab
matcher = Matcher(nlp.vocab)

# add the pattern to the matcher
pattern = [{"TEXT": "iPhone"}, {"TEXT": "X"}]
matcher.add("IPHONE_PATTERN", [pattern])  # (unique id, list of patterns)

# process some text
doc_matcher = nlp_matcher("Upcoming iPhone X release date leaked")

# call the matcher on the doc
matches = matcher(doc_matcher)

# iterate over the matches
for match_id, start, end in matches:  # matches give start and end to slice from doc, match_id is the hash_value of the pattern name
    # get the matched span
    matched_span = doc_matcher[start:end]

    print(matched_span.text)





# separate with new line
print("\n")




# make pattern
pattern = [
    {"IS_DIGIT": True},  # matches digits
    {"LOWER": "fifa"},  # matches if lowercase form leads to fifa
    {"LOWER": "world"},  # matches if lowercase form leads to world
    {"LOWER": "cup"},  # matches if lowercase form leads to cup
    {"IS_PUNCT": True}  # matches if is punctuation
]

# make new doc
doc_matcher = nlp_matcher("2018 FIFA World Cup: France won!")

# initialize the matcher with the shared vocab
matcher = Matcher(nlp.vocab)

# add pattern
matcher.add("FIFA_PATTERN", [pattern])

# call the matcher on the doc
matches = matcher(doc_matcher)

# iterate over the matches
for match_id, start, end in matches:
    # get the matched span
    matched_span = doc_matcher[start:end]

    print(matched_span.text)




# separate with new line
print("\n")




# make pattern
pattern = [
    {"LEMMA": "like", "POS": "VERB"},  # matches verb with the lemma 'like' and a noun
    {"POS": "NOUN"}
]

# make new doc
doc_matcher = nlp_matcher("I liked cats but now I like dogs more.")

# initialize the matcher with the shared vocab
matcher = Matcher(nlp.vocab)

# add pattern
matcher.add("LEMMA_PATTERN", [pattern])

# call the matcher on the doc
matches = matcher(doc_matcher)

# iterate over the matches
for match_id, start, end in matches:
    # get the matched span
    matched_span = doc_matcher[start:end]

    print(matched_span.text)





# separate with new line
print("\n")




# make pattern
pattern = [
    {"LEMMA": "buy"},
    {"POS": "DET", "OP": "?"},  # optional: match 0 or 1 times, matches optional article
    {"POS": "NOUN"}
]

# make new doc
doc_matcher = nlp_matcher("I bought a smartphone. Now I'm buying apps.")

# initialize the matcher with the shared vocab
matcher = Matcher(nlp.vocab)

# add pattern
matcher.add("OPTIONAL_PATTERN", [pattern])

# call the matcher on the doc
matches = matcher(doc_matcher)

# iterate over the matches
for match_id, start, end in matches:
    # get the matched span
    matched_span = doc_matcher[start:end]

    print(matched_span.text)




# separate with new line
print("\n")




doc_matcher = nlp_matcher("Upcoming iPhone X release date leaked as Apple reveals pre-orders")

# create a pattern matching two tokens: 'iPhone' and 'X'
pattern = [{"TEXT": "iPhone"}, {"TEXT": "X"}]

# add the pattern to the matcher
matcher.add("IPHONE_X_PATTERN", [pattern])

# use the matcher on the doc
matches = matcher(doc_matcher)
print("Matches:", [doc_matcher[start:end].text for match_id, start, end in matches])






# separate with new line
print("\n")




# create a final doc
final_doc_1 = nlp_matcher(
    "After making the iOS update you won't notice a radical system-wide "
    "redesign: nothing like the aesthetic upheaval we got with iOS 7. Most of "
    "iOS 11's furniture remains the same as in iOS 10. But you will discover "
    "some tweaks once you delve a little deeper."
)

# make pattern for full iOS versions ("iOS 7", "iOS 11", "iOS 10")
pattern = [{"TEXT": "iOS"}, {"IS_DIGIT": True}]

# add the pattern to the matcher and apply the matcher to the doc
matcher.add("IOS_VERSION_PATTERN", [pattern])
matches = matcher(final_doc_1)
print("Total matches found:", len(matches))  # print number of matches found

# iterate over the matches and print the span text
for match_id, start, end in matches:
    print("Match found:", final_doc_1[start:end].text)





# separate with new line
print("\n")





final_doc_2 = nlp_matcher(
    "i downloaded Fortnite on my laptop and can't open the game at all. Help? "
    "so when I was downloading Minecraft, I got the Windows version where it "
    "is the '.zip' folder and I used the default program to unpack it... do "
    "I also need to download Winzip?"
)

# write a pattern that matches a form of "download" plus proper noun
pattern = [{"LEMMA": "download"}, {"POS": "PROPN"}]

# add pattern to the matcher and apply the matcher to the doc
matcher.add("DOWNLOAD_THINGS_PATTERN", [pattern])
matches = matcher(final_doc_2)
print("Total matches found:", len(matches))

# iterate over the matches and print the span text
for match_id, start, end in matches:
    print("Match found:", final_doc_2[start:end].text)





# separate with new line
print("\n")





