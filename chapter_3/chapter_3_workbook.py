import spacy
from spacy.language import Language  # to modify pipeline
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.tokens import Doc, Token

@Language.component("custom_component")  # decorator to indicate to pipeline where to find component
def custom_component_function(doc):
    # print the doc's length
    print(f"Doc length: {len(doc)}")

    # return the doc object
    return doc


@Language.component("length_component")
def length_component_function(doc):

    # get the doc's length
    doc_length = len(doc)
    print(f"This document is {doc_length} tokens long.")

    # return the doc
    return doc


# load the en_core_web_sm pipeline
nlp = spacy.load("en_core_web_sm")

# print the names of the pipeline components
print(nlp.pipe_names, "\n")

# print the full pipeline of (name, component tuples)
print(nlp.pipeline, "\n")




# print line to separate
print("\n")




# add custom function to pipeline right after tokenizing
nlp.add_pipe("custom_component", first=True)  # other options instead of first are 'last', 'before', and 'after'

# print the pipeline component names
print(f"Pipeline: {nlp.pipe_names}")

# process text with modified pipeline
doc = nlp("Hello world!")  # should print 'Doc length: number_of_tokens'




# print line to separate
print("\n")




# overwrite nlp object
nlp = spacy.load("en_core_web_sm")

# add the length component first in the pipeline and print the pipe names
nlp.add_pipe("length_component", first=True)
print(nlp.pipe_names)

# process a text
doc = nlp("This is a sentence.")






# print line to separate
print("\n")






# overwrite nlp object
nlp = spacy.load("en_core_web_sm")

# construct list of animals
animals = ["Golden Retriever", "cat", "turtle", "Rattus norvegicus"]

# get list of animal patterns in type that works with Phrase Matcher
animal_patterns = list(nlp.pipe(animals))

print("animal_patterns:", animal_patterns)

# create matcher
matcher = PhraseMatcher(nlp.vocab)

# add animal patterns
matcher.add("ANIMAL", animal_patterns)

# define the custom component
@Language.component("animal_component")
def animal_component_function(doc):

    # apply the matcher to the doc
    matches = matcher(doc)

    # create a Span for each match and assign the label 'ANIMAL'
    spans = [Span(doc, start, end, label="ANIMAL") for match_id, start, end in matches]

    # overwrite the doc.ents with the matches spans
    doc.ents = spans

    # return doc
    return doc

# add the component to the pipeline after the 'ner' component
nlp.add_pipe("animal_component", after="ner")
print(nlp.pipe_names)

# process the text and print the text and label for the doc.ents
doc = nlp("I have a cat and a Golden Retriever")
print([(ent.text, ent.label_) for ent in doc.ents])





# print line to separate
print("\n")







# define getter function for token properties
def get_is_colour(token):

    # list of colours
    colours = ["red", "yellow", "blue"]

    # check if token text is in list
    return token.text in colours

# define getter function for span properties
def get_has_colour(span):

    # list of colours
    colours = ["red", "yellow", "blue"]

    # check if any token in span is in list
    return any(token.text in colours for token in span)

# define method with arguments
def has_token(doc, token_text):

    # check if token text is in doc for each token
    in_doc = token_text in [token.text for token in doc]

    return in_doc

# set extensions on the Doc, Token, and Span to register attributes
#Doc.set_extension("title", default=None)  # set no default
Doc.set_extension("has_token", method=has_token)  # added method function
Token.set_extension("is_colour", getter=get_is_colour)  # added getter function
Span.set_extension("has_colour", getter=get_has_colour)

# overwrite nlp object
nlp = spacy.load("en_core_web_sm")

# create doc
doc = nlp("The sky is blue.")

# overwrite extension attribute value
doc[3]._.is_colour = True

# check if fourth token is colour in list
print(f"{doc[3]._.is_colour} - {doc[3].text}")

# check if different spans have colour in list
print(f"{doc[1:4]._.has_colour} - {doc[1:4].text}")
print(f"{doc[0:2]._.has_colour} - {doc[0:2].text}")

# check against method function
print(f"{doc._.has_token('blue')} - blue")
print(f"{doc._.has_token('cloud')} - cloud")






# print line to separate
print("\n")





# overwrite nlp object
nlp = spacy.blank("en")

# register the Token extension attribute 'is_country' with the default value False
Token.set_extension("is_country", default=False)

# process the text and set is_country attribute to True for the token 'Spain'
doc = nlp("I live in Spain.")
doc[3]._.is_country = True

# print the token text and the is_country attribute for all tokens
print([(token.text, token._.is_country) for token in doc])





# print line to separate
print("\n")





# overwrite nlp object
nlp = spacy.blank("en")

# define the getter function that takes a token and returns its reverse text
def get_reversed(token):
    return token.text[::-1]

# register the token property extension 'reversed' with the getter get_reversed
Token.set_extension("reversed", getter=get_reversed)

# process the text and print the reversed attribute for each token
doc = nlp("All generalizations are false, including this one.")
for token in doc:
    print("reversed", token._.reversed)





# print line to separate
print("\n")




# overwrite nlp object
nlp = spacy.blank("en")

# define the getter function
def get_has_number(doc):

    # return if any of the tokens in the doc return True for token.like_num
    return any(token.like_num for token in doc)

# register the Doc property extension 'has_number' with the getter get_has_number
Doc.set_extension("has_number", getter=get_has_number)

# process the text and check the custom has_number attribute
doc = nlp("The museum closed for five years in 2012.")
print(f"has_number: {doc._.has_number}")





# print line to separate
print("\n")





# overwrite nlp object
nlp = spacy.blank("en")

# define the method
def to_html(span, tag):

    # wrap the span text in a HTML tag and return it
    return f"<{tag}>{span.text}</{tag}>"

# register the Span method extension 'to_html' with the method to_html
Span.set_extension("to_html", method=to_html)

# process the text and call the to_html method on the span with the tag name 'strong'
doc = nlp("Hello world, this is a sentence.")
span = doc[0:2]
print(span._.to_html("strong"))





# print line to separate
print("\n")





# overwrite nlp object
nlp = spacy.load("en_core_web_sm")

# define getter that returns wikipedia url if span label is in list
def get_wikipedia_url(span):

    # get a wikipedia url if the span has one of the labels
    if span.label_ in ("PERSON", "ORG", "GPE", "LOCATION"):
        entity_text = span.text.replace(" ", "_")
        return "https://en.wikipedia.org/w/index.php?search=" + entity_text
    
# set the Span extension wikipedia_url using the getter get_wikipedia_url
Span.set_extension("wikipedia_url", getter=get_wikipedia_url)

# get document
doc = nlp(
    "In over fifty years from his very first recordings right through to his "
    "last album, David Bowie was at the vanguard of contemporary culture."
)

# loop over entities and print text and wikipedia url if applicable
for ent in doc.ents:

    # print the text and wikipedia url of the entity
    print(ent.text, ent._.wikipedia_url)





