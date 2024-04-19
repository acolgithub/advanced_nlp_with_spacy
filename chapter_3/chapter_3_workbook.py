import spacy
from spacy.language import Language  # to modify pipeline
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

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










