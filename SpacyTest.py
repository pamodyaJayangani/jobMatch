import spacy
from pyjarowinkler import distance
from nltk.corpus import wordnet

nlp = spacy.load('en_core_web_sm')

doc = nlp(u"This is a sentence.")

doc1 = nlp("Team leader")
doc2 = nlp("self driven team")
print(doc1.similarity(doc2))
print (distance.get_jaro_distance("mary", "mrray", winkler=True, scaling=0.1))

synonyms = []
for syn in wordnet.synsets("EXTRAVERTED"):
    print("2 **!!: " +str(syn))
    for l in syn.lemmas():
        print("2 !!: " + l.name())
        synonyms.append(l.name())