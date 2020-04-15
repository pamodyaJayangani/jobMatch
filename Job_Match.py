import re
import pandas as pd
import sys, os
import numpy as np
import nltk
import operator
import math
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.stem.wordnet import WordNetLemmatizer
# nltk.download('wordnet')
from nltk.corpus import wordnet
import gensim
from gensim.models import Word2Vec
from difflib import SequenceMatcher
import spacy
from pyjarowinkler import distance
import xlwt
from xlwt import Workbook
# from FileWriter import ResultElementTest
from xlrd import open_workbook
import Predict_Test

class Extractor():
    def __init__(self):
        # self.count = int
        self.softskills = self.load_skills('D:\\5THYEAR\\FYP\\jobMatch\\softskills.txt')
        self.hardskills = self.load_skills('D:\\5THYEAR\\FYP\\jobMatch\\hardskills.txt')
        self.jb_distribution = self.build_ngram_distribution('D:\\5THYEAR\\FYP\\jobMatch\\tesla_job_description.txt')
        self.cv_distribution = self.build_ngram_distribution('D:\\5THYEAR\\FYP\\jobMatch\\my_resume2.txt')
        self.cv = self.load_skills('D:\\5THYEAR\\FYP\\jobMatch\\my_resume2.txt')
        self.table = []
        # self.outFile = "D:\\5THYEAR\\FYP\\jobMatch\\Extracted_keywords.csv"
        self.outFile = "D:\\5THYEAR\\FYP\\jobMatch\\Detail\\"
        self.nlp = spacy.load('en_core_web_sm')

        doc = self.nlp(u"This is a sentence.")
        # Workbook is created
        # self.wb = open_workbook('D:\\5THYEAR\\FYP\\jobMatch\\xlwt example.xls')

        # self.wb = open('D:\\5THYEAR\\FYP\\jobMatch\\xlwt example.xls', 'a', newline='', encoding='utf-8')
        # self.count


        # print("::::::::::::::::::::::::::::::;"+(str(self.count)))
        # print(type(str(self.count)))

    def load_skills(self, filename):
        f = open(filename, 'r')
        skills = []
        for line in f:
            # removing punctuation and upper cases
            skills.append(self.clean_phrase(line))
        f.close()
        return list(set(skills))  # remove duplicates

    def build_ngram_distribution(self, filename):
        n_s = [1, 2, 3]  # mono-, bi-, and tri-grams
        dist = {}
        for n in n_s:
            dist.update(self.parse_file(filename, n))
        return dist

    def parse_file(self, filename, n):
        f = open(filename, 'r')
        results = {}
        for line in f:
            words = self.clean_phrase(line).split(" ")
            ngrams = self.ngrams(words, n)
            for tup in ngrams:
                phrase = " ".join(tup)
                if phrase in results.keys():
                    results[phrase] += 1
                else:
                    results[phrase] = 1
        return results

    def clean_phrase(self, line):
        return re.sub(r'[^\w\s]', '', line.replace('\n', '').replace('\t', '').lower())

    def ngrams(self, input_list, n):
        return list(zip(*[input_list[i:] for i in range(n)]))

    def measure1(self, v1, v2):
        print("*** measure1 ")
        return v1 - v2

    def measure2(self, v1, v2):
        print("*** measure2")
        return max(v1 - v2, 0)

    def measure3(self, v1, v2):  # cosine similarity
        # "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
        print("*** measure3")
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i];
            y = v2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
            print(x)
            print(y)
        print("-------------------______________")
        print(sumxx)
        print(sumyy)
        print(sumxy)
        cosineSimilarity = sumxy / math.sqrt(sumxx * sumyy)
        print(cosineSimilarity)
        if(cosineSimilarity == 'nan'):
                cosineSimilarity = 0.0
        return cosineSimilarity

    def similar(self, skill, requirement):
        print("*** similar")
        # for req in requirement:
        #     # print(req+" 5555555")
        #     self.ratio = SequenceMatcher(None,"2 yrs","24 years").ratio()
        #     # print("**** Skill : "+skill+" *** Req : "+req+" Ratio : "+str(self.ratio))
        #     return SequenceMatcher(None,skill,req).ratio() > 0.6
        self.ratio = 0.0;
        for req in requirement:
            doc1 = self.nlp(skill)
            doc2 = self.nlp(req)
            print(doc1.similarity(doc2))

            if(self.ratio < doc1.similarity(doc2)):
                self.wrinkleRatio = distance.get_jaro_distance(skill, req, winkler=True, scaling=0.1)
                if(self.ratio < self.wrinkleRatio and self.wrinkleRatio>0.75):
                    self.ratio = self.wrinkleRatio;
                else:
                    self.ratio = doc1.similarity(doc2);
            # print(skill + " Skill Similarity " + self.nlp((skill)).similarity(self.nlp((skill1))) + " Word" + skill1)
            print(type(req))
            print("!!!!!!!!!!!!!!!!!!!")
            print((skill) + " Skill Similarity " + " Word " + (req))
            return self.ratio > 0.55

    def sendToFile(self):
        print("*** sendToFile"+self.outFile+str(self.fileName[0])+".csv")
        try:
            os.remove(self.outFile+str(self.fileName[0])+".csv")

        except OSError:
            pass
        df = pd.DataFrame(self.table, columns=['type', 'skill', 'job', 'cv', 'm1', 'm2'])
        df_sorted = df.sort_values(by=['job', 'cv'], ascending=[False, False])
        df_sorted.to_csv(self.outFile+str(self.fileName[0])+".csv", columns=['type', 'skill', 'job', 'cv'], index=False)

    def printMeasures(self):
        print("*** printMeasures")
        # if (self.count == 1):
        #     self.wb = Workbook()
        #     # add_sheet is used to create sheet.
        #     self.sheet1 = self.wb.add_sheet('Sheet 1', 'False')
        n_rows = len(self.table)
        v1 = [self.table[m1][4] for m1 in range(n_rows)]
        v2 = [self.table[m2][5] for m2 in range(n_rows)]
        print("Measure 1: ", str(sum(v1)))
        print("Measure 2: ", str(sum(v2)))

        v1 = [self.table[jb][2] for jb in range(n_rows)]
        v2 = [self.table[cv][3] for cv in range(n_rows)]
        print("Measure 3 (cosine sim): ", str(self.measure3(v1, v2)))

        # self.count = self.count+1
        rowCount = 0
        self.sheet1.write(self.count, rowCount, self.fileName)
        for type in ['hard', 'soft', 'general']:
            v1 = [self.table[jb][2] for jb in range(n_rows) if self.table[jb][0] == type]
            v2 = [self.table[cv][3] for cv in range(n_rows) if self.table[cv][0] == type]
            print("Cosine similarity for " + type + " skills: " + str(self.measure3(v1, v2)))
            # print(type(self.count))

            rowCount +=1
            self.sheet1.write(self.count,rowCount,str(self.measure3(v1, v2)))
            # self.sheet1.write(self.count+1,rowCount,  str(self.measure3(v1, v2)))
            # self.sheet1.write(0, 3,  str(self.measure3(v1, v2)))
            # self.sheet1.write(1, 1,  str(self.measure3(v1, v2)))
            # self.sheet1.write(1, 2,  str(self.measure3(v1, v2)))
            # self.sheet1.write(1, 3, str(self.measure3(v1, v2)))
            print(self.count)
            # self.wb.save('D:\\5THYEAR\\FYP\\jobMatch\\xlwt example.xls')
        print(self.cv[0]+"*************^^^^^^^^^^^^^^^^^^%%")
        P = Predict_Test.Prediction()
        personality = P.personality_Prediction(self.cv[0])
        print("*************^^^^^^^^^ {ersonlaity^^^^^^^^^"+personality)
        self.sheet1.write(self.count, rowCount+1, personality)
    def makeTable(self,num,sheet1,fileName):
        print("*** makeTable"+str(num))
        self.count = num
        self.sheet1 = sheet1
        self.fileName = fileName
        # I am interested in verbs, nouns, adverbs, and adjectives
        parts_of_speech = ['CD', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'RB',
                           'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        graylist = ["you", "will"]
        tmp_table = []
        # look if the skills are mentioned in the job description and then in your cv
        lemma = WordNetLemmatizer()

        for skill in self.hardskills:
            print("skill : "+skill)
            # # # normalized = lemma.lemmatize(skill)
            # synonyms = []
            # for syn in wordnet.synsets("Object Oriented Programing"):
            #     print("2 **!!: " +str(syn))
            #     for l in syn.lemmas():
            #         print("2 !!: " + l.name())
            #         synonyms.append(l.name())
            #
            # for skill in set(synonyms):
            #     print("2 : " + skill)
            if skill in self.jb_distribution:
                count_jb = self.jb_distribution[skill]
                if skill in self.cv_distribution:
                    print("1")
                    count_cv = self.cv_distribution[skill]
                elif self.similar(skill, self.cv_distribution):
                    count_cv = self.ratio
                else:
                    count_cv = 0
                m1 = self.measure1(count_jb, count_cv)
                m2 = self.measure2(count_jb, count_cv)
                tmp_table.append(['hard', skill, count_jb, count_cv, m1, m2])
            else:
                if(self.similar(skill, self.jb_distribution)):
                    print("2")

                    count_jb = self.ratio
                    if skill in self.cv_distribution:
                        count_cv = self.cv_distribution[skill]
                    elif self.similar(skill, self.cv_distribution):
                        count_cv = self.ratio
                    else:
                        count_cv = 0
                    m1 = self.measure1(count_jb, count_cv)
                    m2 = self.measure2(count_jb, count_cv)
                    tmp_table.append(['hard', skill, count_jb, count_cv, m1, m2])




        for skill in self.softskills:
            # # normalized = lemma.lemmatize(skill)
            # synonyms = []
            # for syn in wordnet.synsets(skill):
            #     for l in syn.lemmas():
            #         synonyms.append(l.name())
            # for skill in set(synonyms):
            # for skill1 in self.jb_distribution:
            #     doc1 = self.nlp(skill)
            #     doc2 = self.nlp(skill1)
            #     print(doc1.similarity(doc2))
            #     # print(skill + " Skill Similarity " + self.nlp((skill)).similarity(self.nlp((skill1))) + " Word" + skill1)
            #     print(type(skill1))
            #     print("!!!!!!!!!!!!!!!!!!!")
            #     print((skill) + " Skill Similarity " +  " Word " + (skill1))
            if skill in self.jb_distribution :

                 count_jb = self.jb_distribution[skill]
                 if skill in self.cv_distribution:
                     count_cv = self.cv_distribution[skill]
                 elif self.similar(skill, self.cv_distribution):
                     count_cv = self.ratio
                 else:
                     count_cv = 0
                 m1 = self.measure1(count_jb, count_cv)
                 m2 = self.measure2(count_jb, count_cv)
                 tmp_table.append(['soft', skill, count_jb, count_cv, m1, m2])

            else:
                if(self.similar(skill, self.jb_distribution)):
                    count_jb = self.ratio
                    if skill in self.cv_distribution:
                        count_cv = self.cv_distribution[skill]
                    elif self.similar(skill, self.cv_distribution):
                        count_cv = self.ratio
                    else:
                        count_cv = 0
                    m1 = self.measure1(count_jb, count_cv)
                    m2 = self.measure2(count_jb, count_cv)
                    tmp_table.append(['hard', skill, count_jb, count_cv, m1, m2])



        # And now for the general language of the job description:
        # Sort the distribution by the words most used in the job description

        general_language = sorted(self.jb_distribution.items(), key=operator.itemgetter(1), reverse=True)

        for tuple in general_language:
            skill = tuple[0]
            if skill in self.hardskills or skill in self.softskills or skill in graylist:
                continue
            count_jb = tuple[1]
            stop = set(stopwords.words('english'))

            if skill not in stop:
                tokens = nltk.word_tokenize(skill)
                parts = nltk.pos_tag(tokens)
                if all([parts[i][1] in parts_of_speech for i in range(len(parts))]):
                    # synonyms = []
                    # print("3 : "+skill)
                    # for syn in wordnet.synsets(skill):
                    #     for l in syn.lemmas():
                    #         print("5 : "+l.name())
                    #         synonyms.append(l.name())
                    # if set(synonyms):
                    #     for skill in set(synonyms):

                    if skill in self.cv_distribution:
                        count_cv = self.cv_distribution[skill]
                    elif self.similar(skill, self.cv_distribution):
                        count_cv = self.ratio
                    else:
                        count_cv = 0
                    m1 = self.measure1(count_jb, count_cv)
                    m2 = self.measure2(count_jb, count_cv)
                    tmp_table.append(['general', skill, count_jb, count_cv, m1, m2])
                    # else:
                    #     print("6 : " + skill)
                    #     if skill in self.cv_distribution:
                    #         count_cv = self.cv_distribution[skill]
                    #     else:
                    #         count_cv = 0
                    #     m1 = self.measure1(count_jb, count_cv)
                    #     m2 = self.measure2(count_jb, count_cv)
                    #     tmp_table.append(['general', skill, count_jb, count_cv, m1, m2])
        self.table = tmp_table


def main():
    print("*******22")
    K = Extractor()
    K.makeTable()
    K.sendToFile()
    K.printMeasures()


if __name__ == "__main__":
    print("*******")
    main()

