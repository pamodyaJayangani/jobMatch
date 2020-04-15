import glob
import os
import warnings
import textract
import csv
import requests

from flask import (Flask, json, Blueprint, jsonify, redirect, render_template, request,
                   url_for)
from gensim.summarization import summarize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
# from werkzeug import secure_filename
from nltk.stem.wordnet import WordNetLemmatizer

import pdf2txt as pdf
import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from Job_Match import Extractor

import xlwt
from xlwt import Workbook
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

print("AAAAAAAAAAAAAA")


class ResultElementTest:
    def __init__(self, rank, filename):
        self.rank = rank
        self.filename = filename
        print("^^^^^^^^^^^^^^^")

    def getfilepath(loc):
        temp = str(loc)
        temp = temp.replace('\\', '/')
        return temp

    def res(jobfile):
        Resume_Vector = []
        Ordered_list_Resume = []
        Ordered_list_Resume_Score = []
        LIST_OF_FILES = []
        LIST_OF_FILES_PDF = []
        LIST_OF_FILES_DOCX = []
        Resumes = []
        Temp_pdf = []
        count = 0
        wb = Workbook()
        # add_sheet is used to create sheet.
        sheet1 = wb.add_sheet('Sheet 1', 'False')
        sheet1.write(0, 0, "File Name")
        sheet1.write(0, 1, "Hard Skills")
        sheet1.write(0, 2, "Soft skills")
        sheet1.write(0, 3, "General")
        sheet1.write(0, 4, "Personality")
        os.chdir('D:\\5THYEAR\\FYP\\DataSet\\OFS\\1')
        # os.chdir('D:\\5THYEAR\\FYP\\DataSet\\OFS\\Test')
        print("**********88888")
        for file in glob.glob('**/*.pdf', recursive=True):
            LIST_OF_FILES_PDF.append(file)

        for file in glob.glob('**/*.docx', recursive=True):
            LIST_OF_FILES_DOCX.append(file)

        LIST_OF_FILES = LIST_OF_FILES_DOCX + LIST_OF_FILES_PDF
        # LIST_OF_FILES.remove("antiword.exe")
        print("This is LIST OF FILES")

        for num, i in enumerate(LIST_OF_FILES):
            Ordered_list_Resume.append(i)
            Temp = i.split(".")
            if Temp[1] == "pdf" or Temp[1] == "Pdf" or Temp[1] == "PDF":
                try:
                    print("This is PDF", Temp)
                    with open(i, 'rb') as pdf_file:
                        read_pdf = PyPDF2.PdfFileReader(pdf_file)

                        number_of_pages = read_pdf.getNumPages()
                        for page_number in range(number_of_pages):
                            page = read_pdf.getPage(page_number)
                            page_content = page.extractText()
                            page_content = page_content.replace('\n', ' ')
                            # page_content.replace("\r", "")
                            Temp_pdf = str(Temp_pdf) + str(page_content)
                            # Temp_pdf.append(page_content)
                            print("-----------------" + Temp_pdf)
                            row_list = [["SN", "Name", "Contribution"],
                                        [1, "Linus Torvalds", "Linux Kernel"],
                                        [2, "Tim Berners-Lee", "World Wide Web"],
                                        [3, "Guido van Rossum", "Python Programming"]]
                            # with open('D:\\5THYEAR\\FYP\\DataSet\\OFS\\ResumeCSV.csv', 'w', newline='') as file:
                            #     writer = csv.writer(file)
                            #
                            # writer.writerow(["SN", "Name", "Contribution"])
                            # writer.writerows(row_list)
                            # file.close()

                            with open('D:\\5THYEAR\\FYP\\DataSet\\OFS\\ResumeCSV.csv', 'a', newline='', encoding='utf-8') as file:
                                writer = csv.writer(file)
                                # writer.writerow(["SN", "Name", Temp_pdf])
                                # writer.writerow([Temp, Temp_pdf])
                                writer.writerow([Temp, Temp_pdf])
                        Resumes.extend([Temp_pdf])
                        f = open('D:\\5THYEAR\\FYP\\jobMatch\\my_resume2.txt','w', newline='', encoding='utf-8')
                        f.write(Temp_pdf)
                        print("-----------------" + Temp_pdf)
                        f.close()
                        Temp_pdf = ''
                        if __name__ != '__main__':
                            print("*******1"+str(num))
                            K = Extractor()

                            Extractor.__init__(K)
                            Extractor.makeTable(K,num+1,sheet1,Temp)
                            Extractor.sendToFile(K)
                            Extractor.printMeasures(K)

                        # f = open(str(i)+str("+") , 'w')
                        # f.write(page_content)
                        # f.close()
                except Exception as e:
                    print(e)

            if Temp[1] == "docx" or Temp[1] == "Docx" or Temp[1] == "DOCX":
                print("This is DOCX", i)
                try:
                    a = textract.process(i)
                    a = a.replace(b'\n', b' ')
                    a = a.replace(b'\r', b' ')
                    b = str(a)
                    c = [b]
                    Resumes.extend(c)

                    f = open('D:\\5THYEAR\\FYP\\jobMatch\\my_resume2.txt', 'w', newline='', encoding='utf-8')
                    f.write(b)
                    print("-----------------" + b)
                    f.close()

                    if __name__ != '__main__':
                        print("*******11"+str(num))
                        K = Extractor()

                        Extractor.__init__(K)
                        Extractor.makeTable(K, num + 1, sheet1, Temp)
                        Extractor.sendToFile(K)
                        Extractor.printMeasures(K)
                except Exception as e:
                    print(e)

        print("Done Parsing.")
        wb.save('D:\\5THYEAR\\FYP\\jobMatch\\xlwt example.xls')
        # Extractor.wb.save('D:\\5THYEAR\\FYP\\jobMatch\\xlwt example.xls')
        # Job_Desc = 0
        # LIST_OF_TXT_FILES = []
        # os.chdir('..\\')
        # f = open(jobfile, 'r')
        # text = f.read()
        # print("File : ."+text)
        # try:
        #     tttt = str(text)
        #     tttt = summarize(tttt, word_count=100)
        #     lemma = WordNetLemmatizer()
        #     normalized = lemma.lemmatize(tttt)
        #     # print("File : ^^^^ " + tttt)
        #     # count_vect = CountVectorizer()
        #     # X_train_counts = count_vect.fit_transform(tttt)
        #     # X_train_counts.shape
        #     text = [normalized]
        # except:
        #     text = 'None'

        # f.close()
        # vectorizer = TfidfVectorizer(stop_words='english')
        # print(text)
        # vectorizer.fit(text[0])
        # vector = vectorizer.transform(text[0])
        #
        # # tfidf_transformer = TfidfTransformer()
        # # X_train_tfidf = tfidf_transformer.fit_transform(text)
        # # X_train_tfidf.shape
        #
        # Job_Desc = vector.toarray()
        # # print("\n\n")
        # # print("This is vector : ",vector)
        # # print("This is job desc : ", Job_Desc)
        #
        # for i in Resumes:
        #     text = i
        #     tttt = str(text)
        #     # print("i *** : ",i)
        #     try:
        #         tttt = summarize(tttt, word_count=100)
        #         text = [tttt]
        #         vector = vectorizer.transform(text)
        #         # vector = tfidf_transformer.fit_transform(text)
        #         # print("VECTOR *** : ", text)
        #         aaa = vector.toarray()
        #         # print("VECTOR : ",aaa)
        #         Resume_Vector.append(vector.toarray())
        #     except:
        #         pass
        #     print("Resume_Vector")
        #     # print(Resume_Vector)
        #
        # for i in Resume_Vector:
        #     samples = i
        #     # print("SAMPLE : ",i)
        #     neigh = NearestNeighbors(n_neighbors=1)
        #     neigh.fit(samples)
        #     NearestNeighbors(algorithm='auto', leaf_size=30)
        #
        #     Ordered_list_Resume_Score.extend(neigh.kneighbors(Job_Desc)[0][0].tolist())
        #
        # Z = [x for _, x in sorted(zip(Ordered_list_Resume_Score, Ordered_list_Resume))]
        # print("Ordered_list_Resume =========================",Z)
        # print(Ordered_list_Resume)
        # print("Ordered_list_Resume_Score ",sorted(Ordered_list_Resume_Score))
        #
        # flask_return = []
        # # for n,i in enumerate(Z):
        # #     print("Rankkkkk\t" , n+1, ":\t" , i)
        #
        # for n, i in enumerate(Z):
        #     # print("Rank\t" , n+1, ":\t" , i)
        #     # flask_return.append(str("Rank\t" , n+1, ":\t" , i))
        #     name = ResultElement.getfilepath(i)
        #     # name = name.split('.')[0]
        #     # name = Ordered_list_Resume[n]
        #     rank = n + 1
        #     res = ResultElement(rank, name)
        #     flask_return.append(res)
        #     # res.printresult()
        #     print(f"Rank!!{res.rank + 1} :\t {res.filename}")
        # return flask_return
        #
        # if __name__ == '__main__':
        #     inputStr = input("")
        #     sear(inputStr)
