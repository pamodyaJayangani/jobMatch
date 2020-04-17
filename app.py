from FileWriter import ResultElementTest
#
# print("BBBBBBBBBBBBB")
# flask_return = ResultElementTest.res("tesla_job_description.txt")
# print("CCCCCCCCCCCCC")

from flask import Flask, render_template
import xlrd

app = Flask(__name__)

@app.route('/')
def hello_world():
   return render_template('index.html')

@app.route('/similarityView')
def similarityView():
   loc = ("D:\\5THYEAR\\FYP\\jobMatch\\xlwt example.xls")
   wb = xlrd.open_workbook(loc)
   sheet = wb.sheet_by_index(0)

   # For row 0 and column 0
   sheet.cell_value(0, 0)
   resume_similarity = []
   for i in range(sheet.nrows):
      resume_similarity.append(sheet.row_values(i))
   return render_template('similarityView.html',data = resume_similarity, len=len(resume_similarity))


@app.route('/generate')
def generate():
   flask_return = ResultElementTest.res("tesla_job_description.txt")
   return render_template('index.html')

if __name__ == '__main__':
   app.run()