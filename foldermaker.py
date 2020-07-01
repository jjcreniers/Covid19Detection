import openpyxl
import os
import shutil
import sys

path = 'E:/CheXpert-v1.0-small/train_nofindings_AP.xlsx'
wb = openpyxl.load_workbook(filename = path)
ws = wb['Sheet1']

for index, cell in enumerate(ws['A']):
        if index != 0:
            source = "E:/" + cell.value
            target = "E:/CheXpert-v1.0-small/trainsplit/nofinding_" +  str(index) + ".jpg"
            print(str(index))
            # adding exception handling
            try:
                shutil.copy(source, target)
            except IOError as e:
                print("Unable to copy file. %s" % e)
                exit(1)
            except:
                print("Unexpected error:", sys.exc_info())
                exit(1)