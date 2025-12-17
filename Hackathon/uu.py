import pandas as pd
update = []
with open('cardio_train.csv','r+') as file:
    data = file.readlines()
    for line in data:
        newline = line.strip().split(';')
        newline = ','.join(newline)
        update.append(newline+'\n')
    file.seek(0)
    file.writelines(update)
        

