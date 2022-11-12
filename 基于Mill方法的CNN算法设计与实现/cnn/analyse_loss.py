import matplotlib.pyplot as plt
import pandas as pd

xlsx_pathA = "./Aoutput.xlsx"
xlsx_pathB = "./Boutput.xlsx"

dfA = pd.read_excel(xlsx_pathA, sheet_name='resnet110')
dfB = pd.read_excel(xlsx_pathB, sheet_name='resnet110')

dataA_acc = dfA['myeval_acc'].values
dataA_loss = dfA['myeval_loss'].values

dataB_acc = dfB['myeval_acc'].values
dataB_loss = dfB['myeval_loss'].values

epochA = [x+1 for x in range(dataA_acc.size)]
epochB = [y+1 for y in range(dataB_acc.size)]

plt.plot(epochA,
         dataA_loss,
         linestyle='-',
         color='blue',
         label='loss',
         Marker='o',
         MarkerFaceColor='blue'
         )

plt.xlabel('epoch')
plt.legend()
plt.show()
