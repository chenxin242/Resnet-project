import matplotlib.pyplot as plt
import pandas as pd

xlsx_pathA = "./BasedLeNetMill.xlsx"

dfA = pd.read_excel(xlsx_pathA, sheet_name='lenet1')
dfB = pd.read_excel(xlsx_pathA, sheet_name='lenet2')

dataA_acc = dfA['acc50'].values
dataB_acc = dfA['acc1000'].values

dataC_acc = dfB['accmill'].values

epochA = [x+1 for x in range(dataA_acc.size)]
epochB = [x+1 for x in range(dataB_acc.size)]
epochC = [y+1 for y in range(dataC_acc.size)]


plt.plot(epochA,
         dataA_acc,
         linestyle='-',
         color='red',
         label='LeNet'
         )

plt.plot(epochB,
         dataB_acc,
         linestyle='--',
         color='blue',
         label='LeNet'
         )

plt.plot(epochC,
         dataC_acc,
         linestyle='-.',
         color='black',
         label='LeNet-Mill-loss'
         )


plt.xlabel('epoch')
plt.ylabel('accuracy rate')
plt.legend()
plt.show()


# dataB = dfB.values()