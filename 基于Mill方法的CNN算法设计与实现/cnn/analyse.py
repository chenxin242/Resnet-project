import matplotlib.pyplot as plt
import pandas as pd

fig = plt.figure(figsize=(10,6))

xlsx_pathA = "./Aoutput.xlsx"
xlsx_pathB = "./Boutput.xlsx"
xlsx_pathC = "./Dilated.xlsx"
xlsx_pathD = "./DilatedMill.xlsx"

dfA = pd.read_excel(xlsx_pathA, sheet_name='resnet110')
dfB = pd.read_excel(xlsx_pathB, sheet_name='resnet110')
dfC = pd.read_excel(xlsx_pathC, sheet_name='resnet110')
dfD = pd.read_excel(xlsx_pathD, sheet_name='resnet110')


dataA_acc = dfA['myeval_acc'].values
dataA_loss = dfA['myeval_loss'].values

dataB_acc = dfB['myeval_acc'].values
dataB_loss = dfB['myeval_loss'].values

dataC_acc = dfC['myeval_acc'].values
dataC_loss = dfC['myeval_loss'].values

dataD_acc = dfD['myeval_acc'].values
dataD_loss = dfD['myeval_loss'].values


epochA = [x+1 for x in range(dataA_acc.size)]
epochB = [y+1 for y in range(dataB_acc.size)]
epochC = [T+1 for T in range(dataC_acc.size)]
epochD = [W+1 for W in range(dataD_acc.size)]

ax1 = fig.add_subplot(111)

ax1.plot(epochA,
         dataA_acc,
         linestyle='-',
         color='red',
         label='ResNet110-Mill-loss'
         )

ax1.plot(epochB,
         dataB_acc,
         linestyle='--',
         color='blue',
         label='ResNet110'
         )

ax1.plot(epochC,
         dataC_acc,
         linestyle='-.',
         color='yellow',
         label='ResNet110-Dilated'
         )

ax1.plot(epochD,
         dataD_acc,
         linestyle=':',
         color='green',
         label='ResNet110-Dilated-Mill-loss'
         )

plt.xlabel('epoch')
ax1.set_ylabel("accuracy rate")
plt.legend()

ax2 = ax1.twinx()
ax2.plot(epochA,
         dataA_loss,
         linestyle='-',
         color='red',
         )

ax2.plot(epochB,
         dataB_loss,
         linestyle='--',
         color='blue',
         )

ax2.plot(epochC,
         dataC_loss,
         linestyle='-.',
         color='yellow',
         )

ax2.plot(epochD,
         dataD_loss,
         linestyle=':',
         color='green',
         )
ax2.set_ylabel("loss rate")

plt.show()


# dataB = dfB.values()
