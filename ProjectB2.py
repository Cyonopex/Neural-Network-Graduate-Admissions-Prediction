import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#admit_data = np.genfromtxt('admission_predict.csv', delimiter= ',')
#raw_data = admit_data[1:,1:]
#print(raw_data)
raw_data = pd.read_csv('admission_predict.csv')

data = raw_data.drop("Serial No.", axis=1)

corr = data.corr()

sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

plt.show()