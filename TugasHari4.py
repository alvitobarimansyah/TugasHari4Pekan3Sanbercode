# soal no 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('austin_weather.csv')
df.head()

def proses_data(series):
    series.replace('-', np.nan, inplace = True)
    series.fillna(method = 'ffill', inplace = True)
    return series.astype('float')

df['DewPointAvgF'] = proses_data(df['DewPointAvgF'])
df['HumidityAvgPercent'] = proses_data(df['HumidityAvgPercent'])
df['WindAvgMPH'] = proses_data(df['WindAvgMPH'])
df['TempAvgF'] = proses_data(df['TempAvgF'])

fig, ax = plt.subplots(figsize = (14, 8))
mapbar = ax.scatter(df['HumidityAvgPercent'], df['DewPointAvgF'], c = df['TempAvgF'], cmap = 'coolwarm', s = df['WindAvgMPH'] * 20)

fig.colorbar(mapbar)
ax.set_xlabel('HumidityAvg (%)')
ax.set_ylabel('DewPointAvg (F)')
ax.set_title('Austin Weather')
plt.show()

# soal no 2

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('vgsales.csv')
df.groupby('Genre')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].mean()

# soal no 3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('vgsales.csv')
genre = df.groupby('Genre')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].mean()
result = genre.transpose()

plt.style.use('seaborn')

fig, ax = plt.subplots(figsize = (12, 8))

x = np.arange(0, 16, 4)
y = result['Action']
y2 = result['Adventure']
y3 = result['Fighting']
y4 = result['Misc']
y5 = result['Platform']
y6 = result['Puzzle']
y7 = result['Racing']
y8 = result['Role-Playing']
y9 = result['Shooter']
y10 = result['Simulation']
y11 = result['Sports']
y12 = result['Strategy']

ax.bar(x, y, width = 0.2, color = 'r', label = 'Action')
ax.bar(x + 0.2, y2, width = 0.2, color = 'b', label = 'Adventure')
ax.bar(x + 0.4, y3, width = 0.2, color = 'purple', label = 'Fighting')
ax.bar(x + 0.6, y4, width = 0.2, color = 'gray', label = 'Misc')
ax.bar(x + 0.8, y5, width = 0.2, color = 'orange', label = 'Platform')
ax.bar(x + 1.0, y6, width = 0.2, color = 'g', label = 'Puzzle')
ax.bar(x + 1.2, y7, width = 0.2, color = 'pink', label = 'Racing')
ax.bar(x + 1.4, y8, width = 0.2, color = 'r', label = 'Role-Playing')
ax.bar(x + 1.6, y9, width = 0.2, color = 'b', label = 'Shooter')
ax.bar(x + 1.8, y10, width = 0.2, color = 'purple', label = 'Simulation')
ax.bar(x + 2.0, y11, width = 0.2, color = 'gray', label = 'Sports')
ax.bar(x + 2.2, y12, width = 0.2, color = 'orange', label = 'Strategy')

ax.set_xticks(x)
ax.set_xticklabels(result.index)

fig.legend()
ax.set_xlabel('Region Sales')
ax.set_ylabel('Mean Sales')
ax.set_title('Mean Sales Video Games By Genre')
plt.show()

# soal no 4

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('vgsales.csv')
genre = df.groupby('Genre')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].mean()
result = genre.transpose()

plt.style.use('seaborn')

fig, ax = plt.subplots(figsize = (12, 5))

x = np.arange(0, 16, 4)
y = result['Action']
y2 = result['Adventure']
y3 = result['Fighting']
y4 = result['Misc']
y5 = result['Platform']
y6 = result['Puzzle']
y7 = result['Racing']
y8 = result['Role-Playing']
y9 = result['Shooter']
y10 = result['Simulation']
y11 = result['Sports']
y12 = result['Strategy']
width = 3.0

ax.bar(x, y, width, color = 'r', label = 'Action')
ax.bar(x, y2, width, color = 'b', label = 'Adventure', bottom = y)
ax.bar(x, y3, width, color = 'purple', label = 'Fighting', bottom = y + y2)
ax.bar(x, y4, width, color = 'gray', label = 'Misc', bottom = y + y2 + y3)
ax.bar(x, y5, width, color = 'orange', label = 'Platform', bottom = y + y2 + y3 + y4)
ax.bar(x, y6, width, color = 'g', label = 'Puzzle', bottom = y + y2 + y3 + y4 + y5)
ax.bar(x, y7, width, color = 'pink', label = 'Racing', bottom = y + y2 + y3 + y4 + y5 + y6)
ax.bar(x, y8, width, color = 'r', label = 'Role-Playing', bottom = y + y2 + y3 + y4 + y5 + y6 + y7)
ax.bar(x, y9, width, color = 'b', label = 'Shooter', bottom = y + y2 + y3 + y4 + y5 + y6 + y7 + y8)
ax.bar(x, y10, width, color = 'purple', label = 'Simulation', bottom = y + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9)
ax.bar(x, y11, width, color = 'gray', label = 'Sports', bottom = y + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10)
ax.bar(x, y12, width, color = 'orange', label = 'Strategy', bottom = y + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10 + y11)

ax.set_xticks(x)
ax.set_xticklabels(result.index)

fig.legend()
plt.show()