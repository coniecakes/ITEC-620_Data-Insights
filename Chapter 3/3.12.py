import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

GasPrices = pd.read_excel('/Users/coniecakes/PycharmProjects/ITEC-620_Data-Insights/Chapter 3/GasPrices.xlsx')

print(GasPrices)

summary_stats = GasPrices.describe()
print(summary_stats)

#myModel = ols('Price ($)~C(Month)', data=GasPrices).fit()
#anova_table = anova_lm(myModel)
#print(anova_table)

x = np.array(GasPrices['Month'])
y = np.array(GasPrices['Price ($)'])

plt.plot(x, y, label = 'Gas Prices', marker ='*')
coefficients = np.polyfit(x, y, 1)  # 1 represents linear regression (degree 1)
trendline = np.polyval(coefficients, x)
plt.plot(x, trendline, label='Trendline', color='red')
plt.xlabel('Month')
plt.ylabel('Price ($)')
plt.title('Gas Prices Over Time')
plt.legend()
plt.show()

