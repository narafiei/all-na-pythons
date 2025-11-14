# Exersize 1 ----------------
import pandas as pd
# import time
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Read csv -------------------------
plt.close()
df=pd.read_csv('homeprice.csv')
print(df)
plt.title = 'Price by Area'
plt.xplt = 'Area'
plt.yplt = 'Price'
plt.scatter(df.area, df.price, color= 'red', marker='+')
x= df[['area']]
y= df.price

reg=LinearRegression()
reg.fit(df[['area']],df.price)
# draw line of predict --------------
plt.plot(x, reg.predict(x), color ='blue')

plt.show()
# testing   ----------------------
#p= reg.predict([[3300], [1000], [5000]])
m= reg.coef_
b= reg.intercept_
result2 = m * 3300 + b 
print('r2', result2)

scor = reg.score(x,y)
print(scor)

print(df)
d=pd.read_csv('areas.csv')
p= reg.predict(d)

# import to file --------------
d['Predict']= p
d.to_csv('Prediction_file.csv', index=False)
print(d)