import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('scripts/dataclean/orders_cleaned.csv')
print("Cleaned Data Shape:",df.shape)
print(df.columns)
sales_category=df.groupby('Category')['Sales'].sum()
print("\n Sales by Category:\n",sales_category)
sales_category.plot(kind='bar',title='Total Sales by Category')
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.show()
sales_region=df.groupby('Region')['Sales'].sum()
print("\n Sales by Region:\n",sales_region)
sales_region.plot(kind="bar",title='Sales Distribution by Region')
plt.show()