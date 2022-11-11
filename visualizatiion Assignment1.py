
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing excel file with specifics
file = (r'C:\Users\dj\Downloads/2019-2021 IGR  OF STATES.xlsx')
data = pd.read_excel(file, skiprows=1, skipfooter=5, usecols=[1,2,3,5,6,8,9])
data.columns = ['STATE','2019 TAX REVENUE','2019 OTHER REVENUE','2020 TAX REVENUE','2020 OTHER REVENUE','2021 TAX REVENUE','2021 OTHER REVENUE']

#we want to show the top 10 states in terms of tax revenues for the three years
tax_2019 = data[['STATE','2019 TAX REVENUE']]
top_10_tax_2019 = tax_2019.sort_values('2019 TAX REVENUE',ascending=False,).head(10)
top_10_tax_2019.columns = ['STATE','TAX REVENUE']

tax_2020 = data[['STATE','2020 TAX REVENUE']]
top_10_tax_2020 = tax_2020.sort_values('2020 TAX REVENUE',ascending=False,).head(10)
top_10_tax_2020.columns = ['STATE','TAX REVENUE']

tax_2021 = data[['STATE','2021 TAX REVENUE']]
top_10_tax_2021 = tax_2021.sort_values('2021 TAX REVENUE',ascending=False,).head(10)
top_10_tax_2021.columns = ['STATE','TAX REVENUE']

top_10_tax = [top_10_tax_2019, top_10_tax_2020,top_10_tax_2021]
titles = ['2019', '2020', '2021']

def bar_subplots(top_10_tax,title):
    
    plt.figure(figsize=(30,10))
    for i in range(len(top_10_tax)):
        plt.subplot(1,3,i+1).set_title(title[i])
        plt.bar(top_10_tax[i]['STATE'],top_10_tax[i]['TAX REVENUE'])
        plt.xticks(fontsize=11,rotation=90)
        plt.savefig("bar_subplots.png")
    plt.show()
    return

bar_subplots(top_10_tax,titles)

#LINE PLOTS
file2 = r'C:\Users\dj\Downloads/Nigeria geopolitical zones.xlsx'
geopolitical_zones = pd.read_excel(file2)
geopolitical_zones.columns = ['STATE', 'GEOPOLITICAL ZONES']

new_data = pd.merge(data,geopolitical_zones,how='left', on = 'STATE')

geo_zone_group = new_data.groupby('GEOPOLITICAL ZONES').sum()

geo_zone_group['2019 REVENUE'] = geo_zone_group['2019 TAX REVENUE']+geo_zone_group['2019 OTHER REVENUE']
geo_zone_group['2020 REVENUE'] = geo_zone_group['2020 TAX REVENUE']+geo_zone_group['2020 OTHER REVENUE']
geo_zone_group['2021 REVENUE'] = geo_zone_group['2021 TAX REVENUE']+geo_zone_group['2021 OTHER REVENUE']

geo_total_revenue = geo_zone_group[['2019 REVENUE', '2020 REVENUE', '2021 REVENUE']]

geo_total_revenue = geo_total_revenue.T
columns = geo_total_revenue.columns
label = np.arange(len(geo_total_revenue.columns))
plt.figure()
for i,j in zip (columns, label):
    plt.plot(geo_total_revenue.index,geo_total_revenue[i],label=geo_total_revenue.columns[j])
plt.legend()
plt.savefig("Line plot.png")
plt.show()

#
label = ['Abia','Rivers', 'Sokoto', 'Taraba','Yobe','Zamfara']
Tax = [8.967911e+09, 1.159110e+11, 1.772678e+10, 3.409557e+09, 7.022992e+09, 1.274936e+10,]
explodes = [0,0,0,0,0,0]
title = ('2019 States Revenue')

def create_pie_chart(Tax,explodes,label):
    plt.pie(Tax, labels=label, explode=explodes, autopct='%.3f%%', shadow=True)
    plt.title('2019 States Revenue')
    plt.savefig("Pie_chart.png")
    plt.show()

create_pie_chart(Tax,explodes,label)