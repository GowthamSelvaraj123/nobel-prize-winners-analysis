# Nobel Prize Winners Full Analysis
# Complete Code with Matplotlib, Plotly, and Seaborn Visualizations

# ------------------------------
# Step 1: Import Libraries
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ------------------------------
# Step 2: Load Dataset
# ------------------------------
df_data = pd.read_csv('nobel_prize_data.csv')

# ------------------------------
# Step 3: Data Cleaning / Preparation
# ------------------------------

# Convert birth_date to datetime
df_data['birth_date'] = pd.to_datetime(df_data['birth_date'], errors='coerce')

# Calculate share percentage from 'prize_share' column (e.g., 1/2 -> 50%)
split_share = df_data.prize_share.str.split('/', expand=True)
df_data['share_pct'] = pd.to_numeric(split_share[0]) / pd.to_numeric(split_share[1]) * 100

# ------------------------------
# Part 1: Matplotlib Trends Over Time
# ------------------------------

# Challenge 1: Number of Prizes Awarded Over Time
prize_per_year = df_data.groupby('year').count()['prize']
moving_average = prize_per_year.rolling(window=5).mean()

plt.figure(figsize=(16,8), dpi=200)
plt.title('Number of Nobel Prizes Awarded per Year', fontsize=18)
plt.xticks(ticks=np.arange(1900,2021,5), fontsize=14, rotation=45)
plt.yticks(fontsize=14)
ax = plt.gca()
ax.set_xlim(1900,2020)
ax.scatter(prize_per_year.index, prize_per_year.values, c='dodgerblue', alpha=0.7, s=100)
ax.plot(prize_per_year.index, moving_average.values, c='crimson', linewidth=3)
plt.show()

# Challenge 2: Prize Share Over Time
yearly_avg_share = df_data.groupby('year')['share_pct'].mean()
share_moving_average = yearly_avg_share.rolling(window=5).mean()

plt.figure(figsize=(16,8), dpi=200)
plt.title('Number of Prizes and Average Share Over Time', fontsize=18)
plt.xticks(ticks=np.arange(1900,2021,5), fontsize=14, rotation=45)
plt.yticks(fontsize=14)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.invert_yaxis()
ax1.scatter(prize_per_year.index, prize_per_year.values, c='dodgerblue', alpha=0.7, s=100)
ax1.plot(prize_per_year.index, moving_average.values, c='crimson', linewidth=3)
ax2.plot(prize_per_year.index, share_moving_average.values, c='grey', linewidth=3)
plt.show()

# ------------------------------
# Part 2: Top Countries & Choropleth
# ------------------------------

# Top 20 Countries
top_countries = df_data.groupby('birth_country_current', as_index=False).agg({'prize':'count'})
top_countries.sort_values('prize', inplace=True)
top20_countries = top_countries[-20:]

px.bar(x=top20_countries.prize, y=top20_countries.birth_country_current,
       orientation='h', color=top20_countries.prize, color_continuous_scale='Viridis',
       title='Top 20 Countries by Number of Prizes').update_layout(xaxis_title='Number of Prizes', yaxis_title='Country', coloraxis_showscale=False).show()

# Choropleth Map
df_countries = df_data.groupby(['birth_country_current','ISO'], as_index=False).agg({'prize':'count'})
px.choropleth(df_countries, locations='ISO', color='prize', hover_name='birth_country_current',
              color_continuous_scale=px.colors.sequential.matter, title='Global Distribution of Nobel Prizes').update_layout(coloraxis_showscale=True).show()

# Category Breakdown by Country
cat_country = df_data.groupby(['birth_country_current','category'], as_index=False).agg({'prize':'count'})
merged_df = pd.merge(cat_country, top20_countries, on='birth_country_current')
merged_df.columns = ['birth_country_current','category','cat_prize','total_prize']
merged_df.sort_values('total_prize', inplace=True)

px.bar(x=merged_df.cat_prize, y=merged_df.birth_country_current, color=merged_df.category,
       orientation='h', title='Top 20 Countries by Number of Prizes and Category').update_layout(xaxis_title='Number of Prizes', yaxis_title='Country').show()

# Cumulative Prizes Over Time
prize_by_year = df_data.groupby(['birth_country_current','year'], as_index=False).count()[['birth_country_current','year','prize']]
cumulative_prizes = prize_by_year.groupby(['birth_country_current','year']).sum().groupby(level=[0]).cumsum().reset_index()

px.line(cumulative_prizes, x='year', y='prize', color='birth_country_current', hover_name='birth_country_current',
        title='Cumulative Nobel Prizes by Country Over Time').update_layout(xaxis_title='Year', yaxis_title='Number of Prizes').show()

# ------------------------------
# Part 3: Research Institutions and Cities
# ------------------------------

# Top 20 Organisations
top20_orgs = df_data.organization_name.value_counts()[:20]
top20_orgs.sort_values(ascending=True, inplace=True)
px.bar(x=top20_orgs.values, y=top20_orgs.index, orientation='h', color=top20_orgs.values,
       color_continuous_scale=px.colors.sequential.haline,
       title='Top 20 Research Institutions by Number of Prizes').update_layout(xaxis_title='Number of Prizes', yaxis_title='Institution', coloraxis_showscale=False).show()

# Top 20 Organization Cities
top20_org_cities = df_data.organization_city.value_counts()[:20]
top20_org_cities.sort_values(ascending=True, inplace=True)
px.bar(x=top20_org_cities.values, y=top20_org_cities.index, orientation='h', color=top20_org_cities.values,
       color_continuous_scale=px.colors.sequential.Plasma,
       title='Which Cities Do the Most Research?').update_layout(xaxis_title='Number of Prizes', yaxis_title='City', coloraxis_showscale=False).show()

# Top 20 Birth Cities
top20_birth_cities = df_data.birth_city.value_counts()[:20]
top20_birth_cities.sort_values(ascending=True, inplace=True)
px.bar(x=top20_birth_cities.values, y=top20_birth_cities.index, orientation='h', color=top20_birth_cities.values,
       color_continuous_scale=px.colors.sequential.Plasma,
       title='Where were the Nobel Laureates Born?').update_layout(xaxis_title='Number of Prizes', yaxis_title='City of Birth', coloraxis_showscale=False).show()

# Sunburst Chart: Organization Country -> City -> Organization
country_city_org = df_data.groupby(['organization_country','organization_city','organization_name'], as_index=False).agg({'prize':'count'})
country_city_org.sort_values('prize', ascending=False, inplace=True)
px.sunburst(country_city_org, path=['organization_country','organization_city','organization_name'], values='prize', title='Where do Discoveries Take Place?').update_layout(coloraxis_showscale=False).show()

# ------------------------------
# Part 4: Laureate Age Analysis
# ------------------------------

# Calculate Winning Age
df_data['winning_age'] = df_data.year - df_data.birth_date.dt.year

# Oldest & Youngest Winners
print("Oldest Winner:")
print(df_data.nlargest(1, 'winning_age'))
print("Youngest Winner:")
print(df_data.nsmallest(1, 'winning_age'))

# Histogram of Winning Age
plt.figure(figsize=(8,4), dpi=200)
sns.histplot(df_data['winning_age'], bins=30)
plt.xlabel('Age')
plt.title('Distribution of Age on Receipt of Prize')
plt.show()

# Winning Age Over Time
plt.figure(figsize=(8,4), dpi=200)
sns.regplot(data=df_data, x='year', y='winning_age', lowess=True, scatter_kws={'alpha':0.4}, line_kws={'color':'black'})
plt.show()

# Age Differences by Category (Boxplot)
plt.figure(figsize=(8,4), dpi=200)
sns.boxplot(data=df_data, x='category', y='winning_age')
plt.show()

# Age Over Time by Category (lmplot)
sns.lmplot(data=df_data, x='year', y='winning_age', row='category', lowess=True, aspect=2, scatter_kws={'alpha':0.6}, line_kws={'color':'black'})
plt.show()

# Combined lmplot by Category using hue
sns.lmplot(data=df_data, x='year', y='winning_age', hue='category', lowess=True, aspect=2, scatter_kws={'alpha':0.5}, line_kws={'linewidth':5})
plt.show()
