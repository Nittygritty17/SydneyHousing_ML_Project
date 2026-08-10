import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
df = pd.read_csv('data/cleaned_suburbs.csv')

# 1. Price distribution by suburb
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.boxplot(data=df, x='suburb', y='sale_price', ax=axes[0])
axes[0].set_title('Sale Price Distribution by Suburb')
axes[0].set_ylabel('Sale Price ($)')
sns.histplot(df['sale_price'], bins=30, ax=axes[1], kde=True)
axes[1].set_title('Overall Sale Price Distribution (right-skewed)')
plt.tight_layout()
plt.savefig('figures/price_distribution.png', dpi=130)
plt.close()

# 2. Price vs building size, coloured by suburb
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=df, x='building_size_m2', y='sale_price', hue='suburb', alpha=0.7, ax=ax)
ax.set_title('Sale Price vs Building Size')
plt.tight_layout()
plt.savefig('figures/price_vs_building_size.png', dpi=130)
plt.close()

# 3. Price trend over time
fig, ax = plt.subplots(figsize=(8, 5))
tmp = df.dropna(subset=['sale_year'])
sns.lineplot(data=tmp.groupby(['sale_year', 'suburb'])['sale_price'].median().reset_index(),
             x='sale_year', y='sale_price', hue='suburb', marker='o', ax=ax)
ax.set_title('Median Sale Price by Year and Suburb')
plt.tight_layout()
plt.savefig('figures/price_trend.png', dpi=130)
plt.close()

# 4. Correlation heatmap of numeric features
num_cols = ['sale_price','bedrooms','bathrooms','car_spaces','land_size_m2',
            'building_size_m2','property_age','Storeys','Pool','renovated',
            'desc_word_count','luxury_word_count']
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.savefig('figures/correlation_heatmap.png', dpi=130)
plt.close()

# 5. Property type counts by suburb
fig, ax = plt.subplots(figsize=(7, 5))
sns.countplot(data=df, x='suburb', hue='property_type', ax=ax)
ax.set_title('Property Type Counts by Suburb')
plt.tight_layout()
plt.savefig('figures/property_type_counts.png', dpi=130)
plt.close()

print("Correlations with sale_price:")
print(df[num_cols].corr()['sale_price'].sort_values(ascending=False))
