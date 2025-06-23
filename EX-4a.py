import pandas as pd

# Step 1: Load dataset
# Replace 'your_file.csv' with your actual file path
df = pd.read_csv('large_temperature_data.csv')

# Expected columns: 'Date', 'City', 'Temperature'
# Ensure Date is in datetime format
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month_name()

# Step 2: Group by City and Month, then sum temperatures
monthly_sum = df.groupby(['City', 'Month'])['Temperature'].sum().reset_index()

# Step 3: Pivot to get month-wise summary
pivot_df = monthly_sum.pivot(index='City', columns='Month', values='Temperature').fillna(0)

# Step 4: Calculate total summer temperature (June, July, August)
summer_months = ['June', 'July', 'August']
pivot_df['Total_Summer'] = pivot_df[summer_months].sum(axis=1)

# Step 5: Identify city with highest summer temperature
top_city = pivot_df['Total_Summer'].idxmax()
top_value = pivot_df['Total_Summer'].max()

# Step 6: Print the result
print("Month-wise temperature summary:")
print(pivot_df)

print(f"\nCity with highest summer temperature: {top_city} ({top_value})")
