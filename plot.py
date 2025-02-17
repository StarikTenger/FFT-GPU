import pandas as pd
import sys
import matplotlib.pyplot as plt

# Read the CSV file
if len(sys.argv) != 2:
    print("Usage: python graph.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]
df = pd.read_csv(csv_file)

# Group by algorithm and vector size, then calculate the average elapsed time
grouped_df = df.groupby(['algorithm', 'vector_size']).mean().reset_index()

# Pivot the dataframe to have vector sizes as index and algorithms as columns
pivot_df = grouped_df.pivot(index='vector_size', columns='algorithm', values='elapsed_time')

# Plot the data
pivot_df.plot(kind='line', marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Vector Size')
plt.ylabel('Average Elapsed Time, ms')
#plt.title('Average Elapsed Time of Each Algorithm Depending on Vector Size')
plt.legend(title='Algorithm')
plt.grid(True)

# Set x-ticks to be the vector sizes
plt.xticks(pivot_df.index, pivot_df.index, rotation=45)

plt.show()