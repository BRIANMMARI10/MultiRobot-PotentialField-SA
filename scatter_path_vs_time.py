import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("simulation_results_initorient.csv")
df['N'] = df['Simulation'].str.extract(r'N=(\d+)').astype(int)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['Path Length'], df['Time Taken'], c=df['N'], cmap='viridis', alpha=0.8)
plt.colorbar(scatter, label='Number of Robots (N)')
plt.xlabel('Path Length')
plt.ylabel('Time Taken')
plt.title('Path Length vs Time Taken Colored by N')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/scatter_path_vs_time.png')

