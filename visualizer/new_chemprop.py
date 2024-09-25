cnn_cp = [19.84, 25.46, 28.74, 31.22, 33.800000000000004, 35.92, 38.86, 40.739999999999995, 42.46, 44.940000000000005, 46.339999999999996]
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
cnn_rf = [18.279999999999998, 20.86, 23.76, 25.06, 27.139999999999997, 28.499999999999996, 29.86, 31.06, 32.74, 35.54, 38.48]
# Data

# Create x-axis values (active learning cycles)
cycles = range(1, len(cnn_rf) + 1)

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(cycles, cnn_cp, marker='o', label='Chemprop CNN-Score')
plt.plot(cycles, cnn_rf, marker='s', label='Random Forest Regressor CNN-Score')


# Customize the plot
plt.title('Comparison of Docking Scores Across Active Learning Cycles')
plt.xlabel('Active Learning Cycle')
plt.ylabel('Top 5000 of 5000 percentage')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Set x-axis ticks to show all cycles
plt.xticks(cycles)

# Adjust layout
plt.tight_layout()

# Save the plot as a PDF file
with PdfPages('one_docking_score_new_cp.pdf') as pdf:
    pdf.savefig()
    plt.close()

print("The plot has been saved as 'docking_scores_comparison.pdf'")