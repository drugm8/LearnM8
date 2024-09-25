import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Data
cnn_score = [18.32, 21.9, 24.0, 27.06, 28.96, 30.36, 31.68, 33.30, 34.9, 36.52, 38.42]
gen_score = [10.5, 13.58, 16.56, 17.56, 20.1, 21.62, 23.0, 24.98, 26.74, 28.06, 30.06]
convex_plr = [19.92, 21.26, 22.56, 23.22, 24.22, 26.3, 29.84, 35.6, 39.2, 41.14, 43.68]

# Create x-axis values (active learning cycles)
cycles = range(1, len(cnn_score) + 1)

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(cycles, cnn_score, marker='o', label='CNN-Score')
plt.plot(cycles, gen_score, marker='s', label='GenScore-scoring')
plt.plot(cycles, convex_plr, marker='^', label='ConvexPLR')

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
with PdfPages('docking_scores_comparison.pdf') as pdf:
    pdf.savefig()
    plt.close()

print("The plot has been saved as 'docking_scores_comparison.pdf'")