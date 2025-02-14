import os
import numpy as np

# Define languages
languages = ['de', 'en', 'fr', 'it', 'ja', 'kor', 'sr', 'sv', 'th', 'vi', 'zh', 'yue', 'zh-by-char', 'yue-by-char']

# # Define bin edges (ensuring 0-1 range is separate)
# bins = [0.5, 1.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] + list(range(200, 1101, 100)) + [float('inf')]
# 
# # Define bin labels
# bin_labels = ["1", "2-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100"] + \
#              [f"{i+1}-{i+100}" for i in range(100, 1100, 100)] + ["1001+"]

bins = [0.5, 1.5, 10, 20, 30, 40, 50, 100, float('inf')]

# Define bin labels
bin_labels = ["1", "2-10", "11-20", "21-30", "31-40", "41-50", "51-100", ">100"]

# Ensure label count matches bin count - 1
assert len(bin_labels) == len(bins) - 1, "Mismatch between bins and labels!"

# Output file
output_file = "word_occurrence_summary.csv"

# Initialize data storage
summary_counts = {label: [] for label in bin_labels}
summary_counts["unique_tokens"] = []
summary_counts["total_tokens"] = []

# Process each language file
for lang in languages:
    file_path = f"./frequency_counts/{lang}.txt"

    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File not found for {lang}: {file_path}")
        for label in summary_counts:
            summary_counts[label].append(0)
        continue

    # Read occurrences
    occurrences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                try:
                    occurrences.append(int(line.split(": ")[1].strip()))
                except ValueError:
                    print(f"⚠️ Skipping invalid line in {lang}: {line.strip()}")

    # Compute unique and total tokens
    unique_tokens = len(occurrences)
    total_tokens = sum(occurrences)

    # Handle empty file scenario
    if not occurrences:
        print(f"⚠️ No valid data in {lang}, filling with zeros.")
        for label in summary_counts:
            summary_counts[label].append(0)
        continue

    # Bin occurrences
    counts, _ = np.histogram(occurrences, bins=bins)

    # Store counts safely
    for i, label in enumerate(bin_labels):
        summary_counts[label].append(counts[i])

    # Store unique and total token counts
    summary_counts["unique_tokens"].append(unique_tokens)
    summary_counts["total_tokens"].append(total_tokens)

# Write results to file
with open(output_file, "w", encoding="utf-8") as f:
    # Write header
    f.write("Range\t" + "\t".join(languages) + "\n")

    # Write data
    for label in summary_counts:
        f.write(label + "\t" + "\t".join(map(str, summary_counts[label])) + "\n")

print(f"✅ Summary saved to {output_file}")