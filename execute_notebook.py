"""
Execute notebook to populate all outputs and plots directly inside the .ipynb file.
"""
import nbformat
from nbclient import NotebookClient

print("Loading notebook...")
with open("MACSE504_Project_Question7_26MML0020_Rohan_Yadav.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="python3")
print("Executing notebook cells (computing outputs, plots, and benchmarks)...")
client.execute()

with open("MACSE504_Project_Question7_26MML0020_Rohan_Yadav.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

with open("MNIST_PCA_LogisticRegression_CaseStudy7.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook execution completed successfully! All cell outputs and plots saved.")
