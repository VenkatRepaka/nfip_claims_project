"""Execute the result-display notebook in a local Jupyter kernel."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
p=Path(__file__).resolve().parents[1]/'notebooks/10_prediction_audit.ipynb'
n=nbformat.read(p,as_version=4)
NotebookClient(n,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(p.parent)}}).execute()
nbformat.write(n,p)
print('Executed',p,'with',sum(c.cell_type=='code' for c in n.cells),'code cells and no errors.')
