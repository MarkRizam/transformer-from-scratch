"""Execute clean notebooks and keep executed copies outside the source directory."""
from pathlib import Path
import json
import nbformat
from nbclient import NotebookClient

def main():
    root=Path(__file__).resolve().parents[1]; out=root/'results'/'executed-notebooks'; out.mkdir(parents=True,exist_ok=True); status=[]
    for p in sorted((root/'notebooks').glob('*.ipynb')):
        n=nbformat.read(p,as_version=4); NotebookClient(n,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute(); nbformat.write(n,out/p.name); status.append({'notebook':p.name,'status':'pass'}); print(p.name,'PASS',flush=True)
    (root/'results'/'notebook-status.json').write_text(json.dumps(status,indent=2)+'\n'); assert len(status)==8
if __name__=='__main__': main()
