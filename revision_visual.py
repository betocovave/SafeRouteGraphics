import os, sys, runpy
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path('graficas').resolve()))
plt.show=lambda:None
out=Path('graficas/revision'); out.mkdir(exist_ok=True)
for p in sorted(Path('graficas').glob('[0-9]*.py')):
    runpy.run_path(str(p),run_name='__main__')
    fig=plt.gcf(); fig.canvas.draw()
    fig.savefig(out/(p.stem+'.png'),dpi=110)
    print(p.name, 'OK', len(fig.axes),'ejes')
    plt.close('all')
