"""Validación numérica y de callbacks sin abrir ventanas."""
import os, sys, importlib.util
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
plt.show=lambda:None
modules=[]
for p in sorted(Path(__file__).parent.glob('[0-9]*.py')):
    spec=importlib.util.spec_from_file_location(p.stem,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);modules.append(m)
a=modules[1];G=a.construir_red_vial()
assert (len(G),G.number_of_edges())==(30,55)
for start in G:
    for end in G:
        states=a.ejecutar_pasos_astar(G,start,end)
        assert np.isclose(states[-1]['g'],nx.dijkstra_path_length(G,start,end,weight='distancia'))
        for u,v in G.edges:
            assert a.heuristica_euclidiana(G.nodes[u]['pos'],G.nodes[end]['pos']) <= G[u][v]['distancia']+a.heuristica_euclidiana(G.nodes[v]['pos'],G.nodes[end]['pos'])+1e-10
fig=a.animar_astar();anim,button,update,pause=fig._controles
update(len(a.ejecutar_pasos_astar(G,0,29))-1)
assert len(fig.axes[0].lines[55].get_xdata())>0
update(0);assert len(fig.axes[0].lines[55].get_xdata())==0
pause(None);assert button.label.get_text()=='Reanudar'
pause(None);assert button.label.get_text()=='Pausar'
m=modules[3];Gn,*_=m.construir_red_vial_normalizada()
for weights,key in [((1,0,0),'distancia'),((0,1,0),'tiempo'),((0,0,1),'riesgo')]:
    path,*rest=m.calcular_ruta_multicriterio(Gn,0,29,*weights)
    assert np.isclose(sum(Gn[u][v][key] for u,v in zip(path,path[1:])),nx.dijkstra_path_length(Gn,0,29,weight=key))
assert m.calcular_ruta_multicriterio(Gn,0,29,0,0,0)[5]
fig=m.interfaz_multicriterio()
for slider in fig._controles[:3]:slider.set_val(0)
fig.canvas.draw()
for slider in fig._controles[:3]:slider.reset()
m=modules[5];candidates=m.generar_rutas_candidatas_y_pareto(G,0,29)
assert len({tuple(c['path']) for c in candidates})==len(candidates)
for c in candidates:
    dominated=any(o['distancia']<=c['distancia']+1e-10 and o['riesgo']<=c['riesgo']+1e-10 and (o['distancia']<c['distancia']-1e-10 or o['riesgo']<c['riesgo']-1e-10) for o in candidates)
    assert c['es_pareto'] != dominated
fig=m.visualizar_pareto()
for i in range(len(candidates)):fig._controles[-1](i)
fig.canvas.draw()
print(f'OK: A* frente a Dijkstra en 900 pares; heurística consistente; pausa/reanudación/reinicio de animación; pesos extremos y nulos; sliders; {len(candidates)} candidatos Pareto y selección.')
plt.close('all')
