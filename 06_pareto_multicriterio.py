"""
=============================================================================
SafeRouteAI - Programa 6: Gráfica de Alternativas y Frontera de Pareto
=============================================================================
Descripción:
    Genera un conjunto representativo de rutas candidatas entre Origen (0, 0)
    y Destino (5, 4) y proyecta sus métricas en el espacio de objetivos:
      - Eje X: Distancia Total recorrida (km).
      - Eje Y: Riesgo Acumulado (índice sintético adimensional).
      - Color: Tiempo de Viaje Estimado (minutos).

    Identifica algorítmicamente las alternativas NO DOMINADAS (Frontera de
    Pareto del conjunto muestral) y permite seleccionar interactivamente
    cualquier alternativa (mediante clic en la gráfica o botones) para
    visualizar su recorrido exacto sobre el mapa de la red vial en el panel derecho.

Nota Metodológica:
    La frontera de Pareto visualizada se evalúa formalmente sobre el conjunto
    de rutas candidatas generadas por el algoritmo (subconjunto muestral de
    caminos simples). Representa el conjunto de soluciones no dominadas
    dentro de este universo de exploración.

Uso:
    python 06_pareto_multicriterio.py
=============================================================================
"""

import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from estilo_investigacion import configurar_estilo, finalizar_figura

configurar_estilo()
import matplotlib.cm as cm
from matplotlib.widgets import Button

# ---------------------------------------------------------------------------
# 1. CONSTRUCCIÓN DE LA RED VIAL SINTÉTICA
# ---------------------------------------------------------------------------
def campo_riesgo_sintetico(x, y):
    h1 = 18.0 * np.exp(-((x - 2.5) ** 2 + (y - 2.0) ** 2) / (2 * 0.75 ** 2))
    h2 = 12.0 * np.exp(-((x - 4.0) ** 2 + (y - 1.0) ** 2) / (2 * 0.65 ** 2))
    return 1.0 + h1 + h2


def construir_red_vial(ancho=6, alto=5, semilla=42):
    np.random.seed(semilla)
    G = nx.Graph()

    for y in range(alto):
        for x in range(ancho):
            nodo_id = y * ancho + x
            G.add_node(nodo_id, pos=(float(x), float(y)), nombre=f"N{nodo_id}")

    for y in range(alto):
        for x in range(ancho):
            u = y * ancho + x
            if x + 1 < ancho:
                G.add_edge(u, y * ancho + (x + 1))
            if y + 1 < alto:
                G.add_edge(u, (y + 1) * ancho + x)

    diagonales = [
        (1, 0, 2, 1), (2, 1, 3, 2), (3, 2, 4, 3), (4, 3, 5, 4),
        (0, 2, 1, 3), (1, 3, 2, 4)
    ]
    for (x1, y1, x2, y2) in diagonales:
        u = y1 * ancho + x1
        v = y2 * ancho + x2
        G.add_edge(u, v)

    for u, v in G.edges():
        x1, y1 = G.nodes[u]['pos']
        x2, y2 = G.nodes[v]['pos']
        dist = float(np.hypot(x2 - x1, y2 - y1))
        es_avenida = (abs(x2 - x1) == 1 and abs(y2 - y1) == 1) or (y1 == 2 and y2 == 2)
        velocidad = 50.0 if es_avenida else 30.0
        tiempo_min = float((dist / velocidad) * 60.0)

        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        riesgo_tramo = float(campo_riesgo_sintetico(xm, ym) * dist)

        G[u][v]['distancia'] = dist
        G[u][v]['tiempo'] = tiempo_min
        G[u][v]['riesgo'] = riesgo_tramo

    return G


# ---------------------------------------------------------------------------
# 2. GENERACIÓN DE CANDIDATOS Y DETERMINACIÓN DEL FRENTE DE PARETO
# ---------------------------------------------------------------------------
def generar_rutas_candidatas_y_pareto(G, origen, destino, k_por_criterio=18):
    """
    Genera rutas candidatas diversas mediante k-caminos más cortos bajo
    distintos criterios (distancia, tiempo y riesgo) y analiza dominancia.
    """
    candidatos_map = {}

    for peso in ['distancia', 'tiempo', 'riesgo']:
        generador = nx.shortest_simple_paths(G, origen, destino, weight=peso)
        for p in itertools.islice(generador, k_por_criterio):
            clave = tuple(p)
            if clave not in candidatos_map:
                d = sum(G[u][v]['distancia'] for u, v in zip(p[:-1], p[1:]))
                t = sum(G[u][v]['tiempo'] for u, v in zip(p[:-1], p[1:]))
                r = sum(G[u][v]['riesgo'] for u, v in zip(p[:-1], p[1:]))
                candidatos_map[clave] = {
                    'id': len(candidatos_map) + 1,
                    'path': p,
                    'distancia': d,
                    'tiempo': t,
                    'riesgo': r
                }

    candidatos = list(candidatos_map.values())

    # Evaluación de dominancia de Pareto en el plano (Distancia, Riesgo)
    for c1 in candidatos:
        d1, r1 = c1['distancia'], c1['riesgo']
        dominada = False
        for c2 in candidatos:
            d2, r2 = c2['distancia'], c2['riesgo']
            # c2 domina a c1 si es menor o igual en ambos y estrictamente menor en al menos uno
            if (d2 <= d1 + 1e-10 and r2 <= r1 + 1e-10) and (d2 < d1 - 1e-10 or r2 < r1 - 1e-10):
                dominada = True
                break
        c1['es_pareto'] = not dominada

    return candidatos


# ---------------------------------------------------------------------------
# 3. INTERFAZ INTERACTIVA COORDINADA (PARETO + RED)
# ---------------------------------------------------------------------------
def visualizar_pareto():
    G = construir_red_vial()
    origen = 0
    destino = 29
    pos = nx.get_node_attributes(G, 'pos')
    candidatos = generar_rutas_candidatas_y_pareto(G, origen, destino)

    # Separación entre no dominadas (Pareto) y dominadas
    rutas_pareto = sorted([c for c in candidatos if c['es_pareto']], key=lambda x: x['distancia'])
    rutas_dominadas = [c for c in candidatos if not c['es_pareto']]

    # Estado de la ruta seleccionada actualmente
    indice_sel = [0]  # Inicia seleccionando la primera ruta del frente de Pareto

    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    fig, (ax_pareto, ax_grafo) = plt.subplots(1, 2, figsize=(16.5, 9.2), facecolor='#ffffff')
    plt.subplots_adjust(bottom=0.18, top=0.88, left=0.06, right=0.95, wspace=0.16)

    ax_pareto.set_facecolor('#ffffff')
    ax_grafo.set_facecolor('#ffffff')

    # -----------------------------------------------------------------------
    # PANEL IZQUIERDO: ESPACIO DE OBJETIVOS Y FRONTERA DE PARETO
    # -----------------------------------------------------------------------
    tiempos = [c['tiempo'] for c in candidatos]
    cmap_tiempo = plt.get_cmap('plasma')
    norm_tiempo = plt.Normalize(vmin=min(tiempos), vmax=max(tiempos))

    # 1. Dibujar rutas dominadas (círculos semitransparentes)
    d_dom = [c['distancia'] for c in rutas_dominadas]
    r_dom = [c['riesgo'] for c in rutas_dominadas]
    t_dom = [c['tiempo'] for c in rutas_dominadas]

    ax_pareto.scatter(d_dom, r_dom, c=t_dom, cmap=cmap_tiempo, norm=norm_tiempo,
                      s=70, alpha=1.0, edgecolors='#758391', linewidth=1.0, zorder=2, label='Alternativas Dominadas')

    # 2. Conectar el frente de Pareto con una curva escalonada / segmentada
    d_par = [c['distancia'] for c in rutas_pareto]
    r_par = [c['riesgo'] for c in rutas_pareto]
    t_par = [c['tiempo'] for c in rutas_pareto]

    ax_pareto.plot(d_par, r_par, color='#ad6500', linestyle='--', linewidth=2.0, alpha=0.85, zorder=3, label='Frontera de Pareto')

    # 3. Dibujar puntos Pareto (estrellas/diamantes destacados con borde dorado)
    ax_pareto.scatter(d_par, r_par, c=t_par, cmap=cmap_tiempo, norm=norm_tiempo,
                      s=180, marker='D', edgecolors='#ad6500', linewidth=2.0, zorder=4, label='Óptimos de Pareto (No Dominadas)')

    # Marcador dinámico del punto seleccionado en el scatter plot
    scatter_sel = ax_pareto.scatter([], [], s=320, facecolors='none', edgecolors='#0072b2',
                                    linewidth=3.0, zorder=6, label='Selección Activa')

    # Ejes y textos del panel Pareto
    ax_pareto.set_title("Espacio de Compensación: Distancia vs. Riesgo Acumulado", fontsize=12, fontweight='bold', color='#172b3a', pad=12)
    ax_pareto.set_xlabel("Distancia Total Recorrida (km)", color='#334155', fontsize=10)
    ax_pareto.set_ylabel("Riesgo Acumulado (Índice Sintético)", color='#334155', fontsize=10)
    ax_pareto.tick_params(colors='#52616e', labelsize=8.5)
    ax_pareto.grid(True, linestyle=':', color='#b7c2cc', alpha=0.6)

    # Barra de color horizontal para Tiempo de Viaje
    cax = fig.add_axes([0.08, 0.08, 0.35, 0.02])
    sm = cm.ScalarMappable(cmap=cmap_tiempo, norm=norm_tiempo)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label('Tiempo de Viaje Estimado (minutos)', color='#273746', fontsize=9.0)
    cb.ax.tick_params(colors='#52616e', labelsize=8.0)

    ax_pareto.legend(loc='upper right', facecolor='#f1f4f6', edgecolor='#b7c2cc',
                     labelcolor='#273746', fontsize=8.0, framealpha=0.92)

    # -----------------------------------------------------------------------
    # PANEL DERECHO: RED VIAL URBANA Y RUTA SELECCIONADA
    # -----------------------------------------------------------------------
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax_grafo.plot([x1, x2], [y1, y2], color='#b7c2cc', linewidth=1.4, zorder=1)

    for nodo, (x, y) in pos.items():
        ax_grafo.scatter(x, y, s=50, color='#f1f4f6', edgecolors='#758391', linewidth=1.0, zorder=2)

    # Elementos dinámicos de la ruta en el grafo
    linea_ruta_aura, = ax_grafo.plot([], [], color='#0072b2', linewidth=0, alpha=0, zorder=3)
    linea_ruta_activa, = ax_grafo.plot([], [], color='#0072b2', linewidth=3.8, zorder=4)
    puntos_ruta_activa = ax_grafo.scatter([], [], s=100, color='#0072b2', edgecolors='#ffffff', zorder=5)

    # Origen y Destino
    ax_grafo.scatter(pos[origen][0], pos[origen][1], s=250, color='#007f62', edgecolors='#ffffff',
                     linewidth=2.2, marker='o', zorder=7)
    ax_grafo.scatter(pos[destino][0], pos[destino][1], s=260, color='#b34428', edgecolors='#ffffff',
                     linewidth=2.2, marker='s', zorder=7)
    ax_grafo.text(pos[origen][0] - 0.15, pos[origen][1] - 0.28, "ORIGEN", color='#007f62', fontsize=9, fontweight='bold', ha='center', zorder=8)
    ax_grafo.text(pos[destino][0] + 0.15, pos[destino][1] + 0.26, "DESTINO", color='#b34428', fontsize=9, fontweight='bold', ha='center', zorder=8)

    ax_grafo.set_title("Recorrido de la Alternativa Seleccionada en la Red", fontsize=12, fontweight='bold', color='#172b3a', pad=12)
    ax_grafo.set_xlabel("Coordenada X (km)", color='#334155', fontsize=10)
    ax_grafo.set_ylabel("Coordenada Y (km)", color='#334155', fontsize=10)
    ax_grafo.set_xlim(-0.6, 5.7)
    ax_grafo.set_ylim(-0.6, 4.6)
    ax_grafo.tick_params(colors='#52616e', labelsize=8.5)
    ax_grafo.grid(True, linestyle=':', color='#b7c2cc', alpha=0.5)

    # Tarjeta con desglose de la alternativa seleccionada
    tarjeta_info = ax_grafo.text(0.03, 0.96, "", transform=ax_grafo.transAxes, va='top', ha='left',
                                 fontsize=8.8, family='DejaVu Sans', color='#273746',
                                 bbox=dict(boxstyle='square,pad=0.6', facecolor='#f1f4f6', edgecolor='#0072b2', alpha=0.95, linewidth=1.5), zorder=9)

    # Título General y Nota Metodológica
    fig.suptitle("SafeRouteAI - Análisis Multicriterio: Alternativas Candidatas y Frontera de Pareto",
                 fontsize=15, fontweight='bold', color='#172b3a', y=0.96)

    fig.text(0.5, 0.02,
             f"Pareto en distancia y exposición: {len(candidatos)} rutas únicas (hasta 18 por criterio). El tiempo se representa por color.\nNo dominancia evaluada solo en esta muestra; el trazo discontinuo sirve de guía visual.",
             ha='center', color='#52616e', fontsize=8.5, fontstyle='italic')

    # -----------------------------------------------------------------------
    # ACTUALIZACIÓN Y SELECCIÓN DE RUTAS
    # -----------------------------------------------------------------------
    todos_candidatos = rutas_pareto + rutas_dominadas

    def actualizar_seleccion(idx):
        c = todos_candidatos[idx]
        d, r, t, p = c['distancia'], c['riesgo'], c['tiempo'], c['path']
        es_p = c['es_pareto']

        # Actualizar marcador en scatter plot
        scatter_sel.set_offsets([[d, r]])
        color_borde = '#ad6500' if es_p else '#0072b2'
        scatter_sel.set_edgecolors(color_borde)

        # Actualizar trazo en el grafo
        xs = [pos[n][0] for n in p]
        ys = [pos[n][1] for n in p]
        color_ruta = '#007f62' if es_p else '#0072b2'
        linea_ruta_activa.set_data(xs, ys)
        linea_ruta_activa.set_color(color_ruta)
        linea_ruta_aura.set_data(xs, ys)
        linea_ruta_aura.set_color(color_ruta)
        puntos_ruta_activa.set_offsets(np.column_stack([xs, ys]))
        puntos_ruta_activa.set_color(color_ruta)

        etiqueta_estado = "[*] OPTIMA DE PARETO (No dominada)" if es_p else "• DOMINADA (Existe otra mejor)"
        info = (
            f"ALTERNATIVA #{c['id']} / {len(todos_candidatos)}\n"
            f"Estado: {etiqueta_estado}\n"
            "───────────────────────────────\n"
            f"• Distancia Total:  {d:.2f} km\n"
            f"• Tiempo Estimado:  {t:.1f} min\n"
            f"• Riesgo Acumulado: {r:.2f} pts*\n"
            f"• Segmentos:        {len(p) - 1} tramos\n"
            "───────────────────────────────\n"
            "Haz clic en cualquier punto del panel\n"
            "izquierdo para inspeccionar su ruta."
        )
        tarjeta_info.set_text(info)
        tarjeta_info.set_bbox(dict(boxstyle='square,pad=0.6', facecolor='#f1f4f6', edgecolor=color_ruta, alpha=0.95, linewidth=1.5))
        fig.canvas.draw_idle()

    # Selección mediante clic en el panel de dispersión
    def al_hacer_clic(event):
        if event.inaxes == ax_pareto and event.xdata is not None and event.ydata is not None:
            # Encuentra el candidato más cercano en distancia euclidiana escalada
            x_click, y_click = event.xdata, event.ydata
            mejor_dist = float('inf')
            mejor_idx = 0
            for i, c in enumerate(todos_candidatos):
                # Distancia euclidiana normalizada al clic
                px, py = ax_pareto.transData.transform((c['distancia'], c['riesgo']))
                dx, dy = px - event.x, py - event.y
                dist_click = dx ** 2 + dy ** 2
                if dist_click < mejor_dist:
                    mejor_dist = dist_click
                    mejor_idx = i
            indice_sel[0] = mejor_idx
            actualizar_seleccion(mejor_idx)

    fig.canvas.mpl_connect('button_press_event', al_hacer_clic)

    # Botones de navegación
    ax_prev = plt.axes([0.55, 0.08, 0.08, 0.038])
    btn_prev = Button(ax_prev, '< Anterior', color='#f1f4f6', hovercolor='#b7c2cc')
    btn_prev.label.set_color('#334155')
    btn_prev.label.set_fontsize(8.5)

    ax_next = plt.axes([0.64, 0.08, 0.08, 0.038])
    btn_next = Button(ax_next, 'Siguiente >', color='#f1f4f6', hovercolor='#b7c2cc')
    btn_next.label.set_color('#334155')
    btn_next.label.set_fontsize(8.5)

    ax_pareto_btn = plt.axes([0.73, 0.08, 0.11, 0.038])
    btn_pareto = Button(ax_pareto_btn, 'Siguiente Pareto', color='#f1f4f6', hovercolor='#b7c2cc')
    btn_pareto.label.set_color('#ad6500')
    btn_pareto.label.set_fontsize(8.5)
    btn_pareto.label.set_fontweight('bold')


    def paso_prev(event):
        indice_sel[0] = (indice_sel[0] - 1) % len(todos_candidatos)
        actualizar_seleccion(indice_sel[0])

    def paso_next(event):
        indice_sel[0] = (indice_sel[0] + 1) % len(todos_candidatos)
        actualizar_seleccion(indice_sel[0])

    def paso_pareto(event):
        # Busca el siguiente índice que sea Pareto
        for offset in range(1, len(todos_candidatos) + 1):
            cand_idx = (indice_sel[0] + offset) % len(todos_candidatos)
            if todos_candidatos[cand_idx]['es_pareto']:
                indice_sel[0] = cand_idx
                actualizar_seleccion(cand_idx)
                break

    btn_prev.on_clicked(paso_prev)
    btn_next.on_clicked(paso_next)
    btn_pareto.on_clicked(paso_pareto)

    # Iniciar con la primera ruta Pareto seleccionada
    actualizar_seleccion(0)
    fig._controles = (btn_prev, btn_next, btn_pareto, actualizar_seleccion)
    finalizar_figura(fig, 6)
    plt.show()
    return fig


if __name__ == '__main__':
    visualizar_pareto()
