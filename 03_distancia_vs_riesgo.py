"""
=============================================================================
SafeRouteAI - Programa 3: Ruta Más Corta frente a Ruta con Menor Riesgo
=============================================================================
Descripción:
    Compara en dos paneles interactivos paralelos la misma red vial urbana
    bajo dos filosofías de optimización opuestas:
      - Panel Izquierdo: Optimización exclusiva de DISTANCIA (ignora el riesgo).
      - Panel Derecho:   Optimización exclusiva de RIESGO (ignora kilometraje).

    Todas las calles están coloreadas según su nivel de exposición al riesgo
    mediante una escala cromática calibrada. Al pie de la visualización se
    presenta una tabla comparativa exhaustiva que cuantifica el compromiso
    (trade-off) entre distancia adicional y mitigación de riesgo.

Uso:
    python 03_distancia_vs_riesgo.py
=============================================================================
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from estilo_investigacion import configurar_estilo, finalizar_figura

configurar_estilo()
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# 1. CONSTRUCCIÓN DE LA RED VIAL Y CÁLCULO DE CAMPOS SINTÉTICOS
# ---------------------------------------------------------------------------
def campo_riesgo_sintetico(x, y):
    """
    Campo de riesgo sintético con 2 focos críticos.
    Índice aditivo no negativo (fines didácticos, no probabilidad probabilística).
    """
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
# 2. CÁLCULO DE AMBAS RUTAS Y MÉTRICAS ACUMULADAS
# ---------------------------------------------------------------------------
def calcular_metricas_ruta(G, ruta):
    dist_total = sum(G[u][v]['distancia'] for u, v in zip(ruta[:-1], ruta[1:]))
    tiempo_total = sum(G[u][v]['tiempo'] for u, v in zip(ruta[:-1], ruta[1:]))
    riesgo_total = sum(G[u][v]['riesgo'] for u, v in zip(ruta[:-1], ruta[1:]))
    return dist_total, tiempo_total, riesgo_total


# ---------------------------------------------------------------------------
# 3. VISUALIZACIÓN COMPARATIVA EN DOS PANELES
# ---------------------------------------------------------------------------
def graficar_comparacion():
    G = construir_red_vial()
    origen = 0
    destino = 29
    pos = nx.get_node_attributes(G, 'pos')

    # Cálculo algorítmico independiente
    ruta_dist = nx.dijkstra_path(G, origen, destino, weight='distancia')
    d1, t1, r1 = calcular_metricas_ruta(G, ruta_dist)

    ruta_riesgo = nx.dijkstra_path(G, origen, destino, weight='riesgo')
    d2, t2, r2 = calcular_metricas_ruta(G, ruta_riesgo)

    # Configuración de mapa de color para las calles según su riesgo
    riesgos_todos = [G[u][v]['riesgo'] for u, v in G.edges()]
    norm = mcolors.Normalize(vmin=min(riesgos_todos), vmax=max(riesgos_todos))
    cmap = plt.get_cmap('cividis')

    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 9.2), facecolor='#ffffff')
    plt.subplots_adjust(bottom=0.25, top=0.88, left=0.05, right=0.95, wspace=0.15)

    paneles = [
        (ax1, ruta_dist, "A) RUTA MÁS CORTA (Criterio Distancia)",
         "#0072b2", "-", "o", f"Distancia: {d1:.2f} km\nTiempo: {t1:.1f} min\nRiesgo: {r1:.2f} pts"),
        (ax2, ruta_riesgo, "B) MENOR EXPOSICIÓN SINTÉTICA",
         "#007f62", "--", "D", f"Distancia: {d2:.2f} km\nTiempo: {t2:.1f} min\nRiesgo: {r2:.2f} pts")
    ]

    for ax, ruta, subtitulo, col_ruta, estilo_ruta, marcador, txt_resumen in paneles:
        ax.set_facecolor('#ffffff')
        set_ruta = set(zip(ruta[:-1], ruta[1:])) | set(zip(ruta[1:], ruta[:-1]))

        # Dibujar aristas de la red coloreadas por riesgo
        for u, v in G.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            color_arista = cmap(norm(G[u][v]['riesgo']))
            ax.plot([x1, x2], [y1, y2], color=color_arista, linewidth=2.0, alpha=1.0, zorder=1)

        # Resaltar la ruta seleccionada (diferenciada por color, estilo de línea y marcador)
        xs = [pos[n][0] for n in ruta]
        ys = [pos[n][1] for n in ruta]
        # Resplandor
        ax.plot(xs, ys, color=col_ruta, linewidth=0, alpha=0, zorder=3)
        # Línea principal
        ax.plot(xs, ys, color=col_ruta, linewidth=3.5, linestyle=estilo_ruta, zorder=4, label='Ruta Seleccionada')
        # Marcadores de nodos a lo largo de la ruta
        ax.plot(xs, ys, color=col_ruta, marker=marcador, markersize=8, linestyle='None', zorder=5)

        # Nodos urbanos base
        for nodo, (x, y) in pos.items():
            ax.scatter(x, y, s=55, color='#f1f4f6', edgecolors='#758391', linewidth=1.2, zorder=2)

        # Origen y Destino
        ax.scatter(pos[origen][0], pos[origen][1], s=250, color='#007f62', edgecolors='#ffffff',
                   linewidth=2.2, marker='o', zorder=6)
        ax.scatter(pos[destino][0], pos[destino][1], s=260, color='#b34428', edgecolors='#ffffff',
                   linewidth=2.2, marker='s', zorder=6)

        ax.text(pos[origen][0] - 0.15, pos[origen][1] - 0.28, "ORIGEN", color='#007f62', fontsize=9, fontweight='bold', ha='center', zorder=7)
        ax.text(pos[destino][0] + 0.15, pos[destino][1] + 0.26, "DESTINO", color='#b34428', fontsize=9, fontweight='bold', ha='center', zorder=7)

        # Títulos de panel
        ax.set_title(subtitulo, fontsize=12, fontweight='bold', color='#273746', pad=12)
        ax.set_xlabel("Coordenada X (km)", color='#52616e', fontsize=9.5)
        ax.set_ylabel("Coordenada Y (km)", color='#52616e', fontsize=9.5)
        ax.set_xlim(-0.6, 5.7)
        ax.set_ylim(-0.6, 4.6)
        ax.tick_params(colors='#52616e', labelsize=8.5)
        ax.grid(True, linestyle=':', color='#b7c2cc', alpha=0.5)

        # Tarjeta métrica individual en panel
        ax.text(0.03, 0.96, txt_resumen, transform=ax.transAxes, va='top', ha='left',
                fontsize=9.0, family='DejaVu Sans', color='#273746',
                bbox=dict(boxstyle='square,pad=0.5', facecolor='#f1f4f6', edgecolor=col_ruta, alpha=0.92, linewidth=1.5), zorder=8)

    # Barra de color común para el riesgo de las calles
    cax = fig.add_axes([0.30, 0.20, 0.40, 0.02])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label('Índice Sintético de Riesgo del Tramo (Bajo → Alto)', color='#273746', fontsize=9.5, labelpad=5)
    cb.ax.tick_params(colors='#52616e', labelsize=8.5)

    # Título General y Subtítulo
    fig.suptitle("SafeRouteAI - Análisis Comparativo: Distancia Mínima vs. Menor Riesgo Acumulado",
                 fontsize=15, fontweight='bold', color='#172b3a', y=0.96)

    # TABLA COMPARATIVA INFERIOR
    delta_d = ((d2 - d1) / d1) * 100.0
    delta_t = ((t2 - t1) / t1) * 100.0
    delta_r = ((r2 - r1) / r1) * 100.0

    ax_tabla = fig.add_axes([0.15, 0.04, 0.70, 0.12])
    ax_tabla.axis('off')

    datos_tabla = [
        ["Métrica de Evaluación", "Ruta Más Corta", "Ruta Menor Riesgo", "Impacto / Trade-off"],
        ["Distancia Total (km)", f"{d1:.2f} km", f"{d2:.2f} km", f"+{delta_d:.1f}% (+{d2 - d1:.2f} km)"],
        ["Tiempo Estimado (min)", f"{t1:.1f} min", f"{t2:.1f} min", f"+{delta_t:.1f}% (+{t2 - t1:.1f} min)"],
        ["Riesgo Acumulado (pts)*", f"{r1:.2f} pts", f"{r2:.2f} pts", f"{delta_r:.1f}% (Mitigación del {abs(delta_r):.1f}%)"]
    ]

    tabla = ax_tabla.table(cellText=datos_tabla, loc='center', cellLoc='center')
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9.0)
    tabla.scale(1.0, 1.35)

    # Estilos de celdas de la tabla
    for (i, j), cell in tabla.get_celld().items():
        cell.set_edgecolor('#b7c2cc')
        if i == 0:
            cell.set_facecolor('#f1f4f6')
            cell.set_text_props(weight='bold', color='#0072b2')
        else:
            cell.set_facecolor('#ffffff' if i % 2 == 0 else '#f7f8fa')
            if j == 3:
                cell.set_text_props(weight='bold', color='#007f62' if i == 3 else '#b34428')
            else:
                cell.set_text_props(color='#273746')



    finalizar_figura(fig, 3)
    plt.show()
    return fig


if __name__ == '__main__':
    graficar_comparacion()
