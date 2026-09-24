"""
=============================================================================
SafeRouteAI - Programa 4: Optimización Multicriterio con Sliders Interactivos
=============================================================================
Descripción:
    Este programa permite ajustar dinámicamente mediante controles deslizantes
    (sliders) los pesos de importancia de cada criterio en la toma de decisiones:
        Costo = α · d_norm + β · t_norm + γ · r_norm

    - α (Alfa):  Importancia de la Distancia Física (km).
    - β (Beta):  Importancia del Tiempo de Viaje (min).
    - γ (Gamma): Importancia de la Mitigación del Riesgo (índice sintético).

Normalización de Variables:
    Se utiliza escalado proporcional respecto al valor máximo de la red:
        x_norm = x / x_max
    Justificación técnica:
    El min-max clásico ((x - x_min) / (x_max - x_min)) asigna valor 0 al tramo
    más corto (1.0 km), lo cual causaría que recorrer dos calles de 1.0 km cueste
    0 + 0 = 0, rompiendo la desigualdad triangular frente a diagonales de 1.41 km.
    El escalado por el máximo preserva la estricta aditividad y las propiedades
    geométricas euclidianas manteniendo todos los factores en el rango [0, 1].

Manejo del Caso Extremo (α = β = γ = 0):
    Si todos los pesos se fijan en cero, no existe criterio de optimización.
    El algoritmo detecta de forma automática esta singularidad matemática,
    aplica una ponderación equitativa uniforme por defecto (α = β = γ = 1/3)
    y muestra una advertencia visual destacada en la interfaz.

Uso:
    python 04_optimizacion_multicriterio_sliders.py
=============================================================================
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from estilo_investigacion import configurar_estilo, finalizar_figura

configurar_estilo()
from matplotlib.widgets import Slider, Button

# ---------------------------------------------------------------------------
# 1. GENERACIÓN DE RED Y CÁLCULOS DE NORMALIZACIÓN
# ---------------------------------------------------------------------------
def campo_riesgo_sintetico(x, y):
    h1 = 18.0 * np.exp(-((x - 2.5) ** 2 + (y - 2.0) ** 2) / (2 * 0.75 ** 2))
    h2 = 12.0 * np.exp(-((x - 4.0) ** 2 + (y - 1.0) ** 2) / (2 * 0.65 ** 2))
    return 1.0 + h1 + h2


def construir_red_vial_normalizada(ancho=6, alto=5, semilla=42):
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

    distancias, tiempos, riesgos = [], [], []

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

        distancias.append(dist)
        tiempos.append(tiempo_min)
        riesgos.append(riesgo_tramo)

    d_max = max(distancias)
    t_max = max(tiempos)
    r_max = max(riesgos)

    # Asignación de valores normalizados respecto al máximo
    for u, v in G.edges():
        G[u][v]['d_norm'] = G[u][v]['distancia'] / d_max
        G[u][v]['t_norm'] = G[u][v]['tiempo'] / t_max
        G[u][v]['r_norm'] = G[u][v]['riesgo'] / r_max

    return G, d_max, t_max, r_max


# ---------------------------------------------------------------------------
# 2. FUNCIÓN DE COSTO COMBINADA Y OPTIMIZACIÓN
# ---------------------------------------------------------------------------
def calcular_ruta_multicriterio(G, origen, destino, alpha, beta, gamma):
    """
    Asigna costos ponderados a las aristas y resuelve la ruta óptima.
    Maneja el caso alpha=beta=gamma=0 aplicando respaldo uniforme.
    """
    es_caso_cero = (alpha == 0.0 and beta == 0.0 and gamma == 0.0)
    
    if es_caso_cero:
        w_d, w_t, w_r = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    else:
        suma = alpha + beta + gamma
        w_d, w_t, w_r = alpha / suma, beta / suma, gamma / suma

    for u, v in G.edges():
        c = w_d * G[u][v]['d_norm'] + w_t * G[u][v]['t_norm'] + w_r * G[u][v]['r_norm']
        G[u][v]['costo_compuesto'] = c

    ruta = nx.dijkstra_path(G, origen, destino, weight='costo_compuesto')
    
    dist_total = sum(G[u][v]['distancia'] for u, v in zip(ruta[:-1], ruta[1:]))
    tiempo_total = sum(G[u][v]['tiempo'] for u, v in zip(ruta[:-1], ruta[1:]))
    riesgo_total = sum(G[u][v]['riesgo'] for u, v in zip(ruta[:-1], ruta[1:]))
    costo_total = sum(G[u][v]['costo_compuesto'] for u, v in zip(ruta[:-1], ruta[1:]))

    return ruta, dist_total, tiempo_total, riesgo_total, costo_total, es_caso_cero, (w_d, w_t, w_r)


# ---------------------------------------------------------------------------
# 3. INTERFAZ GRÁFICA INTERACTIVA CON SLIDERS
# ---------------------------------------------------------------------------
def interfaz_multicriterio():
    G, d_max, t_max, r_max = construir_red_vial_normalizada()
    origen = 0
    destino = 29
    pos = nx.get_node_attributes(G, 'pos')

    # Valores iniciales de ponderación equilibrada
    alpha_init, beta_init, gamma_init = 0.33, 0.33, 0.34

    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(13.5, 9.0), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    plt.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.32)

    # Dibujo de la red vial base
    lineas_red = []
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        l, = ax.plot([x1, x2], [y1, y2], color='#b7c2cc', linewidth=1.5, zorder=1)
        lineas_red.append(l)

    # Nodos base
    ax.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()],
               s=70, c='#f1f4f6', edgecolors='#758391', linewidth=1.2, zorder=2)

    # Elementos dinámicos de la ruta
    linea_aura, = ax.plot([], [], color='#0072b2', linewidth=0, alpha=0, zorder=3)
    linea_ruta, = ax.plot([], [], color='#0072b2', linewidth=4.0, zorder=4, label='Ruta Óptima Multicriterio')
    scatter_puntos_ruta = ax.scatter([], [], s=110, color='#0072b2', edgecolors='#ffffff', zorder=5)

    # Marcadores de Origen y Destino
    ax.scatter(pos[origen][0], pos[origen][1], s=260, color='#007f62', edgecolors='#ffffff',
               linewidth=2.5, marker='o', zorder=7)
    ax.scatter(pos[destino][0], pos[destino][1], s=260, color='#b34428', edgecolors='#ffffff',
               linewidth=2.5, marker='s', zorder=7)
    ax.text(pos[origen][0] - 0.15, pos[origen][1] - 0.28, "ORIGEN", color='#007f62', fontsize=9.5, fontweight='bold', ha='center', zorder=8)
    ax.text(pos[destino][0] + 0.15, pos[destino][1] + 0.26, "DESTINO", color='#b34428', fontsize=9.5, fontweight='bold', ha='center', zorder=8)

    # Título general
    ax.set_title("SafeRouteAI - Optimización Multicriterio: Costo = α·d_norm + β·t_norm + γ·r_norm",
                 fontsize=14.5, fontweight='bold', color='#172b3a', pad=22)
    ax.text(0.5, 1.02, "Ajuste interactivo de pesos para explorar el compromiso entre Distancia, Tiempo y Riesgo",
            transform=ax.transAxes, ha='center', color='#52616e', fontsize=10.0)

    # Tarjetas informativas
    tarjeta_metricas = ax.text(0.02, 0.96, "", transform=ax.transAxes, va='top', ha='left',
                               fontsize=9.0, family='DejaVu Sans', color='#273746',
                               bbox=dict(boxstyle='square,pad=0.6', facecolor='#f1f4f6', edgecolor='#0072b2', alpha=0.95, linewidth=1.5), zorder=9)

    tarjeta_explicacion = ax.text(0.98, 0.96, "", transform=ax.transAxes, va='top', ha='right',
                                  fontsize=8.5, family='DejaVu Sans', color='#334155',
                                  bbox=dict(boxstyle='square,pad=0.6', facecolor='#f1f4f6', edgecolor='#758391', alpha=0.9), zorder=9)

    ax.set_xlabel("Coordenada Este-Oeste (km)", color='#334155', fontsize=10)
    ax.set_ylabel("Coordenada Norte-Sur (km)", color='#334155', fontsize=10)
    ax.set_xlim(-0.6, 5.7)
    ax.set_ylim(-0.6, 4.6)
    ax.tick_params(colors='#52616e', labelsize=8.5)
    ax.grid(True, linestyle=':', color='#f1f4f6', alpha=0.7)

    # -----------------------------------------------------------------------
    # CONTROLES DESLIZANTES (SLIDERS) Y BOTONES
    # -----------------------------------------------------------------------
    color_fondo_ctrl = '#f1f4f6'

    # Ejes de sliders
    ax_alpha = plt.axes([0.15, 0.20, 0.50, 0.025], facecolor=color_fondo_ctrl)
    ax_beta  = plt.axes([0.15, 0.15, 0.50, 0.025], facecolor=color_fondo_ctrl)
    ax_gamma = plt.axes([0.15, 0.10, 0.50, 0.025], facecolor=color_fondo_ctrl)

    slider_alpha = Slider(ax_alpha, 'α (Distancia) ', 0.0, 1.0, valinit=alpha_init, valstep=0.01, color='#0072b2')
    slider_beta  = Slider(ax_beta,  'β (Tiempo)    ', 0.0, 1.0, valinit=beta_init, valstep=0.01, color='#80558c')
    slider_gamma = Slider(ax_gamma, 'γ (Riesgo)    ', 0.0, 1.0, valinit=gamma_init, valstep=0.01, color='#b34428')

    for s in [slider_alpha, slider_beta, slider_gamma]:
        s.label.set_color('#273746')
        s.label.set_fontsize(9.5)
        s.label.set_fontweight('bold')
        s.valtext.set_color('#172b3a')
        s.valtext.set_fontsize(9.0)

    # Botón Reset
    ax_reset = plt.axes([0.72, 0.15, 0.11, 0.04])
    btn_reset = Button(ax_reset, 'Reiniciar', color='#f1f4f6', hovercolor='#b7c2cc')
    btn_reset.label.set_color('#ad6500')
    btn_reset.label.set_fontweight('bold')


    # Función central de actualización interactiva
    def actualizar_grafica(val=None):
        a = slider_alpha.val
        b = slider_beta.val
        g = slider_gamma.val

        ruta, d, t, r, c_tot, es_cero, (w_d, w_t, w_r) = calcular_ruta_multicriterio(G, origen, destino, a, b, g)

        # Actualizar geometría de la ruta
        xs = [pos[n][0] for n in ruta]
        ys = [pos[n][1] for n in ruta]
        linea_ruta.set_data(xs, ys)
        linea_aura.set_data(xs, ys)
        scatter_puntos_ruta.set_offsets(np.column_stack([xs, ys]))

        # Adaptar color según predominio de criterio
        if g >= a and g >= b:
            color_tema = '#007f62'  # Verde seguridad
        elif a >= g and a >= b:
            color_tema = '#0072b2'  # Cian distancia
        else:
            color_tema = '#80558c'  # Púrpura velocidad
        
        linea_ruta.set_color(color_tema)
        linea_aura.set_color(color_tema)
        scatter_puntos_ruta.set_color(color_tema)
        tarjeta_metricas.set_bbox(dict(boxstyle='square,pad=0.6', facecolor='#f1f4f6', edgecolor=color_tema, alpha=0.95, linewidth=1.5))

        # Texto informativo de resultados
        aviso_cero = "Pesos nulos: respaldo uniforme\n" if es_cero else ""
        info_metricas = (
            f"{aviso_cero}"
            "INDICADORES DE LA RUTA ACTUAL\n"
            "───────────────────────────────\n"
            f"• Distancia Total:  {d:.2f} km\n"
            f"• Tiempo Estimado:  {t:.1f} min\n"
            f"• Riesgo Acumulado: {r:.2f} pts*\n"
            f"• Costo Ponderado:  {c_tot:.3f}\n"
            f"• Nodos en Ruta:    {len(ruta)} intersecciones\n"
            "───────────────────────────────\n"
            f"Ponderación efectiva:\n"
            f"  Distancia: {w_d*100:.1f}% | Tiempo: {w_t*100:.1f}%\n"
            f"  Riesgo:    {w_r*100:.1f}%"
        )
        tarjeta_metricas.set_text(info_metricas)

        info_teorica = (
            "MODELO MATEMÁTICO SAFEROUTEAI\n"
            "─────────────────────────────\n"
            "• Escalamiento: x_norm = x / x_max\n"
            "  (máximo por arista; suma aditiva)\n"
            "• Si α = β = γ = 0:\n"
            "  Ponderación uniforme w = (1/3, 1/3, 1/3)\n"
            "• Parámetros del escenario sintético:\n"
            f"  d_max={d_max:.2f}km, t_max={t_max:.1f}min, r_max={r_max:.2f}"
        )
        tarjeta_explicacion.set_text(info_teorica)

        fig.canvas.draw_idle()

    slider_alpha.on_changed(actualizar_grafica)
    slider_beta.on_changed(actualizar_grafica)
    slider_gamma.on_changed(actualizar_grafica)

    def reiniciar(event):
        slider_alpha.reset()
        slider_beta.reset()
        slider_gamma.reset()

    btn_reset.on_clicked(reiniciar)

    actualizar_grafica()
    fig._controles = (slider_alpha, slider_beta, slider_gamma, btn_reset)
    finalizar_figura(fig, 4)
    plt.show()
    return fig


if __name__ == '__main__':
    interfaz_multicriterio()
