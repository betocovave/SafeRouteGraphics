"""
=============================================================================
SafeRouteAI - Programa 1: Ruta Más Corta con Algoritmo de Dijkstra
=============================================================================
Descripción:
    Este programa modela una red vial urbana espacial coherente mediante un
    grafo no dirigido donde cada arista representa una calle con longitud física
    en kilómetros (km).

    Aplica el algoritmo clásico de Dijkstra para encontrar la ruta de menor
    distancia euclidiana entre el punto de Origen (Suroeste) y el Destino
    (Noreste). Muestra las distancias de cada tramo, resalta el camino óptimo
    y calcula el desglose completo de métricas (distancia, tiempo y riesgo).

Uso:
    python 01_dijkstra_distancia.py
=============================================================================
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from estilo_investigacion import configurar_estilo, finalizar_figura

configurar_estilo()

# ---------------------------------------------------------------------------
# 1. GENERACIÓN DE LA RED VIAL Y DATOS SINTÉTICOS (REPRODUCIBILIDAD FIJA)
# ---------------------------------------------------------------------------
def campo_riesgo_sintetico(x, y):
    """
    Evalúa el campo continuo de riesgo ficticio en las coordenadas (x, y).
    Simula dos focos de alta incidencia (p. ej. congestión severa o zona insegura).
    Retorna un índice adimensional no negativo (no es una probabilidad de asalto).
    """
    # Foco 1: Cruce céntrico (x=2.5, y=2.0)
    h1 = 18.0 * np.exp(-((x - 2.5) ** 2 + (y - 2.0) ** 2) / (2 * 0.75 ** 2))
    # Foco 2: Sector comercial este (x=4.0, y=1.0)
    h2 = 12.0 * np.exp(-((x - 4.0) ** 2 + (y - 1.0) ** 2) / (2 * 0.65 ** 2))
    return 1.0 + h1 + h2


def construir_red_vial(ancho=6, alto=5, semilla=42):
    """
    Construye una red urbana de 30 intersecciones en un área de 5x4 km.
    Incluye calles ortogonales y avenidas diagonales con velocidades diferenciadas.
    """
    np.random.seed(semilla)
    G = nx.Graph()

    # Creación de nodos con coordenadas espaciales fijas en km
    for y in range(alto):
        for x in range(ancho):
            nodo_id = y * ancho + x
            G.add_node(nodo_id, pos=(float(x), float(y)), nombre=f"N{nodo_id}")

    # Calles ortogonales (cuadrícula regular)
    for y in range(alto):
        for x in range(ancho):
            u = y * ancho + x
            if x + 1 < ancho:
                G.add_edge(u, y * ancho + (x + 1))
            if y + 1 < alto:
                G.add_edge(u, (y + 1) * ancho + x)

    # Avenidas diagonales (vías rápidas que conectan sectores estratégicos)
    diagonales = [
        (1, 0, 2, 1), (2, 1, 3, 2), (3, 2, 4, 3), (4, 3, 5, 4),
        (0, 2, 1, 3), (1, 3, 2, 4)
    ]
    for (x1, y1, x2, y2) in diagonales:
        u = y1 * ancho + x1
        v = y2 * ancho + x2
        G.add_edge(u, v)

    # Atributos de cada tramo (distancia, velocidad, tiempo y riesgo)
    for u, v in G.edges():
        x1, y1 = G.nodes[u]['pos']
        x2, y2 = G.nodes[v]['pos']
        dist = float(np.hypot(x2 - x1, y2 - y1))  # Distancia euclidiana en km
        
        # Avenidas tienen mayor límite de velocidad (50 km/h) vs calles secundarias (30 km/h)
        es_avenida = (abs(x2 - x1) == 1 and abs(y2 - y1) == 1) or (y1 == 2 and y2 == 2)
        velocidad = 50.0 if es_avenida else 30.0
        tiempo_min = float((dist / velocidad) * 60.0)

        # Riesgo del tramo: evaluado en el punto medio del segmento
        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        riesgo_tramo = float(campo_riesgo_sintetico(xm, ym) * dist)

        G[u][v]['distancia'] = dist
        G[u][v]['tiempo'] = tiempo_min
        G[u][v]['riesgo'] = riesgo_tramo
        G[u][v]['tipo'] = "Avenida" if es_avenida else "Calle"

    return G


# ---------------------------------------------------------------------------
# 2. CÁLCULO DE LA RUTA MÁS CORTA CON DIJKSTRA
# ---------------------------------------------------------------------------
def resolver_dijkstra_distancia(G, origen, destino):
    """
    Ejecuta el algoritmo de Dijkstra usando la 'distancia' como peso.
    Calcula los acumulados reales sumando los tramos de la ruta obtenida.
    """
    ruta = nx.dijkstra_path(G, source=origen, target=destino, weight='distancia')
    
    # Suma de tramos recorridos
    dist_total = sum(G[u][v]['distancia'] for u, v in zip(ruta[:-1], ruta[1:]))
    tiempo_total = sum(G[u][v]['tiempo'] for u, v in zip(ruta[:-1], ruta[1:]))
    riesgo_total = sum(G[u][v]['riesgo'] for u, v in zip(ruta[:-1], ruta[1:]))
    
    return ruta, dist_total, tiempo_total, riesgo_total


# ---------------------------------------------------------------------------
# 3. VISUALIZACIÓN GRÁFICA PROFESIONAL
# ---------------------------------------------------------------------------
def graficar_dijkstra():
    G = construir_red_vial()
    origen = 0     # Nodo en (0, 0)
    destino = 29   # Nodo en (5, 4)

    ruta, dist_tot, tiempo_tot, riesgo_tot = resolver_dijkstra_distancia(G, origen, destino)
    aristas_ruta = list(zip(ruta[:-1], ruta[1:]))
    set_aristas_ruta = set(aristas_ruta) | set((v, u) for u, v in aristas_ruta)

    # Configuración de figura con estilo limpio y elegante
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(13, 8.5), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    plt.subplots_adjust(bottom=0.14, top=0.90, left=0.08, right=0.96)

    pos = nx.get_node_attributes(G, 'pos')

    # Dibuja aristas secundarias (red base de la ciudad)
    for u, v in G.edges():
        if (u, v) not in set_aristas_ruta:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color='#b7c2cc', linewidth=1.4, linestyle='-', zorder=1)

    # Dibuja la ruta óptima de Dijkstra con resplandor y grosor destacado
    for u, v in aristas_ruta:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        # Línea de resplandor (aura)
        ax.plot([x1, x2], [y1, y2], color='#0072b2', linewidth=0, alpha=0, zorder=2)
        # Línea principal
        ax.plot([x1, x2], [y1, y2], color='#0072b2', linewidth=3.5, zorder=3)

    # Dibuja todos los nodos urbanos
    for nodo, (x, y) in pos.items():
        if nodo in ruta:
            color_nodo = '#0072b2'
            borde = '#ffffff'
            size = 130
        else:
            color_nodo = '#f1f4f6'
            borde = '#758391'
            size = 70
        ax.scatter(x, y, s=size, color=color_nodo, edgecolors=borde, linewidth=1.5, zorder=4)

    # Resalta Origen y Destino con pines visuales
    x_o, y_o = pos[origen]
    x_d, y_d = pos[destino]
    ax.scatter(x_o, y_o, s=260, color='#007f62', edgecolors='#ffffff', linewidth=2.5, marker='o', zorder=6, label='Origen')
    ax.scatter(x_d, y_d, s=280, color='#b34428', edgecolors='#ffffff', linewidth=2.5, marker='s', zorder=6, label='Destino')

    ax.text(x_o - 0.15, y_o - 0.37, "ORIGEN\n(0.0, 0.0)", color='#007f62', fontsize=9.5, fontweight='bold', ha='center', zorder=7)
    ax.text(x_d + 0.15, y_d + 0.26, "DESTINO\n(5.0, 4.0)", color='#b34428', fontsize=9.5, fontweight='bold', ha='center', zorder=7)

    # Etiquetas con distancias en los tramos de la ruta óptima
    for u, v in aristas_ruta:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        d_val = G[u][v]['distancia']
        ax.text(xm, ym + 0.09, f"{d_val:.2f} km", color='#273746', fontsize=8.5, fontweight='bold',
                ha='center', va='center', bbox=dict(boxstyle='square,pad=0.25', facecolor='#ffffff', edgecolor='#0072b2', alpha=0.9), zorder=5)

    # Títulos y subtítulos
    ax.set_title("SafeRouteAI - Optimización de Rutas: Algoritmo de Dijkstra", fontsize=15, fontweight='bold', color='#172b3a', pad=22)
    ax.text(0.5, 1.02, "Minimización estricta de distancia física en red de calles ortogonales y diagonales",
            transform=ax.transAxes, ha='center', color='#52616e', fontsize=10.5)

    # Cuadro flotante con desglose de métricas
    resumen_texto = (
        "MÉTRICAS DEL RECORRIDO ÓPTIMO\n"
        "───────────────────────────────\n"
        f"• Distancia total:   {dist_tot:.2f} km\n"
        f"• Tiempo estimado:   {tiempo_tot:.1f} min\n"
        f"• Riesgo acumulado:  {riesgo_tot:.2f} pts*\n"
        f"• Tramos recorridos: {len(aristas_ruta)} segmentos\n"
        "───────────────────────────────\n"
        "*Índice sintético de exposición\n"
        "  (no representa probabilidad real)"
    )
    ax.text(0.02, 0.96, resumen_texto, transform=ax.transAxes, va='top', ha='left',
            fontsize=9.5, family='DejaVu Sans', color='#273746',
            bbox=dict(boxstyle='square,pad=0.7', facecolor='#f1f4f6', edgecolor='#0072b2', alpha=0.92, linewidth=1.5), zorder=6)

    # Formato de cuadrícula y ejes espaciales
    ax.set_xlabel("Coordenada Este-Oeste (km)", color='#334155', fontsize=10.5, labelpad=8)
    ax.set_ylabel("Coordenada Norte-Sur (km)", color='#334155', fontsize=10.5, labelpad=8)
    ax.set_xlim(-0.6, 5.7)
    ax.set_ylim(-0.6, 4.6)
    ax.tick_params(colors='#52616e', labelsize=9)
    ax.grid(True, linestyle=':', color='#f1f4f6', alpha=0.7)

    # Disclaimer académico visible en pie de página
    ax.text(0.99, -0.10, "Simulación con datos sintéticos reproducibles (Semilla 42) - Fines didácticos SafeRouteAI",
            transform=ax.transAxes, ha='right', color='#52616e', fontsize=8.5, fontstyle='italic')



    finalizar_figura(fig, 1)
    plt.show()
    return fig


if __name__ == '__main__':
    graficar_dijkstra()
