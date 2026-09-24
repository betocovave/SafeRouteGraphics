"""
=============================================================================
SafeRouteAI - Programa 2: Búsqueda de Rutas con Algoritmo A* (Animación)
=============================================================================
Descripción:
    Este programa visualiza de forma interactiva y animada la exploración
    paso a paso del algoritmo A* (A-Star) sobre la red vial urbana de SafeRouteAI.

    Diferencia visualmente en tiempo real:
      1. Nodos no visitados (gris oscuro).
      2. Nodos pendientes en el conjunto abierto / frontera (naranja ámbar).
      3. Nodos ya explorados en el conjunto cerrado (azul violáceo).
      4. Nodo actual en expansión con sus valores g(n), h(n) y f(n).
      5. Ruta óptima final reconstruida (verde esmeralda brillante).

Heurística Admisible:
    h(n) = ||pos(n) - pos(destino)||_2 (distancia euclidiana en línea recta).
    En el plano euclidiano, la línea recta es la distancia geodésica mínima
    posible entre dos puntos. Toda red vial física requiere una distancia
    mayor o igual a la línea recta (costo real >= h(n)). En consecuencia,
    h(n) jamás sobreestima el costo restante, cumpliendo la condición de
    admisibilidad (h(n) <= h*(n)) y garantizando la optimalidad de la solución.

Uso:
    python 02_astar_animacion.py
=============================================================================
"""

import heapq
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from estilo_investigacion import configurar_estilo, finalizar_figura

configurar_estilo()
import matplotlib.animation as animation
from matplotlib.widgets import Button

# ---------------------------------------------------------------------------
# 1. CONSTRUCCIÓN DE LA RED VIAL SINTÉTICA (COHERENTE Y REPRODUCIBLE)
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
# 2. GENERADOR PASO A PASO DEL ALGORITMO A*
# ---------------------------------------------------------------------------
def heuristica_euclidiana(pos_a, pos_b):
    """Calcula la distancia euclidiana en línea recta (admisible y consistente)."""
    return float(np.hypot(pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]))


def ejecutar_pasos_astar(G, origen, destino):
    """
    Genera la secuencia completa de estados de la ejecución de A*.
    Cada estado almacena:
      - nodo_actual
      - open_set (conjunto de nodos pendientes en frontera)
      - closed_set (conjunto de nodos visitados/fijados)
      - g_scores, f_scores
      - ruta_final (None hasta que se alcanza el destino)
    """
    pos = nx.get_node_attributes(G, 'pos')
    
    # Priority Queue: almacena tuplas (f_score, contador, nodo)
    pq = []
    contador = 0
    
    g_score = {nodo: float('inf') for nodo in G.nodes()}
    f_score = {nodo: float('inf') for nodo in G.nodes()}
    predecesores = {}
    
    g_score[origen] = 0.0
    h_ini = heuristica_euclidiana(pos[origen], pos[destino])
    f_score[origen] = h_ini
    heapq.heappush(pq, (f_score[origen], contador, origen))
    
    open_set = {origen}
    closed_set = set()
    historial = []

    # Estado inicial
    historial.append({
        'nodo_actual': origen,
        'open_set': set(open_set),
        'closed_set': set(closed_set),
        'g': g_score[origen],
        'h': h_ini,
        'f': f_score[origen],
        'ruta_final': None,
        'mensaje': "Inicio de búsqueda A*: Nodo origen añadido a la frontera"
    })

    destino_encontrado = False

    while pq:
        f_curr, _, actual = heapq.heappop(pq)
        
        if actual not in open_set:
            continue
            
        open_set.remove(actual)
        closed_set.add(actual)

        g_act = g_score[actual]
        h_act = heuristica_euclidiana(pos[actual], pos[destino])
        
        historial.append({
            'nodo_actual': actual,
            'open_set': set(open_set),
            'closed_set': set(closed_set),
            'g': g_act,
            'h': h_act,
            'f': f_curr,
            'ruta_final': None,
            'mensaje': f"Expandiendo nodo {actual}: g={g_act:.2f} km, h={h_act:.2f} km, f={f_curr:.2f} km"
        })

        if actual == destino:
            destino_encontrado = True
            break

        # Explorar vecinos
        for vecino in G.neighbors(actual):
            if vecino in closed_set:
                continue

            costo_tramo = G[actual][vecino]['distancia']
            tentative_g = g_score[actual] + costo_tramo

            if tentative_g < g_score[vecino]:
                predecesores[vecino] = actual
                g_score[vecino] = tentative_g
                h_vecino = heuristica_euclidiana(pos[vecino], pos[destino])
                f_score[vecino] = tentative_g + h_vecino
                
                contador += 1
                heapq.heappush(pq, (f_score[vecino], contador, vecino))
                open_set.add(vecino)

    # Reconstrucción de la ruta si se llegó a destino
    if destino_encontrado:
        ruta = []
        curr = destino
        while curr in predecesores:
            ruta.append(curr)
            curr = predecesores[curr]
        ruta.append(origen)
        ruta.reverse()

        # Añade fotogramas finales celebrando la ruta óptima
        for _ in range(5):
            historial.append({
                'nodo_actual': destino,
                'open_set': set(open_set),
                'closed_set': set(closed_set),
                'g': g_score[destino],
                'h': 0.0,
                'f': g_score[destino],
                'ruta_final': ruta,
                'mensaje': f"¡Destino alcanzado! Ruta óptima encontrada: {ruta} (Distancia: {g_score[destino]:.2f} km)"
            })

    return historial


# ---------------------------------------------------------------------------
# 3. ANIMACIÓN Y VISUALIZACIÓN DINÁMICA
# ---------------------------------------------------------------------------
def animar_astar():
    G = construir_red_vial()
    origen = 0
    destino = 29
    pos = nx.get_node_attributes(G, 'pos')
    historial = ejecutar_pasos_astar(G, origen, destino)

    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(13.5, 8.5), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    plt.subplots_adjust(bottom=0.15, top=0.88, left=0.08, right=0.96)

    # Elementos visuales persistentes (red base)
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color='#b7c2cc', linewidth=1.4, zorder=1)

    # Colecciones de puntos y líneas dinámicas
    lineas_ruta, = ax.plot([], [], color='#007f62', linewidth=4.5, zorder=6, label='Ruta Óptima')
    lineas_aura, = ax.plot([], [], color='#007f62', linewidth=0, alpha=0, zorder=5)

    scatter_nodos = ax.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()],
                               s=80, c='#f1f4f6', edgecolors='#758391', linewidth=1.5, zorder=2)
    
    scatter_actual = ax.scatter([], [], s=260, facecolors='none', edgecolors='#ad6500',
                                linewidth=3.0, linestyle='--', zorder=7)

    # Origen y Destino estáticos
    ax.scatter(pos[origen][0], pos[origen][1], s=260, color='#007f62', edgecolors='#ffffff',
               linewidth=2.5, marker='o', zorder=8, label='Origen (0)')
    ax.scatter(pos[destino][0], pos[destino][1], s=260, color='#b34428', edgecolors='#ffffff',
               linewidth=2.5, marker='s', zorder=8, label='Destino (29)')
    
    ax.text(pos[origen][0] - 0.15, pos[origen][1] - 0.28, "ORIGEN", color='#007f62', fontsize=9.5, fontweight='bold', ha='center', zorder=9)
    ax.text(pos[destino][0] + 0.15, pos[destino][1] + 0.26, "DESTINO", color='#b34428', fontsize=9.5, fontweight='bold', ha='center', zorder=9)

    # Encabezado
    ax.set_title("SafeRouteAI - Exploración Animada del Algoritmo A*", fontsize=15, fontweight='bold', color='#172b3a', pad=22)
    ax.text(0.5, 1.02, "Heurística Euclidiana Admisible: h(n) = ||n - destino||₂ (nunca sobreestima el costo real)",
            transform=ax.transAxes, ha='center', color='#52616e', fontsize=10.5)

    # Tarjeta de estado en tiempo real
    texto_estado = ax.text(0.02, 0.96, "", transform=ax.transAxes, va='top', ha='left',
                           fontsize=9.5, family='DejaVu Sans', color='#273746',
                           bbox=dict(boxstyle='square,pad=0.7', facecolor='#f1f4f6', edgecolor='#ad6500', alpha=0.92, linewidth=1.5), zorder=10)

    # Tarjeta explicativa de admisibilidad fija en la esquina inferior derecha
    texto_explicacion = (
        "¿POR QUÉ ES ADMISIBLE ESTA HEURÍSTICA?\n"
        "─────────────────────────────────────────\n"
        "• h(n) = Distancia euclidiana en línea recta.\n"
        "• En el plano, la línea recta es la distancia\n"
        "  geodésica mínima entre dos puntos.\n"
        "• La red de calles impone desvíos y giros,\n"
        "  por lo que: costo_real*(n) >= h(n).\n"
        "• Al cumplirse h(n) <= h*(n), A* garantiza\n"
        "  encontrar la ruta de costo mínimo estricto."
    )
    ax.text(0.98, 0.05, texto_explicacion, transform=ax.transAxes, va='bottom', ha='right',
            fontsize=8.5, family='DejaVu Sans', color='#334155',
            bbox=dict(boxstyle='square,pad=0.6', facecolor='#f1f4f6', edgecolor='#758391', alpha=0.9), zorder=10)

    ax.set_xlabel("Coordenada Este-Oeste (km)", color='#334155', fontsize=10.5, labelpad=8)
    ax.set_ylabel("Coordenada Norte-Sur (km)", color='#334155', fontsize=10.5, labelpad=8)
    ax.set_xlim(-0.6, 5.7)
    ax.set_ylim(-0.6, 4.6)
    ax.tick_params(colors='#52616e', labelsize=9)
    ax.grid(True, linestyle=':', color='#f1f4f6', alpha=0.7)

    # Leyenda visual personalizada en la parte superior
    ax.plot([], [], 'o', color='#ad6500', label='Frontera (Open Set)')
    ax.plot([], [], 'o', color='#7664a4', label='Explorado (Closed Set)')
    ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.04), facecolor='#f1f4f6', edgecolor='#758391',
              labelcolor='#273746', fontsize=8.5)

    # Función de actualización de fotograma
    def actualizar(frame):
        estado = historial[frame]
        actual = estado['nodo_actual']
        open_set = estado['open_set']
        closed_set = estado['closed_set']
        ruta = estado['ruta_final']

        colores = []
        bordes = []
        tamanios = []

        for nodo in G.nodes():
            if ruta and nodo in ruta:
                colores.append('#007f62')  # Verde ruta final
                bordes.append('#ffffff')
                tamanios.append(140)
            elif nodo == actual:
                colores.append('#ad6500')  # Amarillo nodo actual
                bordes.append('#ffffff')
                tamanios.append(150)
            elif nodo in open_set:
                colores.append('#ad6500')  # Naranja frontera
                bordes.append('#fdba74')
                tamanios.append(100)
            elif nodo in closed_set:
                colores.append('#7664a4')  # Índigo visitados
                bordes.append('#a5b4fc')
                tamanios.append(90)
            else:
                colores.append('#f1f4f6')  # Gris no explorado
                bordes.append('#758391')
                tamanios.append(70)

        scatter_nodos.set_color(colores)
        scatter_nodos.set_edgecolors(bordes)
        scatter_nodos.set_sizes(tamanios)

        # Destacar nodo actual
        pos_act = pos[actual]
        scatter_actual.set_offsets([pos_act])

        # Si se llegó a la ruta final, dibujarla
        if not ruta:
            lineas_ruta.set_data([], [])
            lineas_aura.set_data([], [])
        if ruta:
            xs = [pos[n][0] for n in ruta]
            ys = [pos[n][1] for n in ruta]
            lineas_ruta.set_data(xs, ys)
            lineas_aura.set_data(xs, ys)
            scatter_actual.set_offsets(np.empty((0, 2)))

        info = (
            f"PASO {frame + 1}/{len(historial)}\n"
            "─────────────────────────────\n"
            f"• Nodo bajo examen:  N{actual}\n"
            f"• g(n) [costo real]: {estado['g']:.2f} km\n"
            f"• h(n) [heurística]: {estado['h']:.2f} km\n"
            f"• f(n) = g + h:      {estado['f']:.2f} km\n"
            f"• En frontera (Open):  {len(open_set)} nodos\n"
            f"• Visitados (Closed):  {len(closed_set)} nodos\n"
            "─────────────────────────────\n"
            f"Estado: {estado['mensaje'][:38]}..."
        )
        texto_estado.set_text(info)

        return scatter_nodos, scatter_actual, lineas_ruta, lineas_aura, texto_estado

    # Control de animación
    anim = animation.FuncAnimation(fig, actualizar, frames=len(historial), interval=400, blit=False, repeat=True)

    # Control de pausa y reanudación
    ax_play = plt.axes([0.08, 0.025, 0.14, 0.045])
    btn_play = Button(ax_play, 'Pausar', color='#f1f4f6', hovercolor='#b7c2cc')
    btn_play.label.set_color('#ad6500')
    btn_play.label.set_fontweight('bold')


    pausado = [False]

    def alternar_pausa(event):
        if pausado[0]:
            anim.resume()
            btn_play.label.set_text('Pausar')
            btn_play.label.set_color('#ad6500')
            pausado[0] = False
        else:
            anim.pause()
            btn_play.label.set_text('Reanudar')
            btn_play.label.set_color('#007f62')
            pausado[0] = True
        plt.draw()

    btn_play.on_clicked(alternar_pausa)

    fig._controles = (anim, btn_play, actualizar, alternar_pausa)
    actualizar(0)
    finalizar_figura(fig, 2)
    plt.show()
    return fig


if __name__ == '__main__':
    animar_astar()
