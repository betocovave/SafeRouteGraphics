"""
=============================================================================
SafeRouteAI - Programa 5: Mapa de Calor de Riesgo y Rutas Alternativas
=============================================================================
Descripción:
    Presenta un mapa de calor continuo en 2D que modela la densidad espacial de
    riesgo sintético sobre el plano de una ciudad ficticia.

    Sobre este campo continuo se superpone la red de calles y se proyectan
    TRES rutas alternativas entre el mismo Origen (0, 0) y Destino (5, 4):
      1. Ruta Directa (Rojo):   Minimiza distancia (atraviesa el foco crítico).
      2. Ruta Equilibrada (Ámbar): Compromiso entre distancia y seguridad.
      3. Ruta Segura (Verde):   Evita completamente los focos de mayor riesgo.

Coherencia Matemática:
    El algoritmo deriva el costo de riesgo de cada arista evaluando exactamente
    el mismo campo continuo R(x, y) visualizado en el mapa:
        r_uv = R(x_medio, y_medio) · distancia_uv
    De este modo, los valores visuales del mapa de calor corresponden de forma
    rigurosa y verificable con las decisiones algorítmicas de SafeRouteAI.

Uso:
    python 05_mapa_calor_rutas.py
=============================================================================
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from estilo_investigacion import configurar_estilo, finalizar_figura

configurar_estilo()
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# 1. CAMPO CONTINUO DE RIESGO 2D (DEFINICIÓN ANALÍTICA)
# ---------------------------------------------------------------------------
def evaluar_campo_riesgo(X, Y):
    """
    Campo bidimensional continuo con dos focos críticos ficticios:
      - Foco A (Centro de la ciudad): (x=2.5, y=2.0)
      - Foco B (Sector comercial):    (x=4.0, y=1.0)
    Retorna la densidad de riesgo sintético por kilómetro.
    """
    foco_1 = 18.0 * np.exp(-((X - 2.5) ** 2 + (Y - 2.0) ** 2) / (2 * 0.75 ** 2))
    foco_2 = 12.0 * np.exp(-((X - 4.0) ** 2 + (Y - 1.0) ** 2) / (2 * 0.65 ** 2))
    return 1.0 + foco_1 + foco_2


# ---------------------------------------------------------------------------
# 2. CONSTRUCCIÓN DE LA RED VIAL DERIVADA DEL CAMPO DE RIESGO
# ---------------------------------------------------------------------------
def construir_red_vial_desde_campo(ancho=6, alto=5, semilla=42):
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

        # El riesgo se evalúa directamente sobre el campo analítico en el punto medio
        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        densidad_riesgo = float(evaluar_campo_riesgo(xm, ym))
        riesgo_tramo = float(densidad_riesgo * dist)

        G[u][v]['distancia'] = dist
        G[u][v]['tiempo'] = tiempo_min
        G[u][v]['riesgo'] = riesgo_tramo

        distancias.append(dist)
        tiempos.append(tiempo_min)
        riesgos.append(riesgo_tramo)

    d_max = max(distancias)
    t_max = max(tiempos)
    r_max = max(riesgos)

    for u, v in G.edges():
        G[u][v]['d_norm'] = G[u][v]['distancia'] / d_max
        G[u][v]['t_norm'] = G[u][v]['tiempo'] / t_max
        G[u][v]['r_norm'] = G[u][v]['riesgo'] / r_max

    return G


# ---------------------------------------------------------------------------
# 3. GENERACIÓN DE LAS TRES RUTAS ALTERNATIVAS
# ---------------------------------------------------------------------------
def obtener_rutas_alternativas(G, origen, destino):
    """
    Genera 3 rutas con perfiles estratégicos bien diferenciados:
      - Ruta 1: Enfoque 100% Distancia (mínimo kilometraje)
      - Ruta 2: Enfoque Balanceado (60% mitigación de riesgo, 40% distancia/tiempo)
      - Ruta 3: Enfoque 100% Seguridad (menor riesgo acumulado absoluto)
    """
    # 1. Ruta Directa (Mínima Distancia)
    p_directa = nx.dijkstra_path(G, origen, destino, weight='distancia')

    # 2. Ruta Equilibrada
    for u, v in G.edges():
        G[u][v]['costo_bal'] = (0.20 * G[u][v]['d_norm'] +
                                0.20 * G[u][v]['t_norm'] +
                                0.60 * G[u][v]['r_norm'])
    p_equilibrada = nx.dijkstra_path(G, origen, destino, weight='costo_bal')

    # 3. Ruta Segura (Mínimo Riesgo)
    p_segura = nx.dijkstra_path(G, origen, destino, weight='riesgo')

    rutas = [
        {"nombre": "Ruta A (Directa)", "path": p_directa, "color": "#0072b2",
         "estilo": "-", "marcador": "o", "estrategia": "Mínima distancia"},
        {"nombre": "Ruta B (Equilibrada)", "path": p_equilibrada, "color": "#80558c",
         "estilo": "-.", "marcador": "^", "estrategia": "Pesos d/t/r: 0.20/0.20/0.60"},
        {"nombre": "Ruta C (Menor riesgo)", "path": p_segura, "color": "#007f62",
         "estilo": "--", "marcador": "s", "estrategia": "Mínima exposición sintética"}
    ]

    for r in rutas:
        path = r["path"]
        r["distancia"] = sum(G[u][v]['distancia'] for u, v in zip(path[:-1], path[1:]))
        r["tiempo"] = sum(G[u][v]['tiempo'] for u, v in zip(path[:-1], path[1:]))
        r["riesgo"] = sum(G[u][v]['riesgo'] for u, v in zip(path[:-1], path[1:]))

    return rutas


# ---------------------------------------------------------------------------
# 4. VISUALIZACIÓN GRÁFICA CON MAPA DE CALOR
# ---------------------------------------------------------------------------
def graficar_mapa_calor():
    G = construir_red_vial_desde_campo()
    origen = 0
    destino = 29
    pos = nx.get_node_attributes(G, 'pos')
    rutas = obtener_rutas_alternativas(G, origen, destino)

    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(14.0, 9.2), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    plt.subplots_adjust(bottom=0.22, top=0.90, left=0.08, right=0.88)

    # 1. GENERACIÓN DEL MAPA DE CALOR CONTINUO DE FONDO
    x_grid = np.linspace(-0.6, 5.6, 350)
    y_grid = np.linspace(-0.6, 4.6, 300)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = evaluar_campo_riesgo(X, Y)

    # Contornos de calor con colormap YlOrRd
    niveles = np.linspace(Z.min(), Z.max(), 40)
    mapa_contornos = ax.contourf(X, Y, Z, levels=niveles, cmap='YlOrRd', alpha=0.55, zorder=1)
    
    # Líneas sutiles de curvas de nivel
    ax.contour(X, Y, Z, levels=niveles[::5], colors='#78350f', linewidths=0.5, alpha=0.35, zorder=2)

    # 2. DIBUJO DE LA RED VIAL SOBRE EL MAPA
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color='#b7c2cc', linewidth=1.2, alpha=0.75, zorder=3)

    for nodo, (x, y) in pos.items():
        ax.scatter(x, y, s=45, color='#f1f4f6', edgecolors='#52616e', linewidth=1.0, zorder=4)

    # 3. SUPERPOSICIÓN DE LAS TRES RUTAS ALTERNATIVAS
    for r in rutas:
        path = r["path"]
        xs = [pos[n][0] for n in path]
        ys = [pos[n][1] for n in path]

        # Halo/Aura difuminada
        ax.plot(xs, ys, color=r["color"], linewidth=0, alpha=0, zorder=5)
        # Trazo principal
        ax.plot(xs, ys, color=r["color"], linewidth=3.2, linestyle=r["estilo"],
                marker=r["marcador"], markersize=6.5, markeredgecolor='#ffffff', markeredgewidth=1.0,
                zorder=6, label=f"{r['nombre']} ({r['distancia']:.1f} km, {r['riesgo']:.1f} pts)")

    # 4. IDENTIFICACIÓN DE ORIGEN Y DESTINO
    ax.scatter(pos[origen][0], pos[origen][1], s=260, color='#007f62', edgecolors='#ffffff',
               linewidth=2.5, marker='o', zorder=8)
    ax.scatter(pos[destino][0], pos[destino][1], s=260, color='#b34428', edgecolors='#ffffff',
               linewidth=2.5, marker='s', zorder=8)
    ax.text(pos[origen][0] - 0.15, pos[origen][1] - 0.28, "ORIGEN (0,0)", color='#ffffff',
            fontsize=9.5, fontweight='bold', ha='center',
            bbox=dict(boxstyle='square,pad=0.2', facecolor='#007f62', edgecolor='none'), zorder=9)
    ax.text(pos[destino][0] + 0.15, pos[destino][1] + 0.26, "DESTINO (5,4)", color='#ffffff',
            fontsize=9.5, fontweight='bold', ha='center',
            bbox=dict(boxstyle='square,pad=0.2', facecolor='#b34428', edgecolor='none'), zorder=9)

    # Títulos
    ax.set_title("SafeRouteAI - Mapa de Calor de Riesgo y Navegación de Rutas Alternativas",
                 fontsize=14.5, fontweight='bold', color='#172b3a', pad=22)
    ax.text(0.5, 1.02, "Exposición por tramo ≈ R(punto medio) × longitud; aproximación por punto medio",
            transform=ax.transAxes, ha='center', color='#52616e', fontsize=10.0)

    ax.set_xlabel("Coordenada Este-Oeste (km)", color='#334155', fontsize=10)
    ax.set_ylabel("Coordenada Norte-Sur (km)", color='#334155', fontsize=10)
    ax.set_xlim(-0.6, 5.7)
    ax.set_ylim(-0.6, 4.6)
    ax.tick_params(colors='#52616e', labelsize=8.5)
    ax.grid(True, linestyle=':', color='#b7c2cc', alpha=0.4)

    # Leyenda de Rutas
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), facecolor='#ffffff',
              edgecolor='#b7c2cc', labelcolor='#273746', fontsize=8.5, framealpha=0.92)

    # Barra de color vertical para el mapa de calor
    cax = fig.add_axes([0.90, 0.28, 0.02, 0.58])
    cb = fig.colorbar(mapa_contornos, cax=cax)
    cb.set_label('Densidad del Campo de Riesgo R(x,y)\n[Índice Sintético / km]', color='#334155', fontsize=9.0)
    cb.ax.tick_params(colors='#52616e', labelsize=8.0)

    # TABLA COMPARATIVA INFERIOR
    ax_tabla = fig.add_axes([0.08, 0.04, 0.72, 0.12])
    ax_tabla.axis('off')

    cabeceras = ["Ruta Candidata", "Estrategia Algorítmica", "Distancia", "Tiempo", "Riesgo Acumulado*"]
    filas = [cabeceras]
    for r in rutas:
        filas.append([
            r["nombre"],
            r["estrategia"],
            f"{r['distancia']:.2f} km",
            f"{r['tiempo']:.1f} min",
            f"{r['riesgo']:.2f} pts"
        ])

    tabla = ax_tabla.table(cellText=filas, loc='center', cellLoc='center')
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8.5)
    tabla.scale(1.0, 1.30)

    for (i, j), cell in tabla.get_celld().items():
        cell.set_edgecolor('#b7c2cc')
        if i == 0:
            cell.set_facecolor('#f1f4f6')
            cell.set_text_props(weight='bold', color='#0072b2')
        else:
            cell.set_facecolor('#ffffff' if i % 2 == 0 else '#f7f8fa')
            if j == 0:
                cell.set_text_props(weight='bold', color=rutas[i - 1]["color"])
            else:
                cell.set_text_props(color='#273746')



    finalizar_figura(fig, 5)
    plt.show()
    return fig


if __name__ == '__main__':
    graficar_mapa_calor()
