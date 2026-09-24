"""Convenciones editoriales compartidas por las figuras de SafeRouteAI."""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

TITULOS = {
    1: 'Figura 1. Ruta de distancia mínima mediante Dijkstra',
    2: 'Figura 2. Exploración de la red mediante A*',
    3: 'Figura 3. Comparación de distancia mínima y exposición mínima',
    4: 'Figura 4. Sensibilidad de la ruta a la ponderación multicriterio',
    5: 'Figura 5. Campo de exposición sintética y rutas alternativas',
    6: 'Figura 6. Alternativas y no dominancia en distancia–exposición',
}

def configurar_estilo():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'text.color': '#273746', 'axes.labelcolor': '#334155',
        'axes.edgecolor': '#758391', 'axes.linewidth': 0.7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.color': '#52616e', 'ytick.color': '#52616e',
        'axes.axisbelow': True, 'legend.framealpha': 1,
        'savefig.facecolor': 'white', 'savefig.dpi': 300,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def _reubicar(texto, fig, x, y, fontsize=10):
    texto.set_transform(fig.transFigure)
    texto.set_position((x, y))
    texto.set_ha('left')
    texto.set_va('top')
    texto.set_fontsize(fontsize)
    texto.set_clip_on(False)


def finalizar_figura(fig, numero):
    """Separar datos, resultados y notas sin modificar valores ni callbacks."""
    fig.suptitle(TITULOS[numero], x=0.07, y=0.975, ha='left', fontsize=16,
                 fontweight='semibold', color='#172b3a')
    fig.text(0.07, 0.935, 'SafeRouteAI  |  Experimento computacional reproducible',
             fontsize=10, color='#52616e')
    for axis in fig.axes:
        axis.set_title('')
    mapas = [ax for ax in fig.axes if 'Coordenada' in ax.get_xlabel()]
    for ax in mapas:
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Coordenada este–oeste (km)')
        ax.set_ylabel('Coordenada norte–sur (km)')
        ax.grid(True, color='#dce2e7', linewidth=0.6, linestyle=':')
        for text in list(ax.texts):
            # Old subtitles and footnotes are replaced by figure-level captions.
            if text.get_transform() == ax.transAxes and (text.get_position()[1] > 1 or text.get_position()[1] < 0):
                text.remove()
    if numero in (1, 2, 4):
        ax = mapas[0]
        ax.set_title('')
        ax.set_position([0.07, 0.29 if numero == 4 else 0.19, 0.57, 0.58 if numero == 4 else 0.68])
        cards = [t for t in ax.texts if t.get_transform() == ax.transAxes]
        for i, t in enumerate(cards):
            _reubicar(t, fig, 0.69, 0.85 if i == 0 else (0.43 if numero == 4 else 0.45), 9.5)
        if numero == 1:
            ax.legend(loc='upper left', bbox_to_anchor=(0, -0.12), ncol=2, frameon=False)
        elif numero == 2:
            ax.get_legend().remove()
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.09, 0.058), ncol=3, frameon=False, fontsize=8.5)
            fig.axes[1].set_position([0.70, 0.07, 0.14, 0.04])
            fig.text(0.07, 0.895, 'h(n) = distancia euclidiana al destino; f(n) = g(n) + h(n).', fontsize=9)
        else:
            fig.text(0.07, 0.895, 'Costo = Σ [w_d · d/d_max + w_t · t/t_max + w_r · r/r_max]', fontsize=10)
    elif numero == 3:
        for i, ax in enumerate(mapas):
            ax.set_position([0.07 + i * 0.47, 0.34, 0.40, 0.53])
            ax.set_title(('(a) Mínima distancia', '(b) Mínima exposición sintética')[i], loc='left', fontsize=12)
            for t in list(ax.texts):
                if t.get_transform() == ax.transAxes:
                    t.remove()  # Metrics already appear in the comparison table.
        fig.axes[2].set_position([0.32, 0.265, 0.36, 0.016])
        fig.axes[3].set_position([0.07, 0.09, 0.87, 0.115])
    elif numero == 5:
        ax = mapas[0]
        ax.set_title('')
        ax.set_position([0.07, 0.29, 0.59, 0.59])
        ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1), frameon=False, fontsize=9)
        fig.axes[1].set_position([0.75, 0.37, 0.018, 0.34])
        fig.axes[2].set_position([0.07, 0.10, 0.87, 0.12])
        for table in fig.axes[2].tables:
            widths = [0.21, 0.34, 0.13, 0.13, 0.19]
            for (row, col), cell in table.get_celld().items():
                cell.set_width(widths[col])
        fig.text(0.07, 0.23, 'Exposición por tramo ≈ R(punto medio) × longitud (aproximación por punto medio).', fontsize=9)
    elif numero == 6:
        fig.set_size_inches(18, 10)
        axp, axg = fig.axes[:2]
        axp.set_position([0.065, 0.28, 0.35, 0.57])
        axg.set_position([0.48, 0.30, 0.29, 0.52])
        axp.set_title('(a) Distancia y exposición', loc='left', fontsize=12)
        axg.set_title('(b) Recorrido seleccionado', loc='left', fontsize=12)
        for t in axg.texts:
            if t.get_transform() == axg.transAxes:
                _reubicar(t, fig, 0.80, 0.80, 8.5)
        for t in fig.texts:
            if t.get_text().startswith('Pareto en distancia'):
                t.set_position((0.07, 0.14))
                t.set_ha('left')
                t.set_fontsize(9)
        fig.axes[2].set_position([0.10, 0.205, 0.28, 0.016])
    fig.text(0.07, 0.018,
             'Nota. Red sintética determinista: 30 nodos y 55 tramos; origen N0, destino N29. '
             'Exposición en unidades sintéticas (pts); no es una probabilidad observada.',
             fontsize=8.5, color='#52616e')
    for ax in fig.axes:
        for table in ax.tables:
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor('#ccd5dc')
                cell.set_linewidth(0.5)
                cell.set_text_props(color='#273746')
                if row == 0:
                    cell.set_facecolor('#e9eef2')
                    cell.set_text_props(weight='semibold')
    fig.canvas.draw_idle()
