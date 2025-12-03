# Fichier pour les fonctions de plot

import matplotlib.pyplot as plt


def pretty_plot(
    x, 
    y_vals, 
    legend, 
    colors, 
    title, 
    xlab, 
    ylab, 
    vertical_lines=None, 
    legend_vertical=None, 
    colors_vertical=None
):
    """
    Trace un graphique avec des courbes et des lignes verticales optionnelles.

    Args:
        x (list or array-like): Les valeurs de l'axe des abscisses.
        y_vals (list of list or array-like): Les valeurs de l'axe des ordonnées pour chaque courbe.
        legend (list of str): Les légendes associées à chaque courbe.
        colors (list of str): Les couleurs associées à chaque courbe.
        title (str): Le titre du graphique.
        xlab (str): Le label de l'axe des abscisses.
        ylab (str): Le label de l'axe des ordonnées.
        vertical_lines (list, optional): Les positions des lignes verticales. Defaults à None.
        legend_vertical (list of str, optional): Les légendes associées aux lignes verticales. Defaults à None.
        colors_vertical (list of str, optional): Les couleurs des lignes verticales. Defaults à None.
    """
    # Tracer les courbes
    for i in range(len(y_vals)):
        y = y_vals[i]
        plt.plot(x, y, label=legend[i], color=colors[i])
    
    # Tracer les lignes verticales si demandées
    if vertical_lines is not None:
        for i, v in enumerate(vertical_lines):
            plt.axvline(
                x=v, 
                color=colors_vertical[i], 
                linestyle='--', 
                label=legend_vertical[i]
            )

    # Ajouter les labels, le titre et la légende
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.show()
