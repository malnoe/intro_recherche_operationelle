# Fichier regroupant les fonctions permettant d'afficher les résultats des optimisations de façon esthétique.

import numpy as np


def pretty_result(solution, theta_true):
    """
    Fonction d'affichage des résultats des problèmes (P1) et (P2).
    Affiche les résultats de l'optimisation.

    Args:
        solution (list): Sortie de la fonction probleme_P1() ou probleme_P2().
        theta_true (numpy.ndarray): Veritable valeur de theta.
    """
    status = solution[0]
    valeur_probleme = solution[1]
    theta_estime = solution[2]
    residuals = solution[3]
    norm_inf = solution[4]
    
    # Statut de l'optimisation
    print(f"Le statut de l'estimation est : {status}.\n")
    # Valeur du problème
    print(f"La valeur du problème qui a été minimisée est de {valeur_probleme:.4f}.\n")
    # Comparaison veritable valeur du parametre theta et valeur estimee
    print(
        f"La véritable valeur de theta est :\n{theta_true}\n\n"
        f"et la valeur estimée par la fonction est :\n{theta_estime}.\n"
    )
    # Norme infinie du vecteur parametre estime
    print(f"La norme infinie du vecteur paramètre estimé est de {norm_inf:.4f}.\n")
    # Norme infinie de la différence entre theta_true et theta_estime
    print(
        f"La norme infinie du vecteur theta_true - theta_estim est de "
        f"{np.linalg.norm(theta_true - theta_estime, np.inf):.4f}.\n"
    )
    # Residus
    print(f"Les résidus sont :\n{residuals}.\n")
    # Norme infinie du vecteur résidus
    print(
        f"La norme infinie du vecteur des résidus est de "
        f"{np.linalg.norm(residuals, np.inf):.4f}.\n"
    )
    





