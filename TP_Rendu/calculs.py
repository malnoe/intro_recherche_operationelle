# Fichier regroupant les fonctions utilisées pour faire des calculs sur les résultats des optimisations

import numpy as np
import cvxpy as cp

n = 10


def norm_inf_difference_thetas(list_values, value_comp):
    """
    Calcule la norme infinie de la différence entre un vecteur theta estimé et un vecteur theta de référence
    pour une liste de vecteurs theta estimés.

    Args:
        list_values (list of numpy.ndarray): Liste de vecteurs theta estimés.
        value_comp (numpy.ndarray): Vecteur theta de référence.

    Returns:
        list: Liste des normes infinies des différences.
    """
    res = []
    for value in list_values:
        # Calcul de la norme infinie de la différence
        vect_difference = value - value_comp 
        res.append(np.linalg.norm(vect_difference, np.inf))
    return res


def probleme_P2(X, y, S, n, verbose=False):
    """
    Résout le problème d'optimisation (P2) : ||X*theta - y||^2_2,
    avec les contraintes theta >= 0 et sum(theta) <= S.

    Args:
        X (numpy.ndarray): Matrice de taille m*n.
        y (numpy.ndarray): Vecteur de taille m.
        S (int or float): Valeur contraignant la somme des coefficients de theta.
        n (int): Taille du vecteur theta.
        verbose (bool, optional): Affiche les détails de la résolution si True. Par défaut False.

    Returns:
        list: Contient :
            - status (str): Statut de l'optimisation.
            - valeur_probleme (float): Valeur minimisée du problème.
            - theta_estime (numpy.ndarray): Solution optimisée pour theta.
            - residuals (numpy.ndarray): Résidus y - X*theta.
            - norme_inf (float): Norme infinie de la solution estimée.
    """
    # Définition du problème d'optimisation
    theta = cp.Variable(n)
    L = cp.norm(X @ theta - y, 2)**2
    objective = cp.Minimize(L)
    contrainte_positivite = theta >= 0
    contrainte_stabilite = sum(theta) <= S
    prob = cp.Problem(objective, [contrainte_positivite, contrainte_stabilite])
    
    # Résolution du problème
    prob.solve(verbose=verbose)
    
    # Extraction des résultats
    status = prob.status
    valeur_probleme = L.value
    theta_estime = theta.value
    residuals = y - X @ theta_estime
    norme_inf = np.linalg.norm(theta_estime, np.inf)

    return [status, valeur_probleme, theta_estime, residuals, norme_inf]


def grille_solutions_P2(X, y, grille_S, n):
    """
    Résout le problème (P2) pour une grille de valeurs S.

    Args:
        X (numpy.ndarray): Matrice de taille m*n.
        y (numpy.ndarray): Vecteur de taille m.
        grille_S (list of int or float): Liste des valeurs de S.
        n (int): Taille du vecteur theta.

    Returns:
        list: Contient :
            - status_vals (list of str): Statuts des optimisations.
            - problem_vals (list of float): Valeurs minimisées des problèmes.
            - theta_estim_vals (list of numpy.ndarray): Solutions optimisées pour theta.
            - residuals_vals (list of numpy.ndarray): Résidus y - X*theta pour chaque solution.
            - norm_inf_vals (list of float): Normes infinies des solutions estimées.
    """
    # Initialisation des listes de résultats
    status_vals = []
    problem_vals = []
    theta_estim_vals = []
    residuals_vals = []
    norm_inf_vals = []

    # Résolution du problème pour chaque valeur de S dans la grille
    for S in grille_S:
        solution = probleme_P2(X, y, S, n)
        status_vals.append(solution[0])
        problem_vals.append(solution[1])
        theta_estim_vals.append(solution[2])
        residuals_vals.append(solution[3])
        norm_inf_vals.append(solution[4])

    return [status_vals, problem_vals, theta_estim_vals, residuals_vals, norm_inf_vals]


def probleme_P2_dual(X, y, S, n, verbose=False):
    """
    Résout le problème d'optimisation (P2) avec retour sur les valeurs duales.

    Args:
        X (numpy.ndarray): Matrice de taille m*n.
        y (numpy.ndarray): Vecteur de taille m.
        S (int or float): Valeur contraignant la somme des coefficients de theta.
        n (int): Taille du vecteur theta.
        verbose (bool, optional): Affiche les détails de la résolution si True. Par défaut False.

    Returns:
        list: Contient :
            - status (str): Statut de l'optimisation.
            - valeur_probleme (float): Valeur minimisée du problème.
            - theta_estime (numpy.ndarray): Solution optimisée pour theta.
            - residuals (numpy.ndarray): Résidus y - X*theta.
            - norme_inf (float): Norme infinie de la solution estimée.
            - dual_positivite (numpy.ndarray): Valeurs duales pour la contrainte de positivité.
            - dual_stabilite (float): Valeur duale pour la contrainte de stabilité.
    """
    # Définition du problème d'optimisation
    theta = cp.Variable(n)
    L = cp.norm(X @ theta - y, 2)**2
    objective = cp.Minimize(L)
    contrainte_positivite = theta >= 0
    contrainte_stabilite = sum(theta) <= S
    prob = cp.Problem(objective, [contrainte_positivite, contrainte_stabilite])

    # Résolution du problème
    prob.solve(verbose=verbose)

    # Extraction des résultats
    status = prob.status
    valeur_probleme = L.value
    theta_estime = theta.value
    residuals = y - X @ theta_estime
    norme_inf = np.linalg.norm(theta_estime, np.inf)
    dual_positivite = prob.constraints[0].dual_value
    dual_stabilite = prob.constraints[1].dual_value

    return [status, valeur_probleme, theta_estime, residuals, norme_inf, dual_positivite, dual_stabilite]



def grille_solutions_P2_dual(X, y, grille_S, n):
    """
    Résout le problème (P2) pour une grille de valeurs S avec retour sur les valeurs duales.

    Args:
        X (numpy.ndarray): Matrice de taille m*n.
        y (numpy.ndarray): Vecteur de taille m.
        grille_S (list of int or float): Liste des valeurs de S.
        n (int): Taille du vecteur theta.

    Returns:
        list: Contient :
            - valeur_probleme_vals (list of float): Valeurs minimisées des problèmes.
            - dual_positivite_vals (list of numpy.ndarray): Valeurs duales pour la contrainte de positivité.
            - dual_norminf_positivite_vals (list of float): Normes infinies des valeurs duales de positivité.
            - dual_stabilite_vals (list of float): Valeurs duales pour la contrainte de stabilité.
    """
    # Initialisation des listes de résultats
    dual_positivite_vals = []
    dual_stabilite_vals = []
    valeur_probleme_vals = []
    dual_norminf_positivite_vals = []

    # Résolution du problème pour chaque valeur de S dans la grille
    for S in grille_S:
        solution_P2_dual = probleme_P2_dual(X, y, S, n)
        dual_positivite_vals.append(solution_P2_dual[5])
        dual_norminf_positivite_vals.append(np.linalg.norm(solution_P2_dual[5], np.inf))
        dual_stabilite_vals.append(solution_P2_dual[6])
        valeur_probleme_vals.append(solution_P2_dual[1])

    return [valeur_probleme_vals, dual_positivite_vals, dual_norminf_positivite_vals, dual_stabilite_vals]