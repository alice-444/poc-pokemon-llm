# 📚 Documentation — PoC Pokémon LLM

Documentation technique complète du projet de fine-tuning d'un LLM sur des fiches Pokédex.
Pour une présentation rapide, voir le [README principal](../README.md).

## Sommaire

| # | Document | Contenu |
|---|----------|---------|
| 1 | [Installation](01-installation.md) | Prérequis, environnement virtuel, dépendances |
| 2 | [Architecture & pipeline](02-architecture.md) | Vue d'ensemble, flux de données, ordre d'exécution |
| 3 | [Données](03-donnees.md) | Extraction PokéAPI, format brut, génération du dataset d'instructions |
| 4 | [Entraînement](04-entrainement.md) | Modèle de base, hyperparamètres, template de prompt, checkpoints |
| 5 | [Inférence](05-inference.md) | Utiliser le modèle entraîné, paramètres de génération |
| 6 | [Suivi : MLflow & DVC](06-suivi-mlflow-dvc.md) | Tracking des expériences, versionnage des données |
| 7 | [Dépannage](07-depannage.md) | Erreurs courantes et solutions |
| 8 | [Limites & pistes](08-limites.md) | Limites connues du PoC et améliorations possibles |
| 9 | [Intégration continue (CI/CD)](09-ci-cd.md) | Pipeline GitHub Actions, validation, mode CI |

## Parcours recommandé

```
Installation (1) → Architecture (2) → Données (3) → Entraînement (4) → Inférence (5)
```

Les documents 6 à 9 sont transverses : consulte-les selon le besoin (suivi des runs, résolution d'un bug, réflexion sur l'évolution du projet, ou fonctionnement de la CI).
