# 8. Limites & pistes d'amélioration

[← Dépannage](07-depannage.md) · [Sommaire](README.md) · [Suivant : CI/CD →](09-ci-cd.md)

Ce projet est un **PoC pédagogique** : il démontre une chaîne MLOps complète, pas un modèle de production. Voici ses limites connues et des pistes pour aller plus loin.

## Limites connues

| Limite                              | Explication                                                                                                                                                          |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Valeurs inventées**               | Le modèle reproduit le *format* d'une fiche mais peut se tromper sur les stats/talents : dataset trop petit et entraînement court pour mémoriser 151 fiches exactes. |
| **Dataset restreint**               | 453 paires issues de 151 Pokémon, avec seulement 3 reformulations de question (peu de diversité linguistique).                                                       |
| **Couverture limitée**              | Uniquement la 1ʳᵉ génération (151 Pokémon).                                                                                                                          |
| **Pas d'évaluation de modèle**      | Aucune métrique de qualité (exactitude des fiches, hallucinations) n'est calculée — seule la `loss` est suivie. La validation existante (`validate.py`) porte sur l'intégrité des **données**, pas sur la qualité du modèle.                                                      |
| **Fine-tuning complet**             | Tous les poids sont mis à jour : coûteux en mémoire, peu adapté à du matériel léger.                                                                                 |
| **Pas de séparation train/val**     | Le dataset n'est pas découpé : pas de mesure d'overfitting.                                                                                                          |
| **Traduction des types incomplète** | Le mapping FR dans `prepare.py` ne couvre que certains types ; d'autres restent en anglais.                                                                          |

## Pistes d'amélioration

### Données

- **Enrichir le dataset** : descriptions Pokédex officielles, évolutions, faiblesses/résistances, plus de reformulations.
- **Étendre aux générations suivantes** (PokéAPI en propose plus de 1000).
- **Compléter la traduction des types** dans [`src/prepare.py`](../src/prepare.py).

### Entraînement

- **LoRA / PEFT** : fine-tuning paramètre-efficace pour réduire fortement la mémoire et accélérer l'entraînement.
- **Split train/validation** + `eval_steps` pour suivre l'overfitting.
- **Recherche d'hyperparamètres** (learning rate, epochs, longueur de séquence).

### Évaluation

- **Métriques dédiées** : taux d'exactitude des stats/talents par rapport à la vérité terrain (les données brutes en sont la source idéale).
- **Détection d'hallucinations** : comparer la sortie générée aux champs de `raw_pokemons.json`.

### Industrialisation

- **Ajouter les étapes `extract` et `train` au pipeline DVC** (actuellement seul `prepare` est déclaré dans `dvc.yaml`) — cela permettrait à la CI d'appeler `dvc repro` au lieu d'enchaîner les scripts.
- **Servir le modèle** via une petite API (FastAPI) ou une interface (Gradio / Streamlit).
- **Figer les versions** des dépendances (`pip freeze`) pour une reproductibilité totale.
- **Étendre la CI** : ajouter de vrais tests unitaires, et une évaluation de la qualité du modèle (pas seulement la validation des données).

> ✅ **Déjà en place** : une CI GitHub Actions rejoue le pipeline (extract → prepare → validate → train léger) à chaque push sur `main`, et `validate.py` contrôle l'intégrité des données. Voir [CI/CD](09-ci-cd.md).

## Avertissement

Les données proviennent de [PokéAPI](https://pokeapi.co/). *Pokémon* et les noms associés sont des marques déposées de Nintendo / Game Freak / The Pokémon Company. Projet purement éducatif et non commercial.
