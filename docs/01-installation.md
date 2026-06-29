# 1. Installation

[← Sommaire](README.md) · [Suivant : Architecture →](02-architecture.md)

## Prérequis

| Élément    | Détail                                                                          |
| ---------- | ------------------------------------------------------------------------------- |
| **Python** | 3.13                                                                            |
| **Git**    | pour cloner le dépôt                                                            |
| **DVC**    | optionnel, pour récupérer les données versionnées                               |
| **GPU**    | recommandé (≥ 8 Go VRAM) pour le fine-tuning ; le CPU fonctionne mais lentement |
| **RAM**    | ≥ 16 Go conseillé en mode CPU                                                   |
| **Disque** | ~5 Go (modèle de base + checkpoints + artefacts MLflow)                         |

> Le projet fonctionne entièrement sur CPU : le code bascule automatiquement en `fp16` si un GPU CUDA est détecté, sinon en `fp32`.

## Étapes

### 1. Cloner le dépôt

```bash
git clone https://github.com/alice-444/poc-pokemon-llm.git
cd poc-pokemon-llm
```

### 2. Créer l'environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

Le fichier [`requirements.txt`](../requirements.txt) regroupe les dépendances par étape :

| Paquet         | Rôle                                                   |
| -------------- | ------------------------------------------------------ |
| `requests`     | Appels HTTP vers PokéAPI (extraction)                  |
| `dvc`          | Versionnage des données et orchestration du pipeline   |
| `torch`        | Backend de calcul (entraînement & inférence)           |
| `transformers` | Chargement du modèle, tokenizer, boucle d'entraînement |
| `datasets`     | Chargement et tokenisation du dataset JSON             |
| `accelerate`   | Support multi-device / placement du modèle             |
| `mlflow`       | Suivi des expériences et des artefacts                 |

> ℹ️ Les versions indiquées dans `requirements.txt` sont des minimums. Pour figer ton environnement exact après installation : `pip freeze > requirements.txt`.

## Vérifier l'installation

```bash
python -c "import torch, transformers, datasets, mlflow; print('OK')"
```

Si la commande affiche `OK`, l'environnement est prêt. Passe à l'[architecture du pipeline](02-architecture.md).
