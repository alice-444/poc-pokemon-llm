# 7. Dépannage

[← Suivi](06-suivi-mlflow-dvc.md) · [Sommaire](README.md) · [Suivant : Limites →](08-limites.md)

Erreurs courantes rencontrées sur ce projet et leurs solutions.

---

## `HFValidationError: Repo id must use alphanumeric chars...`

```
OSError: ... Repo id must use alphanumeric chars, '-', '_' or '.' ...: './best_pokemon_model'.
```

**Cause** : un chemin local (ex. `./best_pokemon_model`) est passé à `from_pretrained`, mais il n'existe pas relativement au répertoire courant. Transformers le prend alors pour un identifiant de dépôt Hugging Face, qui est invalide.

**Solution** : pointer vers le dossier réel du modèle. Dans [`generate.py`](../best_pokemon_model/generate.py), on résout le chemin par rapport au script lui-même :

```python
import os
model_path = os.path.dirname(os.path.abspath(__file__))
```

Ainsi le script fonctionne quel que soit le répertoire de lancement.

---

## `ValueError: Using a 'device_map' ... requires 'accelerate'`

```
ValueError: Using a `device_map` ... requires `accelerate`. You can install it with `pip install accelerate`
```

**Cause** : `device_map="auto"` nécessite la librairie `accelerate`, absente de l'environnement.

**Solutions** (au choix) :

1. Installer accelerate : `pip install accelerate`.
2. **Ne pas utiliser `device_map`** (inutile en mono-GPU / CPU) et placer le modèle manuellement — c'est ce que fait `generate.py` :

```python
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=...)
model.to("cuda" if torch.cuda.is_available() else "cpu")
```

---

## `'torch_dtype' is deprecated! Use 'dtype' instead!`

**Cause** : les versions récentes de Transformers ont renommé l'argument.

**Solution** : remplacer `torch_dtype=...` par `dtype=...` dans l'appel à `from_pretrained`.

---

## `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'no_cuda'`

```
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'no_cuda'
```

**Cause** : les versions récentes de Transformers ont supprimé l'argument `no_cuda` de `TrainingArguments` au profit de `use_cpu` (même effet : forcer l'entraînement sur CPU). Survient typiquement en mode CI (`train.py --ci`).

**Solution** : dans [`src/train.py`](../src/train.py), remplacer `no_cuda=True` par `use_cpu=True`.

```python
training_args = TrainingArguments(
    ...
    use_cpu=True,   # force le CPU (ex- no_cuda)
)
```

---

## `AttributeError: ... 'parse_pretrained'`

**Cause** : `AutoTokenizer.parse_pretrained` n'existe pas. D'anciennes versions de [`src/train.py`](../src/train.py) tentaient cet appel derrière un garde-fou `hasattr(...)` trompeur (jamais vrai). Le code utilise désormais directement `from_pretrained`.

**Solution** : utiliser directement la bonne méthode :

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

---

## La réponse du modèle est tronquée

**Cause** : `max_new_tokens` trop faible coupe la génération en pleine phrase.

**Solution** : augmenter `max_new_tokens` dans [`generate.py`](../best_pokemon_model/generate.py) (réglé à `200`).

---

## Le modèle invente des statistiques

Ce n'est **pas un bug** : avec un dataset de 453 paires et un modèle de 1,1 Md de paramètres faiblement entraîné, le modèle apprend le *format* mais mémorise mal les *valeurs* exactes. Voir [Limites & pistes](08-limites.md) pour les améliorations possibles.

---

## `mlflow ui` n'affiche aucune expérience

**Cause** : lancé hors de la racine du projet, ou backend non précisé.

**Solution** :

```bash
cd /chemin/vers/poc-pokemon-llm
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## `ModuleNotFoundError: No module named 'torch'` (ou transformers, datasets…)

**Cause** : dépendances non installées ou mauvais environnement virtuel actif.

**Solution** :

```bash
source .venv/bin/activate
pip install -r requirements.txt
```
