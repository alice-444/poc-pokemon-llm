"""
Script de validation des données du pipeline Pokémon LLM.
Vérifie l'intégrité et la qualité des fichiers JSON produits
par les étapes d'extraction et de préparation.

Utilisé dans la CI pour garantir que les données sont conformes
avant de lancer l'entraînement.
"""

import json
import sys
import argparse
import os


def valider_raw_pokemons(filepath, min_count):
    """Valide le fichier raw_pokemons.json (données brutes extraites de PokéAPI)."""
    print(f"🔍 Validation de {filepath}...")
    erreurs = []

    # --- Existence et format JSON ---
    if not os.path.exists(filepath):
        print(f"  ❌ Fichier introuvable : {filepath}")
        return False

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON invalide : {e}")
        return False

    # --- Structure : liste de dicts ---
    if not isinstance(data, list):
        print(f"  ❌ Le fichier doit contenir une liste, trouvé : {type(data).__name__}")
        return False

    if len(data) < min_count:
        erreurs.append(f"Nombre de Pokémon insuffisant : {len(data)} < {min_count} attendus")

    # --- Validation de chaque Pokémon ---
    cles_requises = {"id", "name", "types", "stats", "abilities", "height", "weight"}
    stats_requises = {"hp", "attack", "defense", "speed"}

    for i, pokemon in enumerate(data):
        if not isinstance(pokemon, dict):
            erreurs.append(f"[{i}] Entrée invalide : attendu dict, trouvé {type(pokemon).__name__}")
            continue

        # Vérifier les clés requises
        cles_manquantes = cles_requises - set(pokemon.keys())
        if cles_manquantes:
            erreurs.append(f"[{i}] Clés manquantes : {cles_manquantes}")
            continue

        nom = pokemon.get("name", f"index_{i}")

        # Types : liste de chaînes non vide
        if not isinstance(pokemon["types"], list) or len(pokemon["types"]) == 0:
            erreurs.append(f"[{nom}] 'types' doit être une liste non vide")
        elif not all(isinstance(t, str) for t in pokemon["types"]):
            erreurs.append(f"[{nom}] 'types' doit contenir uniquement des chaînes")

        # Stats : dict avec les clés requises
        if not isinstance(pokemon["stats"], dict):
            erreurs.append(f"[{nom}] 'stats' doit être un dictionnaire")
        else:
            stats_manquantes = stats_requises - set(pokemon["stats"].keys())
            if stats_manquantes:
                erreurs.append(f"[{nom}] Stats manquantes : {stats_manquantes}")

        # Abilities : liste de chaînes
        if not isinstance(pokemon["abilities"], list):
            erreurs.append(f"[{nom}] 'abilities' doit être une liste")
        elif not all(isinstance(a, str) for a in pokemon["abilities"]):
            erreurs.append(f"[{nom}] 'abilities' doit contenir uniquement des chaînes")

        # Height et weight : nombres positifs
        for champ in ("height", "weight"):
            valeur = pokemon.get(champ)
            if not isinstance(valeur, (int, float)) or valeur <= 0:
                erreurs.append(f"[{nom}] '{champ}' doit être un nombre positif, trouvé : {valeur}")

    # --- Résultat ---
    if erreurs:
        print(f"  ❌ {len(erreurs)} erreur(s) détectée(s) :")
        for err in erreurs:
            print(f"     • {err}")
        return False

    print(f"  ✅ {len(data)} Pokémon validés avec succès")
    return True


def valider_instructions(filepath, raw_filepath):
    """Valide le fichier pokedex_instructions.json (dataset d'entraînement)."""
    print(f"🔍 Validation de {filepath}...")
    erreurs = []

    # --- Existence et format JSON ---
    if not os.path.exists(filepath):
        print(f"  ❌ Fichier introuvable : {filepath}")
        return False

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON invalide : {e}")
        return False

    # --- Structure : liste de dicts ---
    if not isinstance(data, list):
        print(f"  ❌ Le fichier doit contenir une liste, trouvé : {type(data).__name__}")
        return False

    if len(data) == 0:
        print("  ❌ Le dataset d'instructions est vide")
        return False

    # --- Cohérence avec raw_pokemons.json ---
    if os.path.exists(raw_filepath):
        with open(raw_filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        attendu = len(raw_data) * 3  # 3 variantes par Pokémon
        if len(data) != attendu:
            erreurs.append(
                f"Nombre d'instructions incohérent : {len(data)} trouvées, "
                f"{attendu} attendues ({len(raw_data)} Pokémon × 3 variantes)"
            )

    # --- Validation de chaque entrée ---
    cles_requises = {"instruction", "input", "output"}

    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            erreurs.append(f"[{i}] Entrée invalide : attendu dict, trouvé {type(entry).__name__}")
            continue

        cles_manquantes = cles_requises - set(entry.keys())
        if cles_manquantes:
            erreurs.append(f"[{i}] Clés manquantes : {cles_manquantes}")
            continue

        # instruction et output doivent être des chaînes non vides
        if not isinstance(entry["instruction"], str) or len(entry["instruction"].strip()) == 0:
            erreurs.append(f"[{i}] 'instruction' doit être une chaîne non vide")

        if not isinstance(entry["output"], str) or len(entry["output"].strip()) == 0:
            erreurs.append(f"[{i}] 'output' doit être une chaîne non vide")

    # --- Résultat ---
    if erreurs:
        print(f"  ❌ {len(erreurs)} erreur(s) détectée(s) :")
        for err in erreurs:
            print(f"     • {err}")
        return False

    print(f"  ✅ {len(data)} paires d'instructions validées avec succès")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation des données du pipeline Pokémon LLM")
    parser.add_argument("--min-count", type=int, default=1,
                        help="Nombre minimum de Pokémon attendus dans raw_pokemons.json (défaut : 1)")
    args = parser.parse_args()

    raw_path = "data/raw_pokemons.json"
    instructions_path = "data/pokedex_instructions.json"

    print("=" * 50)
    print("🔬 Validation des données du pipeline")
    print("=" * 50)

    resultats = []

    # Validation des données brutes
    resultats.append(valider_raw_pokemons(raw_path, args.min_count))

    # Validation du dataset d'instructions
    resultats.append(valider_instructions(instructions_path, raw_path))

    print("=" * 50)

    if all(resultats):
        print("🎉 Toutes les validations sont passées avec succès !")
        sys.exit(0)
    else:
        print("💥 Des erreurs ont été détectées. Vérifiez les messages ci-dessus.")
        sys.exit(1)
