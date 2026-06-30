import os
import json

# Traduction des types Pokémon en français (couvre toute la 1re génération).
# Tout type absent du dictionnaire est conservé tel quel (capitalisé) par sécurité.
TYPES_FR = {
    "normal": "Normal",
    "fire": "Feu",
    "water": "Eau",
    "electric": "Électrik",
    "grass": "Plante",
    "ice": "Glace",
    "fighting": "Combat",
    "poison": "Poison",
    "ground": "Sol",
    "flying": "Vol",
    "psychic": "Psy",
    "bug": "Insecte",
    "rock": "Roche",
    "ghost": "Spectre",
    "dragon": "Dragon",
}

def format_to_instructions(raw_data_path, output_path):
    print(f"Transformation des données depuis {raw_data_path}...")
    
    with open(raw_data_path, "r", encoding="utf-8") as f:
        pokemons = json.load(f)
    
    instruction_dataset = []
    
    for p in pokemons:
        # Traduction des types en français, type par type (mapping complet et sûr)
        types_fr = ", ".join(TYPES_FR.get(t, t.capitalize()) for t in p["types"])
        
        # Formatter les statistiques pour le texte (on exploite aussi les stats spéciales)
        stats_txt = (
            f"PV: {p['stats']['hp']}, Attaque: {p['stats']['attack']}, "
            f"Défense: {p['stats']['defense']}, "
            f"Attaque Spéciale: {p['stats']['special-attack']}, "
            f"Défense Spéciale: {p['stats']['special-defense']}, "
            f"Vitesse: {p['stats']['speed']}"
        )

        # Phrase sur les talents — omise si le Pokémon n'en a aucun (évite « Ses talents sont : . »)
        talents = ", ".join(p["abilities"])
        talents_txt = f"Ses talents sont : {talents}. " if talents else ""

        # La réponse textuelle que le LLM devra générer
        reponse = (
            f"{p['name']} est un Pokémon de type {types_fr}. "
            f"Il mesure {p['height']}m et pèse {p['weight']}kg. "
            f"{talents_txt}"
            f"Ses statistiques de base sont - {stats_txt}."
        )
        
        # Augmentation des données : 3 variantes de questions pour enrichir le dataset
        variantes_questions = [
            f"Quelles sont les caractéristiques de {p['name']} ?",
            f"Donne-moi la fiche Pokedex de {p['name']}.",
            f"Peux-tu me décrire le Pokémon {p['name']} (types, stats, talents) ?"
        ]
        
        for question in variantes_questions:
            instruction_dataset.append({
                "instruction": question,
                "input": "", # Utilisé si on avait un contexte supplémentaire, inutile ici
                "output": reponse
            })

    # Sauvegarde du nouveau dataset au format JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(instruction_dataset, f, indent=4, ensure_ascii=False)
        
    print(f"Terminé ! {len(instruction_dataset)} paires d'instructions générées et sauvegardées dans {output_path}")

if __name__ == "__main__":
    input_file = "data/raw_pokemons.json"
    output_file = "data/pokedex_instructions.json"
    
    format_to_instructions(input_file, output_file)