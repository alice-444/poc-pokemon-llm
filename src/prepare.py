import os
import json

def format_to_instructions(raw_data_path, output_path):
    print(f"Transformation des données depuis {raw_data_path}...")
    
    with open(raw_data_path, "r", encoding="utf-8") as f:
        pokemons = json.load(f)
    
    instruction_dataset = []
    
    for p in pokemons:
        # Traduction rapide des types pour avoir des prompts en Français propre
        types_fr = ", ".join(p["types"]).replace("electric", "Électrik").replace("fire", "Feu").replace("water", "Eau").replace("grass", "Plante").replace("poison", "Poison").replace("flying", "Vol").replace("bug", "Insecte").replace("normal", "Normal")
        
        # Formatter les statistiques pour le texte
        stats_txt = f"PV: {p['stats']['hp']}, Attaque: {p['stats']['attack']}, Défense: {p['stats']['defense']}, Vitesse: {p['stats']['speed']}"
        talents = ", ".join(p["abilities"])
        
        # La réponse textuelle que le LLM devra générer
        reponse = (
            f"{p['name']} est un Pokémon de type {types_fr}. "
            f"Il mesure {p['height']}m et pèse {p['weight']}kg. "
            f"Ses talents sont : {talents}. "
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