import os
import json
import time
import requests

def fetch_pokemon_data(limit=151):
    print(f"Début de l'extraction des {limit} premiers Pokémon...")
    base_url = "https://pokeapi.co/api/v2/pokemon/"
    pokemons_list = []
    echecs = []  # IDs n'ayant pas pu être récupérés

    for idx in range(1, limit + 1):
        # Sécurité : On attend 0.5 seconde entre chaque requête pour respecter l'API
        time.sleep(0.5)

        try:
            response = requests.get(f"{base_url}{idx}", timeout=10)
            if response.status_code == 200:
                data = response.json()

                # On extrait uniquement les informations utiles pour le LLM
                pokemon_info = {
                    "id": data["id"],
                    "name": data["name"].capitalize(),
                    "types": [t["type"]["name"] for t in data["types"]],
                    "stats": {s["stat"]["name"]: s["base_stat"] for s in data["stats"]},
                    "abilities": [a["ability"]["name"] for a in data["abilities"] if not a["is_hidden"]],
                    "height": data["height"] / 10, # PokéAPI donne la hauteur en décimètres -> conversion en mètres
                    "weight": data["weight"] / 10  # PokéAPI donne le poids en hectogrammes -> conversion en kg
                }

                pokemons_list.append(pokemon_info)
                print(f" [{idx}/{limit}] {pokemon_info['name']} récupéré avec succès.")
            else:
                print(f" ❌ Erreur pour le Pokémon ID {idx} : Code {response.status_code}")
                echecs.append(idx)

        except requests.exceptions.RequestException as e:
            print(f" 💥 Erreur réseau lors de la requête pour l'ID {idx} : {e}")
            echecs.append(idx)

    # Bilan : signaler les éventuels trous dans l'extraction
    print(f"\n📊 {len(pokemons_list)}/{limit} Pokémon récupérés.")
    if echecs:
        print(f" ⚠️ {len(echecs)} échec(s) — IDs non récupérés : {echecs}")

    return pokemons_list

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extraction des Pokémon depuis PokéAPI")
    parser.add_argument("--limit", type=int, default=151,
                        help="Nombre de Pokémon à extraire (défaut : 151, réduire pour la CI)")
    args = parser.parse_args()

    # S'assurer que le dossier data existe
    os.makedirs("data", exist_ok=True)
    
    # Lancement de l'extraction
    all_data = fetch_pokemon_data(limit=args.limit)
    
    # Sauvegarde dans le dossier data/raw_pokemons.json
    output_path = "data/raw_pokemons.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n Extraction terminée ! Fichier sauvegardé sous : {output_path}")