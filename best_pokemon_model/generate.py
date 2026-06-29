import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generer_reponse_pokemon(instruction):
    # Le modèle se trouve dans le même dossier que ce script,
    # quel que soit le répertoire depuis lequel on le lance.
    model_path = os.path.dirname(os.path.abspath(__file__))
    
    print("Chargement de ton LLM Pokémon personnalisé...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    # Placer le modèle sur le GPU si disponible, sinon sur le CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Respecter EXACTEMENT le même format (Prompt Template) que pendant l'entraînement
    prompt = f"### Instruction:\n{instruction}\n\n### Réponse:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("\nLe LLM réfléchit...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,     # Longueur max de la description du Pokémon
            temperature=0.7,        # Un peu de créativité (0.1 = très robotique, 1.0 = très créatif)
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Décoder le texte généré
    reponse_complete = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Nettoyer l'affichage pour ne voir que la réponse du LLM
    reponse_propre = reponse_complete.split("### Réponse:\n")[-1]
    return reponse_propre

if __name__ == "__main__":
    # Teste ton modèle ici avec une question présente dans ton fichier JSON !
    question = "Donne-moi la fiche Pokedex de Pikachu."
    
    reponse = generer_reponse_pokemon(question)
    print("="*40)
    print(f"Question : {question}")
    print(f"Réponse du modèle :\n{reponse}")
    print("="*40)