import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

def train_pokemon_llm(ci_mode=False):
    # 1. Définir le modèle de base (léger pour le PoC)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    if not ci_mode:
        # 2. Configurer MLflow (uniquement en mode normal)
        import mlflow
        mlflow.set_experiment("pokemon-llm-finetuning")
        
        # Activer la sauvegarde automatique du modèle final dans les artefacts MLflow
        os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "True"
    
    print(f"Chargement du modèle et du tokenizer : {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Assigner un token de padding si non existant (cas classique de GPT/Llama)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("Chargement et préparation du dataset...")
    # Charger le fichier JSON généré à la Phase 3
    dataset = load_dataset("json", data_files={"train": "data/pokedex_instructions.json"})

    # Fonction pour formater et tokeniser chaque exemple selon le template attendu.
    # On tokenise SANS padding (le data collator s'en charge dynamiquement par batch)
    # et on masque la partie « prompt » dans les labels : la loss ne porte que sur la réponse.
    MAX_LENGTH = 256

    def format_prompts(examples):
        input_ids_list, attention_list, labels_list = [], [], []
        for inst, out in zip(examples["instruction"], examples["output"]):
            prompt = f"### Instruction:\n{inst}\n\n### Réponse:\n"
            # L'<eos> final apprend au modèle à s'arrêter ; il reste un label valide
            # (non masqué) car le padding est géré séparément avec label = -100.
            full = f"{prompt}{out}{tokenizer.eos_token}"

            full_ids = tokenizer(full, truncation=True, max_length=MAX_LENGTH)["input_ids"]
            prompt_len = len(tokenizer(prompt)["input_ids"])

            # labels = input_ids, mais on masque les tokens du prompt (loss = réponse seule)
            labels = list(full_ids)
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100

            input_ids_list.append(full_ids)
            attention_list.append([1] * len(full_ids))
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_list,
            "labels": labels_list,
        }

    tokenized_dataset = dataset.map(format_prompts, batched=True, remove_columns=dataset["train"].column_names)

    if ci_mode:
        # Mode CI : entraînement minimal pour valider le pipeline
        print("⚡ Mode CI activé : entraînement léger (1 epoch, 20 steps max, CPU)")
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,
            max_steps=20,                      # Limite à 20 steps pour la CI
            per_device_train_batch_size=2,
            logging_steps=5,
            save_steps=50,
            learning_rate=5e-5,
            weight_decay=0.01,
            report_to="none",                  # Pas de MLflow en CI
            logging_dir="./logs",
            fp16=False,                        # Pas de GPU en CI
            use_cpu=True,                      # Force le CPU
        )
    else:
        # 3. Configurer les arguments d'entraînement
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,                # 3 passages complets sur le dataset
            per_device_train_batch_size=2,     # Petite taille pour éviter les saturations mémoire (OOM)
            logging_steps=10,                  # Log des métriques toutes les 10 étapes
            save_steps=100,
            learning_rate=5e-5,
            weight_decay=0.01,
            report_to="mlflow",                # 💥 CRUCIAL : Dit à HF d'envoyer les logs à MLflow
            logging_dir="./logs",
            fp16=torch.cuda.is_available(),    # Active la précision mixte si un GPU est dispo
        )

    # 4. Initialiser le Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        # Padding dynamique par batch : pad les input_ids avec pad_token et les labels avec -100
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
        ),
    )

    # 5. Lancer l'entraînement
    if ci_mode:
        # En CI, pas de MLflow
        print("Début du Fine-Tuning en mode CI...")
        trainer.train()
        
        print("\nEntraînement CI terminé ! Sauvegarde du modèle...")
        trainer.save_model("./best_pokemon_model")
        tokenizer.save_pretrained("./best_pokemon_model")
    else:
        # Mode normal : entraînement sous la supervision de MLflow
        import mlflow
        print("Début du Fine-Tuning monitoré par MLflow...")
        with mlflow.start_run() as run:
            # Enregistrer manuellement des tags ou paramètres personnalisés si besoin
            mlflow.log_param("dataset_size", len(dataset["train"]))
            
            trainer.train()
            
            print("\nEntraînement terminé ! Sauvegarde du modèle...")
            trainer.save_model("./best_pokemon_model")
            tokenizer.save_pretrained("./best_pokemon_model")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tuning du modèle Pokémon LLM")
    parser.add_argument("--ci", action="store_true",
                        help="Mode CI : entraînement léger sans MLflow (1 epoch, 20 steps, CPU)")
    args = parser.parse_args()

    train_pokemon_llm(ci_mode=args.ci)