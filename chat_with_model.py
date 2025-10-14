# chat_with_model.py

from transformers import pipeline
import os


def main():
    """
    Diese Funktion lädt das trainierte Modell und startet einen interaktiven Chat.
    """
    output_dir = "./ergebnisse"
    base_model_path = "./distilbert-local"  # Pfad zum ursprünglichen Tokenizer

    # --- Verbesserte Logik zum Finden des Modells ---
    model_path = output_dir
    if not os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        print(f"Kein finales Modell in '{output_dir}' gefunden. Suche nach dem letzten Checkpoint...")
        try:
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
            model_path = os.path.join(output_dir, latest_checkpoint)
        except IndexError:
            print(f"FEHLER: Kein Checkpoint in '{output_dir}' gefunden. Bitte zuerst train_model.py ausführen.")
            return

    print(f"Lade Modell von: {model_path}")
    print(f"Lade Tokenizer von: {base_model_path}")

    # === 1. Pipeline mit dem trainierten Modell erstellen ===
    try:
        # Getrennte Pfade für Modell und Tokenizer
        classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=base_model_path  # Verwende den Original-Tokenizer
        )
    except Exception as e:
        print(f"Fehler beim Laden der Pipeline: {e}")
        return

    # === Label-Mapping aus dem Modell auslesen ===
    id2label = classifier.model.config.id2label

    # === 2. Interaktiven Chat starten ===
    print("\n🤖 Chat mit dem Klassifizierungsmodell gestartet! Tippe 'exit' zum Beenden.")
    print("-" * 30)

    while True:
        user_input = input("Du: ")
        if user_input.lower() in ["exit", "quit", "ende"]:
            print("Bot: Auf Wiedersehen!")
            break

        if not user_input.strip():
            continue

        prediction = classifier(user_input)[0]
        label_str = prediction['label']
        score = prediction['score']

        # Label-ID in Text umwandeln
        label_id = int(label_str.split('_')[-1])
        human_readable_label = id2label[label_id]

        print(f"Bot: Ich klassifiziere dies als '{human_readable_label}' mit einer Konfidenz von {score:.2%}.")


if __name__ == "__main__":
    main()