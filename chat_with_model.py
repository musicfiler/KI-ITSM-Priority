# chat_with_model.py
# Bisher nicht funktional

from transformers import pipeline
import os

def main():
    """
    This function loads the trained model and starts an interactive chat.
    """
    # Suchen Sie den neuesten Checkpoint im Ergebnisordner
    # Der Trainer speichert Modelle in Unterordnern wie "checkpoint-500", "checkpoint-1000" etc.
    # Wir nehmen hier an, dass der letzte Checkpoint der beste ist.
    try:
        checkpoints = [d for d in os.listdir("./ergebnisse") if d.startswith("checkpoint-")]
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        model_path = os.path.join("./ergebnisse", latest_checkpoint)
        print(f"Loading model from: {model_path}")
    except IndexError:
        print("Error: No checkpoint found. Please run train_model.py first.")
        return

    # === 1. Pipeline mit dem trainierten Modell erstellen ===
    classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)

    # === 2. Interaktiven Chat starten ===
    print("\n🤖 Chat with your classification model has started! Type 'exit' to end.")
    print("-" * 30)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "ende"]:
            print("Bot: Goodbye!")
            break

        prediction = classifier(user_input)[0]
        label_id = prediction['label']
        score = prediction['score']

        # Konvertieren Sie die Label-ID zurück in den ursprünglichen Text (optional, aber nützlich)
        # Die Zuordnung wird vom Tokenizer oder Modell gespeichert.
        # Hier zeigen wir das Label, das das Modell ausgibt (z.B. LABEL_0)
        print(f"Bot: I classify this as '{label_id}' with {score:.2%} confidence.")

if __name__ == "__main__":
    main()