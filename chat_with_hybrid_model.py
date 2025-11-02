# chat_with_model.py

# Importiert die 'pipeline'-Funktion von transformers, um ein vortrainiertes Modell einfach zu nutzen.
from transformers import pipeline
# Importiert das 'os'-Modul für Interaktionen mit dem Betriebssystem, z.B. um Dateien zu finden.
import os


def main():
    """
    Diese Funktion lädt das trainierte Modell und startet einen interaktiven Chat.
    """
    # Definiert den Ordner, in dem die Trainingsergebnisse gespeichert sind.
    output_dir = "./ergebnisse_hybrid"
    # Definiert den Pfad zum ursprünglichen, nicht-feinabgestimmten Modell (wird für den Tokenizer benötigt).
    base_model_path = "./distilbert-local"

    # --- Verbesserte Logik zum Finden des Modells ---
    # Wir gehen zuerst davon aus, dass das finale Modell im Haupt-Ergebnisordner liegt.
    model_path = output_dir
    # Prüft, ob die Modelldatei 'pytorch_model.bin' im Hauptordner existiert.
    if not os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        # Wenn nicht, suchen wir nach Checkpoint-Unterordnern.
        print(f"Kein finales Modell in '{output_dir}' gefunden. Suche nach dem letzten Checkpoint...")
        try:
            # Erstellt eine Liste aller Ordner, die mit "checkpoint-" beginnen.
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            # Sortiert die Checkpoints numerisch (basierend auf der Zahl nach dem Bindestrich) und wählt den letzten aus.
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
            # Setzt den Pfad zum neuesten Checkpoint-Ordner.
            model_path = os.path.join(output_dir, latest_checkpoint)
        except IndexError:
            # Falls keine Checkpoints gefunden werden, wird eine Fehlermeldung ausgegeben und das Skript beendet.
            print(f"FEHLER: Kein Checkpoint in '{output_dir}' gefunden. Bitte zuerst train_model.py ausführen.")
            return

    # Gibt dem Benutzer Feedback, welches Modell und welcher Tokenizer geladen werden.
    print(f"Lade Modell von: {model_path}")
    print(f"Lade Tokenizer von: {base_model_path}")

    # === 1. Pipeline mit dem trainierten Modell erstellen ===
    try:
        # Erstellt die Klassifizierungs-Pipeline.
        classifier = pipeline(
            "text-classification",  # Gibt den Task an, den die Pipeline ausführen soll.
            model=model_path,       # Lädt das feinabgestimmte Modell aus unserem Checkpoint.
            tokenizer=base_model_path # Lädt den unveränderten Original-Tokenizer.
        )
    except Exception as e:
        # Fängt mögliche Fehler beim Laden ab und gibt eine verständliche Meldung aus.
        print(f"Fehler beim Laden der Pipeline: {e}")
        return

    # === Feste Zuordnung der Label-Namen basierend auf der Trainings-Logik ===
    # Diese Liste MUSS exakt mit der im train_model.py-Skript übereinstimmen.
    # Sie übersetzt die numerischen IDs (0, 1, 2, ...) in lesbare Namen.
    priority_order = [
        "critical",
        "high",
        "medium",
        "low",
        "very_low"
    ]

    # === 2. Interaktiven Chat starten ===
    # Gibt eine Startnachricht für den Benutzer aus.
    print("\n🤖 Chat mit dem Klassifizierungsmodell gestartet! Tippe 'exit' zum Beenden.")
    print("-" * 30)

    # Startet eine Endlosschleife für den interaktiven Chat.
    while True:
        # Wartet auf eine Texteingabe vom Benutzer.
        user_input = input("Du: ")
        # Prüft, ob der Benutzer den Chat beenden möchte.
        if user_input.lower() in ["exit", "quit", "ende"]:
            # Gibt eine Abschiedsnachricht aus.
            print("Bot: Auf Wiedersehen!")
            # Beendet die Endlosschleife.
            break

        # Ignoriert leere Eingaben (z.B. wenn der Benutzer nur Enter drückt).
        if not user_input.strip():
            continue

        # Übergibt die Benutzereingabe an die Pipeline zur Klassifizierung.
        # '[0]' wählt das Ergebnis mit der höchsten Wahrscheinlichkeit aus.
        prediction = classifier(user_input)[0]
        # Holt das vorhergesagte Label (z.B. 'LABEL_2').
        label_str = prediction['label']
        # Holt den Konfidenzwert der Vorhersage (z.B. 0.9233).
        score = prediction['score']

        # --- Label-ID in Text umwandeln ---
        # Extrahiert die Zahl aus dem Label-String (z.B. 'LABEL_2' -> 2).
        label_id = int(label_str.split('_')[-1])

        # Verwendet die extrahierte ID als Index, um den lesbaren Namen aus der Liste zu holen.
        if 0 <= label_id < len(priority_order):
            # Wenn die ID gültig ist, wird der entsprechende Name aus der Liste geholt.
            human_readable_label = priority_order[label_id]
        else:
            # Falls eine unerwartete ID zurückkommt, wird ein Fallback-Text verwendet.
            human_readable_label = "Unbekanntes Label"

        # Gibt die finale, formatierte Antwort des Bots aus.
        # '{score:.2%}' formatiert den Konfidenzwert als Prozentzahl mit zwei Nachkommastellen.
        print(f"Bot: Ich klassifiziere dies als '{human_readable_label}' mit einer Konfidenz von {score:.2%}.")


# Dieser Standard-Block führt die main()-Funktion aus, wenn das Skript direkt gestartet wird.
if __name__ == "__main__":
    main()