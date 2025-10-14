# train_model.py

# Erforderliche Bibliotheken importieren
import os
import sys
import time
import torch  # Hinzugefügt, um die GPU-Verfügbarkeit zu prüfen
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset


def main():
    """
    Diese Funktion steuert den gesamten Prozess:
    1. Konfiguration und Vorab-Prüfungen durchführen
    2. Daten laden
    3. Modell und Tokenizer vorbereiten
    4. Daten verarbeiten (Tokenisierung und Label-Vorbereitung)
    5. Modell trainieren
    6. Modell speichern
    """
    print("Starte den Trainingsprozess...")

    # === Schritt 1: Konfiguration, Diagnose und Geräte-Prüfung ===

    # --- NEU: GPU-Verfügbarkeit prüfen ---
    if torch.cuda.is_available():
        print("✅ GPU gefunden! Das Training wird auf der GPU ausgeführt. 🚀")
    else:
        print(
            "⚠️ Keine GPU gefunden oder PyTorch ist nicht für GPU konfiguriert. Das Training wird auf der CPU ausgeführt (deutlich langsamer).")
    # --------------------------------------------------------

    # Diagnose #1: Wo wird das Skript ausgeführt?
    print(f"➡️  Aktuelles Arbeitsverzeichnis: {os.getcwd()}")

    output_dir = "./ergebnisse"
    base_log_dir = "logs"

    # (Der Rest der Konfiguration bleibt gleich)
    # ... (Code für Konfliktlösung, Abfrage zum Überschreiben, etc.)
    if os.path.isfile(base_log_dir):
        backup_name = f"logs_als_datei_gesichert_{int(time.time())}.txt"
        print(f"⚠️  Warnung: Eine Datei namens '{base_log_dir}' blockiert die Erstellung des Log-Verzeichnisses.")
        print(f"✅ Die Datei wird sicher umbenannt in '{backup_name}'.")
        os.rename(base_log_dir, backup_name)

    overwrite_output = False
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        print(f"⚠️  Es sind bereits Daten im Ausgabeverzeichnis '{output_dir}' vorhanden.")

        while True:
            choice = input("Möchten Sie die vorhandenen Ergebnisse überschreiben? (j/n): ").lower()
            if choice in ['j', 'ja']:
                overwrite_output = True
                print("✅ Vorhandene Daten werden überschrieben.")
                break
            elif choice in ['n', 'nein']:
                print("❌ Vorgang vom Benutzer abgebrochen.")
                sys.exit()
            else:
                print("Ungültige Eingabe. Bitte 'j' für Ja oder 'n' für Nein eingeben.")
    else:
        overwrite_output = False

    # Dynamisches Log-Verzeichnis für diesen Lauf erstellen
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    run_log_dir = os.path.join(base_log_dir, f"{timestamp}_{script_name}")
    print(f"Logs für diesen Durchlauf werden in '{run_log_dir}' gespeichert.")

    # === Schritt 2: Dataset laden ===
    print("Lade das Dataset...")
    dataset = load_dataset('csv', data_files='trainingsdaten/dataset-tickets-german_normalized_50_5_2.csv')

    # === Schritt 3: Label-Spalte vorbereiten und Anzahl ermitteln ===
    print("Wandle die 'priority'-Spalte in Klassen-Labels um...")
    dataset = dataset.class_encode_column("priority")
    num_unique_labels = dataset['train'].features['priority'].num_classes
    print(f"✅ {num_unique_labels} einzigartige Labels in der 'priority'-Spalte gefunden.")

    # === Schritt 4: Basis-Modell und Tokenizer laden ===
    print("Lade das Basis-Modell und den Tokenizer...")
    modell_name = "./distilbert-local"
    tokenizer = AutoTokenizer.from_pretrained(modell_name)
    model = AutoModelForSequenceClassification.from_pretrained(modell_name, num_labels=num_unique_labels)

    # === Schritt 5: Tokenize-Funktion definieren und anwenden ===
    def tokenize_function(examples):
        combined_texts = [str(subject) + " " + str(body) for subject, body in
                          zip(examples["subject"], examples["body"])]
        return tokenizer(combined_texts, padding="max_length", truncation=True)

    print("Tokenisiere das Dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # === Schritt 6: Finale Vorbereitung der Labels für den Trainer ===
    print("Benenne die 'priority'-Spalte in 'labels' um...")
    tokenized_datasets = tokenized_datasets.rename_column("priority", "labels")
    tokenized_datasets = tokenized_datasets.remove_columns(['subject', 'body', 'queue', 'language'])

    # === Schritt 7: Trainings-Argumente definieren ===
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_dir=run_log_dir,
        overwrite_output_dir=overwrite_output,
        report_to="none",
    )

    # === Schritt 8: Trainer initialisieren ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )

    # === Schritt 9: Training starten ===
    print("Starte das Training...")
    trainer.train()

    print("\n🎉 Training erfolgreich abgeschlossen! Das Modell wurde im Ordner './ergebnisse' gespeichert.")


if __name__ == "__main__":
    main()