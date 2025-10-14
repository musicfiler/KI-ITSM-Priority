# train_model.py

# Erforderliche Bibliotheken importieren
import os
import sys
import time
import torch  # Hinzugefügt, um die GPU-Verfügbarkeit zu prüfen
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, ClassLabel # ClassLabel für die feste Label-Zuordnung


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

    # GPU-Verfügbarkeit prüfen, um das Training zu beschleunigen
    if torch.cuda.is_available():
        print("✅ GPU gefunden! Das Training wird auf der GPU ausgeführt. 🚀")
    else:
        print(
            "⚠️ Keine GPU gefunden oder PyTorch ist nicht für GPU konfiguriert. Das Training wird auf der CPU ausgeführt (deutlich langsamer).")

    # Diagnose: Ausgeben, in welchem Ordner das Skript arbeitet
    print(f"➡️  Aktuelles Arbeitsverzeichnis: {os.getcwd()}")

    output_dir = "./ergebnisse"
    base_log_dir = "logs"

    # Automatische Konfliktlösung, falls eine Datei namens 'logs' existiert
    if os.path.isfile(base_log_dir):
        backup_name = f"logs_als_datei_gesichert_{int(time.time())}.txt"
        print(f"⚠️  Warnung: Eine Datei namens '{base_log_dir}' blockiert die Erstellung des Log-Verzeichnisses.")
        print(f"✅ Die Datei wird sicher umbenannt in '{backup_name}'.")
        os.rename(base_log_dir, backup_name)

    # Abfrage, ob bestehende Ergebnisse überschrieben werden sollen
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

    # Eindeutiges Log-Verzeichnis für jeden Trainingslauf erstellen
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    run_log_dir = os.path.join(base_log_dir, f"{timestamp}_{script_name}")
    print(f"Logs für diesen Durchlauf werden in '{run_log_dir}' gespeichert.")

    # === Schritt 2: Dataset laden ===
    # Wir laden die CSV-Datei aus dem richtigen Unterordner.
    print("Lade das Dataset...")
    dataset = load_dataset('csv', data_files='trainingsdaten/dataset-tickets-german_normalized_50_5_2.csv')

    # === Schritt 3: Label-Spalte mit logischer Reihenfolge vorbereiten ===
    # Wir definieren die Label-Reihenfolge explizit, um eine logische Zuordnung
    # (ID 0 = höchste Prio, ID 4 = niedrigste Prio) zu garantieren.
    priority_order = [
        "critical",
        "high",
        "medium",
        "low",
        "very_low"
    ]
    print("Wandle die 'priority'-Spalte in Klassen-Labels mit fester Reihenfolge um...")
    class_label_feature = ClassLabel(names=priority_order)
    dataset = dataset.map(
        lambda examples: {"priority": class_label_feature.str2int(examples["priority"])},
        batched=True
    )
    dataset['train'].features['priority'] = class_label_feature
    num_unique_labels = len(priority_order)
    print(f"✅ 'priority'-Spalte erfolgreich in {num_unique_labels} Labels mit logischer Reihenfolge umgewandelt.")


    # === Schritt 4: Basis-Modell und Tokenizer laden ===
    # Wir übergeben die eben ermittelte Anzahl an Labels direkt an das Modell,
    # damit es die richtige Anzahl an Ausgabe-Neuronen hat.
    print("Lade das Basis-Modell und den Tokenizer...")
    modell_name = "./distilbert-local"
    tokenizer = AutoTokenizer.from_pretrained(modell_name)
    model = AutoModelForSequenceClassification.from_pretrained(modell_name, num_labels=num_unique_labels)

    # === Schritt 5: Tokenize-Funktion definieren und anwenden ===
    # Diese Funktion kombiniert Betreff und Textkörper und wandelt sie in Token-IDs um,
    # die das Modell versteht.
    def tokenize_function(examples):
        combined_texts = [str(subject) + " " + str(body) for subject, body in
                          zip(examples["subject"], examples["body"])]
        return tokenizer(combined_texts, padding="max_length", truncation=True)
    print("Tokenisiere das Dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # === Schritt 6: Finale Vorbereitung der Labels für den Trainer ===
    # Wir benennen die Spalte 'priority' in 'labels' um, da der Trainer diesen Namen erwartet.
    print("Benenne die 'priority'-Spalte in 'labels' um...")
    tokenized_datasets = tokenized_datasets.rename_column("priority", "labels")
    # Optional: Nicht mehr benötigte Spalten entfernen, um das Dataset übersichtlich zu halten.
    tokenized_datasets = tokenized_datasets.remove_columns(['subject', 'body', 'queue', 'language'])

    # === Schritt 7: Trainings-Argumente definieren ===
    # Hier legen wir alle Hyperparameter für das Training fest.
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",  # Keine Evaluierung während des Trainings
        num_train_epochs=3,  # Anzahl der Trainingsdurchläufe
        per_device_train_batch_size=8,  # Wie viele Beispiele pro Schritt verarbeitet werden
        logging_dir=run_log_dir,
        overwrite_output_dir=overwrite_output,
        report_to="none",  # Deaktiviert TensorBoard-Logging, um den Fehler zu umgehen
    )

    # === Schritt 8: Trainer initialisieren ===
    # Der Trainer verbindet das Modell, die Trainingsargumente und die Datensätze.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )

    # === Schritt 9: Training starten ===
    # Dieser Befehl startet den eigentlichen Feinabstimmungsprozess.
    print("Starte das Training...")
    trainer.train()

    print("\n🎉 Training erfolgreich abgeschlossen! Das Modell wurde im Ordner './ergebnisse' gespeichert.")


# Dieser Standard-Block führt die main()-Funktion aus, wenn das Skript direkt gestartet wird.
if __name__ == "__main__":
    main()