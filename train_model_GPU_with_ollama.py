# train_model.py

# Erforderliche Bibliotheken importieren
import os
import sys
import time
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, ClassLabel
import pandas as pd  # <<< NEU HINZUGEFÜGT >>>
import ollama  # <<< NEU HINZUGEFÜGT >>>
import json  # <<< NEU HINZUGEFÜGT >>>
from tqdm import tqdm  # <<< NEU HINZUGEFÜGT >>> (Für eine schöne Fortschrittsanzeige)

# === Konfiguration für die Ollama-Analyse ===
# <<< NEU HINZUGEFÜGT: Kompletter Konfigurationsblock >>>
OLLAMA_MODEL = 'gemma2-gpu'  # Das zu verwendende Ollama-Modell
OLLAMA_OUTPUT_CSV = 'trainingsdaten/dataset-tickets-german_ergaenzt_ollama.csv'  # Name der Ausgabedatei

# Gewichtung für die Berechnung des Gesamtwertes. Passen Sie diese bei Bedarf an.
OLLAMA_SCORE_WEIGHTS = {
    'urgency': 0.3,  # Einfluss der von Ollama bewerteten Dringlichkeit
    'impact': 0.3,  # Einfluss der von Ollama bewerteten Auswirkung
    'priority': 0.4  # Einfluss der ursprünglichen Priorität aus den Daten
}


# === Ende der Ollama-Konfiguration ===


def analyze_text_with_ollama(text: str) -> dict:
    """
    Sendet Text an Ollama zur Analyse von Dringlichkeit und Auswirkung.
    """
    prompt = f"""
    Analysiere das folgende Ticket. Bewerte "Dringlichkeit" und "Auswirkung" auf einer Skala von 1 (sehr niedrig) bis 10 (sehr hoch).
    Antworte ausschließlich mit einem JSON-Objekt im Format {{"Dringlichkeit": <Wert>, "Auswirkung": <Wert>}}.

    Ticket: "{text}"
    """
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            format='json'
        )
        data = json.loads(response['message']['content'])
        if 'Dringlichkeit' in data and 'Auswirkung' in data:
            return {
                'Dringlichkeit': int(data['Dringlichkeit']),
                'Auswirkung': int(data['Auswirkung'])
            }
        return {'Dringlichkeit': 0, 'Auswirkung': 0}
    except Exception as e:
        # <<< GEÄNDERT: Viel bessere Fehleranzeige >>>
        print("\n" + "="*80)
        print("!!!!!!!!! FEHLER BEI DER OLLAMA-ANALYSE !!!!!!!!!")
        print(f"Beim Versuch, das Modell '{OLLAMA_MODEL}' zu verwenden, ist ein Fehler aufgetreten.")
        print(f"Fehlermeldung: {e}")
        print("="*80 + "\n")
        return {'Dringlichkeit': 0, 'Auswirkung': 0}


# <<< NEU HINZUGEFÜGT: Funktion zur Score-Berechnung >>>
def calculate_score(urgency: int, impact: int, priority_value: int, max_priority: int) -> float:
    """
    Berechnet einen gewichteten Score.
    """
    # Normalisiert Ollama-Werte (1-10) auf die Skala der Priorität (z.B. 1-5)
    normalized_urgency = (urgency / 10) * max_priority
    normalized_impact = (impact / 10) * max_priority

    score = (
            normalized_urgency * OLLAMA_SCORE_WEIGHTS['urgency'] +
            normalized_impact * OLLAMA_SCORE_WEIGHTS['impact'] +
            priority_value * OLLAMA_SCORE_WEIGHTS['priority']
    )
    return round(score, 2)


# <<< NEU HINZUGEFÜGT: Hauptfunktion für die Ollama-Analyse >>>
def run_ollama_analysis(dataset, priority_order):
    """
    Führt die Ollama-Analyse für das gesamte Dataset durch und speichert das Ergebnis.
    """
    print("\n--- Starte Ollama-Voranalyse ---")

    # Konvertiere das 'datasets'-Objekt in einen Pandas DataFrame für leichtere Handhabung
    df = dataset['train'].to_pandas()

    # Erstelle ein Mapping von Prioritäts-Namen zu numerischen Werten (z.B. critical=5, high=4, ...)
    # Höherer Wert = höhere Priorität
    priority_mapping = {label: len(priority_order) - i for i, label in enumerate(priority_order)}
    max_priority_value = len(priority_order)

    results = []
    total_rows = df.shape[0]  # Gesamtzahl für die Anzeige holen

    # Iteriere mit einer Fortschrittsanzeige (tqdm) durch das DataFrame
    for index, row in tqdm(df.iterrows(), total=total_rows, desc="Analysiere Tickets"):
        combined_text = str(row["subject"]) + " " + str(row["body"])

        # <<< HINZUGEFÜGT: Konsolenausgabe vor dem Ollama-Aufruf >>>
        # Gibt eine gekürzte Version des zu analysierenden Textes aus.
        print(f"\n[INFO] Verarbeite Ticket {index + 1}/{total_rows}: '{combined_text[:90]}...'")

        # Analyse mit Ollama
        analysis_result = analyze_text_with_ollama(combined_text)

        # <<< HINZUGEFÜGT: Konsolenausgabe nach dem Ollama-Aufruf >>>
        print(f"↳ ✅ Ollama-Analyse abgeschlossen: {analysis_result}")

        # Numerische Priorität holen
        priority_text = row['priority']
        priority_value = priority_mapping.get(priority_text, 0)

        # Score berechnen
        score = calculate_score(
            analysis_result['Dringlichkeit'],
            analysis_result['Auswirkung'],
            priority_value,
            max_priority_value
        )

        results.append({
            'Dringlichkeit_Ollama': analysis_result['Dringlichkeit'],
            'Auswirkung_Ollama': analysis_result['Auswirkung'],
            'Score_Gesamt': score
        })

    # Füge die Ergebnisse als neue Spalten zum DataFrame hinzu
    results_df = pd.DataFrame(results)
    df_erweitert = pd.concat([df.reset_index(drop=True), results_df], axis=1)

    # Speichere die erweiterte Datei
    df_erweitert.to_csv(OLLAMA_OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"\n✅ Analyse abgeschlossen. Erweiterte Daten wurden in '{OLLAMA_OUTPUT_CSV}' gespeichert.")
    print("Vorschau der ersten Zeilen:")
    print(df_erweitert.head())
    print("--------------------------------\n")


def main():
    """
    Diese Funktion steuert den gesamten Prozess:
    1. Konfiguration und Vorab-Prüfungen durchführen
    2. Daten laden
    3. (Optional) Ollama-Analyse durchführen
    4. Modell und Tokenizer vorbereiten
    5. Daten verarbeiten
    6. Modell trainieren
    7. Modell speichern
    """
    print("Starte den Trainingsprozess...")

    # === Schritt 1: Konfiguration, Diagnose und Geräte-Prüfung ===

    if torch.cuda.is_available():
        print("✅ GPU gefunden! Das Training wird auf der GPU ausgeführt. 🚀")
    else:
        print("⚠️ Keine GPU gefunden. Das Training wird auf der CPU ausgeführt (deutlich langsamer).")

    print(f"➡️  Aktuelles Arbeitsverzeichnis: {os.getcwd()}")

    output_dir = "./ergebnisse"
    base_log_dir = "logs"

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

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    run_log_dir = os.path.join(base_log_dir, f"{timestamp}_{script_name}")
    print(f"Logs für diesen Durchlauf werden in '{run_log_dir}' gespeichert.")

    # === Schritt 2: Dataset laden ===
    print("Lade das Dataset...")
    try:
        # <<< GEÄNDERT: Bessere Fehlerbehandlung, falls die Datei nicht existiert >>>
        dataset_path = 'trainingsdaten/dataset-tickets-german_normalized_50_5_2.csv'
        if not os.path.exists(dataset_path):
            print(f"❌ Fehler: Die Dataset-Datei '{dataset_path}' wurde nicht gefunden.")
            sys.exit()
        dataset = load_dataset('csv', data_files=dataset_path)
    except Exception as e:
        print(f"❌ Ein Fehler ist beim Laden des Datasets aufgetreten: {e}")
        sys.exit()

    # === Schritt 3: Label-Spalte mit logischer Reihenfolge vorbereiten ===
    priority_order = ["critical", "high", "medium", "low", "very_low"]

    # === Schritt 3a: Optionale Ollama-Voranalyse durchführen ===
    # <<< NEU HINZUGEFÜGT: Interaktive Abfrage für die Ollama-Analyse >>>
    while True:
        run_analysis_choice = input("Möchten Sie eine Vorab-Analyse der Daten mit Ollama durchführen? (j/n): ").lower()
        if run_analysis_choice in ['j', 'ja']:
            run_ollama_analysis(dataset, priority_order)
            break
        elif run_analysis_choice in ['n', 'nein']:
            print("✅ Ollama-Analyse wird übersprungen.")
            break
        else:
            print("Ungültige Eingabe. Bitte 'j' für Ja oder 'n' für Nein eingeben.")

    print("\n--- Setze Trainingsprozess fort ---")
    print("Wandle die 'priority'-Spalte in Klassen-Labels mit fester Reihenfolge um...")
    # Wir nehmen die ursprünglichen Text-Labels, bevor wir sie in Zahlen umwandeln
    original_labels_for_map = dataset['train']['priority']
    class_label_feature = ClassLabel(names=priority_order)

    dataset = dataset.map(
        lambda examples: {"priority": class_label_feature.str2int(examples["priority"])},
        batched=True
    )
    dataset['train'].features['priority'] = class_label_feature
    num_unique_labels = len(priority_order)
    print(f"✅ 'priority'-Spalte erfolgreich in {num_unique_labels} Labels mit logischer Reihenfolge umgewandelt.")

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