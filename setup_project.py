# setup_project.py
# Dieses Skript automatisiert das Projekt-Setup:
# 1. Herunterladen des Basis-Modells von Hugging Face.
# 2. Herunterladen des Trainings-Datensatzes von Kaggle (optional, falls nicht vorhanden).

import os
import kagglehub
from transformers import AutoTokenizer, AutoModel

# --- Konfiguration ---
KAGGLE_DATASET_ID = 'tobiasbueck/multilingual-customer-support-tickets'
EXPECTED_CSV_FILE = 'dataset-tickets-german_normalized_50_5_2.csv'

# Verzeichnisse
TRAININGS_DATA_DIR = './trainingsdaten'
MODEL_DIR = './distilbert-local'
MODEL_NAME = 'distilbert-base-uncased'


def download_huggingface_model():
    """Lädt das distilbert-base-uncased Modell und den Tokenizer herunter."""
    print("-" * 50)
    print(f"Prüfe Hugging Face Modell '{MODEL_NAME}'...")

    if os.path.exists(MODEL_DIR):
        print(f"✅ Verzeichnis '{MODEL_DIR}' existiert bereits. Download wird übersprungen.")
        return

    print(f"Lade Modell und Tokenizer für '{MODEL_NAME}' herunter...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(MODEL_DIR)
        model.save_pretrained(MODEL_DIR)
        print(f"✅ Modell und Tokenizer erfolgreich in '{MODEL_DIR}' gespeichert.")
    except Exception as e:
        print(f"❌ FEHLER beim Modell-Download: {e}")


def download_kaggle_dataset():
    """Lädt den Kaggle-Datensatz herunter, falls er nicht bereits manuell hinzugefügt wurde."""
    print("-" * 50)
    print(f"Prüfe Kaggle Datensatz '{KAGGLE_DATASET_ID}'...")

    final_csv_path = os.path.join(TRAININGS_DATA_DIR, EXPECTED_CSV_FILE)

    # Optional: Prüfen, ob die Datei bereits existiert
    if os.path.exists(final_csv_path):
        print(f"✅ Trainingsdaten '{final_csv_path}' existieren bereits. Download wird übersprungen.")
        return

    print("Trainingsdaten nicht gefunden. Versuche automatischen Download via KaggleHub...")
    try:
        # Erstelle das Zielverzeichnis
        os.makedirs(TRAININGS_DATA_DIR, exist_ok=True)

        # Lade den Datensatz herunter und speichere ihn im Zielverzeichnis
        kagglehub.dataset_download(KAGGLE_DATASET_ID, path=TRAININGS_DATA_DIR)

        # Hinweis: kagglehub entpackt automatisch, wenn möglich.
        # Falls die Zieldatei nach dem Download existiert, war es erfolgreich.
        if os.path.exists(final_csv_path):
            print(f"✅ Datensatz erfolgreich heruntergeladen und in '{TRAININGS_DATA_DIR}' abgelegt.")
        else:
            print(
                f"❌ FEHLER: Download schien erfolgreich, aber die Zieldatei '{EXPECTED_CSV_FILE}' wurde nicht gefunden.")
            print("Bitte lade den Datensatz manuell herunter (siehe README.md).")

    except Exception as e:
        print(f"❌ FEHLER beim Kaggle-Download: {e}")
        print("Der automatische Download ist fehlgeschlagen. Bitte folge der manuellen Anleitung in der README.md.")
        print("Mögliche Ursache: Fehlende 'kaggle.json' API-Datei.")


def main():
    """Führt alle Setup-Schritte aus."""
    print("Starte Projekt-Setup...")
    download_huggingface_model()
    download_kaggle_dataset()
    print("-" * 50)
    print("🎉 Setup abgeschlossen!")


if __name__ == "__main__":
    main()