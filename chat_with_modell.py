# chat_with_modell.py (Modular / Multi-Modell / Bugfix)

import os
import re
import sys
import torch
import pandas as pd
import numpy as np
import glob  # Zum Finden von Modell-Ordnern
import json  # Zum Lesen von config.json und trainer_state.json
import time  # Für Wartezeit, wenn keine Modelle gefunden werden

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Importe für die Validierungs-Funktion
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Globale Konfiguration für Vokabulare ---
BASE_DIR = "trainingsdaten"
NEG_CSV = os.path.join(BASE_DIR, "neg_arguments_multilingual.csv")
POS_CSV = os.path.join(BASE_DIR, "pos_arguments_multilingual.csv")
SLA_CSV = os.path.join(BASE_DIR, "sla_arguments_multilingual.csv")

# Standard-Datei für die "Batch-Prüfung" (KORRIGIERTER PFAD)
DEFAULT_DATA_FILE = os.path.join(BASE_DIR, "5_prio_stufen/dataset-tickets-german_normalized_50_5_2.csv")

# Die speziellen Tokens (müssen im Vokabular-Preprocessing übereinstimmen)
NEW_TOKENS = ['KEY_CORE_APP', 'KEY_CRITICAL', 'KEY_REQUEST', 'KEY_NORMAL']

# Die Gewichte für das Preprocessing (aus den Trainings-Skripten)
SLA_WEIGHT = 5
NEG_WEIGHT = 4
POS_WEIGHT = 1

# Standard-Prioritätsliste als Fallback
DEFAULT_PRIORITY_ORDER = [
    "critical",
    "high",
    "medium",
    "low",
    "very_low"
]


# ==============================================================================
# HILFSFUNKTIONEN (EXAKT KOPIERT AUS DEM TRAININGS-SKRIPT)
# ==============================================================================

def load_vocab_from_csvs() -> (list, list, list):
    """Lädt die Vokabularlisten aus den CSV-Dateien."""
    print("Lade Vokabular-Listen aus CSV-Dateien...")
    try:
        df_neg = pd.read_csv(NEG_CSV)
        df_pos = pd.read_csv(POS_CSV)
        df_sla = pd.read_csv(SLA_CSV)

        # Stelle sicher, dass alles als String geladen wird, auch Zahlen
        neg_vocab = df_neg['term'].dropna().astype(str).tolist()
        pos_vocab = df_pos['term'].dropna().astype(str).tolist()
        sla_vocab = df_sla['term'].dropna().astype(str).tolist()

        print(f"  {len(neg_vocab)} negative, {len(pos_vocab)} positive, {len(sla_vocab)} SLA-Begriffe geladen.")
        return neg_vocab, pos_vocab, sla_vocab

    except FileNotFoundError as e:
        print(f"❌ FEHLER: Vokabular-Datei nicht gefunden: {e}")
        print("Stelle sicher, dass die CSV-Dateien im Ordner 'trainingsdaten' liegen.")
        sys.exit()
    except KeyError:
        print("❌ FEHLER: CSV-Dateien müssen eine Spalte namens 'term' haben.")
        sys.exit()


def preprocess_with_vocab(
        text: str,
        neg_vocab: list,
        pos_vocab: list,
        sla_vocab: list,
        sla_weight: int = 1,
        neg_weight: int = 1,
        pos_weight: int = 1
) -> str:
    """
    Reichert einen Text mit speziellen Signal-Wörtern (KEY_...) an.
    """
    if not isinstance(text, str):
        return ""

    text_lower = text.lower()

    if any(re.search(r'\b' + re.escape(p) + r'\b', text_lower) for p in sla_vocab):
        feature_string = " ".join(["KEY_CORE_APP"] * sla_weight)
        return f"{feature_string} [SEP] {text}"

    if any(re.search(r'\b' + re.escape(p) + r'\b', text_lower) for p in neg_vocab):
        feature_string = " ".join(["KEY_CRITICAL"] * neg_weight)
        return f"{feature_string} [SEP] {text}"

    if any(re.search(r'\b' + re.escape(p) + r'\b', text_lower) for p in pos_vocab):
        feature_string = " ".join(["KEY_REQUEST"] * pos_weight)
        return f"{feature_string} [SEP] {text}"

    feature_string = "KEY_NORMAL"
    return f"{feature_string} [SEP] {text}"


# ==============================================================================
# INFERENZ- / CHAT-FUNKTIONEN
# ==============================================================================

def load_model_and_tokenizer(model_path):
    """Lädt das trainierte Modell und den Tokenizer aus dem gewählten Pfad."""
    print(f"Lade trainiertes Modell aus '{model_path}'...")
    try:
        # Lade Tokenizer und Modell aus dem Stammverzeichnis des Ergebnisordners
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except OSError:
        print(f"❌ FEHLER: Modell-Ordner '{model_path}' nicht gefunden oder korrupt.")
        print("Stelle sicher, dass 'config.json' und 'pytorch_model.bin' vorhanden sind.")
        return None, None, None  # KORRIGIERT: Gebe Tupel zurück
    except Exception as e:
        print(f"❌ FEHLER beim Laden des Modells: {e}")
        return None, None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()  # Wichtig: Modell in den Inferenz-Modus schalten

    print(f"✅ Modell erfolgreich geladen und auf '{device}' verschoben.")
    return model, tokenizer, device


def predict_priority(subject, body, model, tokenizer, device, vocabs, priority_list, max_len):
    """
    Führt eine einzelne Vorhersage für ein neues Ticket durch.
    Akzeptiert jetzt 'priority_list' und 'max_len'.
    """
    neg_vocab, pos_vocab, sla_vocab = vocabs

    # 1. Text kombinieren
    raw_text = str(body) + " " + str(subject)

    # 2. Text anreichern
    enriched_text = preprocess_with_vocab(
        raw_text,
        neg_vocab, pos_vocab, sla_vocab,
        sla_weight=SLA_WEIGHT,
        neg_weight=NEG_WEIGHT,
        pos_weight=POS_WEIGHT
    )

    # 3. Tokenisieren
    # KORRIGIERT: Verwendet die 'max_len' aus der config.json (oder Fallback)
    inputs = tokenizer(
        enriched_text,
        padding="max_length",
        truncation=True,
        max_length=max_len,  # Dynamische Länge
        return_tensors="pt"
    )

    inputs = {key: val.to(device) for key, val in inputs.items()}

    # 5. Vorhersage durchführen
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)[0]
    predicted_index = torch.argmax(probabilities).item()

    predicted_label = priority_list[predicted_index]
    confidence = probabilities[predicted_index].item()

    return predicted_label, confidence


# ==============================================================================
# VALIDIERUNGS-MODUS (KORRIGIERT)
# ==============================================================================

def run_batch_evaluation(model, tokenizer, device, vocabs, priority_list, max_len):
    """
    Führt die Batch-Prüfung auf 50 zufälligen Samples einer CSV-Datei durch.
    Akzeptiert 'priority_list' und 'max_len'.
    """
    print("\n--- Batch-Prüfung (50 Samples) ---")

    # 1. Dynamische CSV-Abfrage
    default_file = DEFAULT_DATA_FILE
    filepath = input(f"Pfad zur CSV-Datei eingeben (Standard: {default_file}): ")
    if not filepath:
        filepath = default_file

    # 2. CSV laden
    try:
        df = pd.read_csv(filepath)
        print(f"Lade {len(df)} Zeilen aus {filepath}...")
    except FileNotFoundError:
        print(f"❌ FEHLER: Datei nicht gefunden: {filepath}")
        return
    except Exception as e:
        print(f"❌ FEHLER beim Lesen der CSV: {e}")
        return

    # 3. Spalten prüfen und 50 Samples ziehen
    try:
        # KORRIGIERT: Prüfe auf die Spaltennamen, die im Trainings-Skript
        # *standardmäßig* verwendet werden.
        if not all(col in df.columns for col in ['subject', 'body', 'priority']):
            print("❌ FEHLER: CSV muss Spalten 'subject', 'body' und 'priority' enthalten.")
            print("   (Die Batch-Prüfung unterstützt derzeit keine übersetzten Spaltennamen.)")
            return
        df_sample = df.sample(n=50, random_state=42)
    except ValueError:
        print(f"❌ FEHLER: Datei hat weniger als 50 Zeilen. Bitte größere Datei wählen.")
        return
    except Exception as e:
        print(f"❌ FEHLER: {e}")
        return

    print(f"Starte Vorhersage für 50 zufällige Tickets (Modell mit {len(priority_list)} Labels)...")

    predictions = []
    originals = []

    # 4. Vorhersage-Schleife
    for index, row in df_sample.iterrows():
        original_label = row['priority']
        subject = row['subject']
        body = row['body']

        predicted_label, _ = predict_priority(subject, body, model, tokenizer, device, vocabs, priority_list, max_len)

        predictions.append(predicted_label)
        originals.append(original_label)

    print("✅ Batch-Prüfung abgeschlossen.")
    print("-" * 30)

    # 5. Ergebnisse auswerten (Text)
    accuracy = accuracy_score(originals, predictions)
    print(f"Gesamt-Genauigkeit (Accuracy) der 50 Samples: {accuracy:.2%}")
    print("\nDetail-Auswertung (Classification Report):")

    # Stelle sicher, dass die Labels für den Report mit der 'priority_list' übereinstimmen
    report_labels = [label for label in priority_list if label in set(originals) | set(predictions)]

    report = classification_report(
        originals,
        predictions,
        labels=report_labels,
        target_names=report_labels,  # Zeige nur die Namen an, die auch gefunden wurden
        zero_division=0
    )
    print(report)
    print("-" * 30)

    # 6. Ergebnisse auswerten (Grafik)
    try:
        cm = confusion_matrix(originals, predictions,
                              labels=priority_list)  # Verwende die volle Liste für die Achsen-Reihenfolge

        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=priority_list,
            yticklabels=priority_list
        )
        plt.title(f'Konfusionsmatrix (50 Samples / Genauigkeit: {accuracy:.2%})')
        plt.ylabel('Wahres Label (aus CSV)')
        plt.xlabel('Vorhergesagtes Label (Modell)')

        output_filename = "konfusionsmatrix_check.png"
        plt.savefig(output_filename)
        print(f"✅ Grafik wurde als '{output_filename}' gespeichert.")

    except Exception as e:
        print(f"❌ FEHLER beim Erstellen der Grafik: {e}")
        print("  (Stelle sicher, dass 'matplotlib' und 'seaborn' installiert sind: pip install matplotlib seaborn)")

    print("-" * 30)


# ==============================================================================
# MODIFIZIERTE HAUPT-SCHLEIFEN
# ==============================================================================

def interactive_chat_mode(model, tokenizer, device, vocabs, priority_list, max_len):
    """
    Startet die interaktive Chat-Schleife.
    Akzeptiert 'priority_list' und 'max_len'.
    """
    print("\n--- Interaktiver KI-Priorisierungs-Chat ---")
    print(f"Modell: {os.path.basename(model.config.name_or_path)} ({len(priority_list)} Labels, max_len={max_len})")
    print("Gib 'exit' oder 'quit' ein, um zum Hauptmenü zurückzukehren.")
    print("-" * 30)

    while True:
        subject = input("Betreff (Subject): ")
        if subject.lower() in ["exit", "quit"]:
            break

        body = input("Text (Body):       ")
        if body.lower() in ["exit", "quit"]:
            break

        try:
            label, conf = predict_priority(subject, body, model, tokenizer, device, vocabs, priority_list, max_len)

            print("-" * 30)
            print(f"🤖 Vorhergesagte Priorität: >> {label.upper()} <<")
            print(f"   (Konfidenz: {conf:.2%})")
            print("-" * 30)

        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}")

    print("Interaktiver Modus beendet.")


def main_menu(model, tokenizer, device, vocabs, priority_list, max_len):
    """
    Zeigt das Hauptmenü NACHDEM ein Modell geladen wurde.
    Akzeptiert alle geladenen Komponenten.
    """
    while True:
        print("\n--- Hauptmenü KI-Priorisierung ---")
        print(
            f"✅ Geladenes Modell: {os.path.basename(model.config.name_or_path)} (Labels: {len(priority_list)}, max_len: {max_len})")
        print("1: Interaktiver Chat-Modus")
        print("2: Batch-Prüfung (50 zufällige Samples + Grafik)")
        print("q: Beenden (und neues Modell wählen)")

        choice = input("Wähle eine Option: ").strip().lower()

        if choice == '1':
            interactive_chat_mode(model, tokenizer, device, vocabs, priority_list, max_len)
        elif choice == '2':
            run_batch_evaluation(model, tokenizer, device, vocabs, priority_list, max_len)
        elif choice in ['q', 'exit', 'quit']:
            print("Zurück zur Modellauswahl.")
            # Lösche Modell und Tokenizer aus dem Speicher
            del model
            del tokenizer
            if device == "cuda":
                torch.cuda.empty_cache()
            break
        else:
            print("Ungültige Eingabe, bitte '1', '2' oder 'q' wählen.")


# ==============================================================================
# NEUE FUNKTIONEN: MODELLAUSWAHL (KORRIGIERT & GEPATCHT)
# ==============================================================================

def find_available_models():
    """
    Sucht nach allen 'ergebnisse_*' Ordnern und extrahiert Metriken
    und Label-Konfigurationen.

    KORRIGIERT: Lädt Modelle auch dann, wenn Checkpoints gelöscht wurden,
    solange config.json im Stammverzeichnis vorhanden ist.
    """
    print("Suche nach trainierten Modellen (Ordner 'ergebnisse_*')...")
    # KORRIGIERT: Füge "./" hinzu, um relative Pfade korrekt zu finden
    model_dirs = glob.glob("./ergebnisse_*")
    available_models = []

    for path in model_dirs:
        if not os.path.isdir(path):
            continue

        config_path = os.path.join(path, "config.json")

        # --- KORREKTUR: PRIMÄRE PRÜFUNG ---
        # Ein Modell ist gültig, wenn es eine config.json im Stammverzeichnis hat,
        # da 'trainer.save_model()' dies dort speichert.
        if not os.path.exists(config_path):
            print(f"ℹ️  '{path}' enthält keine 'config.json'. Überspringe.")
            continue

        try:
            # 1. Konfiguration (Labels) lesen
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            id2label = config.get("id2label", {})
            if not id2label:
                print(f"⚠️ Warnung: 'config.json' in {path} enthält keine 'id2label'-Info. Überspringe.")
                continue

            priority_list = [v for k, v in sorted(id2label.items(), key=lambda item: int(item[0]))]

            # Hole die Tokenizer-Länge (max_len), die beim Training verwendet wurde
            # Fallback auf 512 (alt) oder 256 (alt-alt)
            max_len = config.get("model_max_length", 512)  # DistilBERT-Limit
            if "max_position_embeddings" in config:
                max_len = config.get("max_position_embeddings", 512)  # Standard-Name

            # --- PATCH: Generische 'LABEL_0' Fehler abfangen ---
            is_generic_label = False
            if not priority_list or priority_list[0].startswith("LABEL_"):
                is_generic_label = True
                if len(priority_list) == len(DEFAULT_PRIORITY_ORDER):
                    priority_list = DEFAULT_PRIORITY_ORDER
                else:
                    print(
                        f"⚠️ Warnung: {path} hat generische Labels UND eine unerwartete Label-Anzahl ({len(priority_list)}).")
                    is_generic_label = False  # Patch konnte nicht angewendet werden

            # 2. Metriken (Optional)
            # Versuche, die Metriken aus dem *letzten* Checkpoint zu laden
            best_metric_val = "N/A"
            metric_name = "N/A"

            checkpoints = [
                d for d in os.listdir(path)
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(path, d))
            ]

            if checkpoints:
                try:
                    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
                    last_checkpoint_dir = checkpoints[-1]
                    state_path = os.path.join(path, last_checkpoint_dir, "trainer_state.json")

                    if os.path.exists(state_path):
                        with open(state_path, 'r', encoding='utf-8') as f:
                            state = json.load(f)
                        best_metric_val = state.get("best_metric", 0.0)
                        metric_name = state.get("metric_for_best_model", "best_metric")
                except Exception:
                    # Wenn das Lesen des Status fehlschlägt, ist es nicht schlim
                    metric_name = "Fehler beim Lesen"

            available_models.append({
                "path": path,
                "metric_val": best_metric_val,
                "metric_name": metric_name,
                "labels": priority_list,
                "is_generic_label": is_generic_label,
                "max_len": max_len  # NEU: Speichere die max_len
            })
        except Exception as e:
            print(f"Fehler beim Lesen der Konfiguration von {path}: {e}")

    return available_models


def select_model(models):
    """
    Zeigt dem Benutzer die gefundenen Modelle an und lässt ihn eines auswählen.
    """
    if not models:
        print("\n" + "=" * 50)
        print("❌ FEHLER: Keine gültigen, trainierten Modelle gefunden.")
        print("Stelle sicher, dass die Ordner './ergebnisse_*' existieren und")
        print("dass eine 'config.json' darin enthalten ist.")
        print("\nWarte 10 Sekunden und versuche es erneut...")
        print("=" * 50 + "\n")
        time.sleep(10)
        return None  # Signalisiert dem main-loop, es erneut zu versuchen

    print("\n--- Verfügbare trainierte Modelle ---")

    # Sortiere Modelle, sodass die Neuesten (nach Datum im Namen) oben sind
    try:
        models.sort(key=lambda x: x['path'], reverse=True)
    except Exception:
        pass  # Ignoriere Sortierfehler

    for i, model_info in enumerate(models):
        label_count = len(model_info['labels'])

        # Metrik formatieren (geht jetzt mit N/A um)
        if isinstance(model_info['metric_val'], float):
            metric_str = f"{model_info['metric_name']} = {model_info['metric_val']:.4f}"
        else:
            metric_str = f"{model_info['metric_name']} = {model_info['metric_val']}"

        labels_str = ", ".join(model_info['labels'])

        print(f"\n  [{i + 1}] {model_info['path']}")

        if model_info.get("is_generic_label"):
            print(
                f"      Labels ({label_count}): {labels_str}  <- [!] WARNUNG: config.json war fehlerhaft, Labels wurden ersetzt.")
        else:
            print(f"      Labels ({label_count}): {labels_str}")

        print(f"      Metrik: {metric_str}")
        print(f"      Max. Tokenlänge: {model_info['max_len']}")

    while True:
        try:
            choice_str = input(f"\nWähle ein Modell zum Laden (1-{len(models)}) [oder 'q' zum Beenden]: ")
            if choice_str.lower() in ['q', 'exit', 'quit']:
                sys.exit("Programm beendet.")

            choice_idx = int(choice_str) - 1
            if 0 <= choice_idx < len(models):
                return models[choice_idx]  # Gibt das ausgewählte Info-Dict zurück
            else:
                print(f"Ungültige Zahl. Bitte 1-{len(models)} wählen.")
        except ValueError:
            print("Ungültige Eingabe. Bitte eine Zahl wählen.")


# ==============================================================================
# NEUE HAUPT-FUNKTION (Startpunkt)
# ==============================================================================

def main():
    """
    Haupt-Startpunkt des Skripts.
    """
    # 1. Vokabulare einmalig laden
    try:
        vocabs = load_vocab_from_csvs()
    except Exception as e:
        print(f"Kritischer Fehler beim Laden der globalen Vokabulare: {e}")
        sys.exit()

    # 2. Hauptschleife für die Modellauswahl
    while True:
        models_list = find_available_models()
        selected_model_info = select_model(models_list)

        if selected_model_info is None:
            continue  # Springt zum Anfang der Schleife und sucht erneut

        MODEL_PATH = selected_model_info["path"]
        PRIORITY_ORDER = selected_model_info["labels"]
        MAX_LEN = selected_model_info["max_len"]  # NEU

        # 3. Ausgewähltes Modell laden
        model, tokenizer, device = load_model_and_tokenizer(MODEL_PATH)

        if model is None:
            print(f"Fehler beim Laden von {MODEL_PATH}. Kehre zur Auswahl zurück.")
            time.sleep(2)
            continue  # Kehre zur Modellauswahl zurück

        # 4. Starte das Untermenü mit den geladenen, modellspezifischen Daten
        main_menu(model, tokenizer, device, vocabs, PRIORITY_ORDER, MAX_LEN)


if __name__ == "__main__":
    main()