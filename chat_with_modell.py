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

# NEU: Standard-Prioritätsliste als Fallback,
# falls die config.json generische 'LABEL_0'-Namen enthält.
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

        neg_vocab = df_neg['term'].dropna().tolist()
        pos_vocab = df_pos['term'].dropna().tolist()
        sla_vocab = df_sla['term'].dropna().tolist()

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
    (Exakt kopiert aus dem Trainings-Skript)
    """
    if not isinstance(text, str):
        return ""

    text_lower = text.lower()

    # --- HIERARCHISCHE PRÜFUNG ---
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
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except OSError:
        print(f"❌ FEHLER: Modell-Ordner '{model_path}' nicht gefunden.")
        print("Stelle sicher, dass das Training abgeschlossen wurde und der Ordner existiert.")
        sys.exit()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()  # Wichtig: Modell in den Inferenz-Modus schalten

    print(f"✅ Modell erfolgreich geladen und auf '{device}' verschoben.")
    return model, tokenizer, device


def predict_priority(subject, body, model, tokenizer, device, vocabs, priority_list):
    """
    Führt eine einzelne Vorhersage für ein neues Ticket durch.
    Akzeptiert jetzt eine dynamische 'priority_list'.
    """
    neg_vocab, pos_vocab, sla_vocab = vocabs

    # 1. Text kombinieren (EXAKTE REIHENFOLGE WIE IM TRAINING!)
    raw_text = str(body) + " " + str(subject)

    # 2. Text anreichern (EXAKTE LOGIK WIE IM TRAINING!)
    enriched_text = preprocess_with_vocab(
        raw_text,
        neg_vocab, pos_vocab, sla_vocab,
        sla_weight=SLA_WEIGHT,
        neg_weight=NEG_WEIGHT,
        pos_weight=POS_WEIGHT
    )

    # 3. Tokenisieren
    #    max_length muss mit dem Training übereinstimmen (war 256)
    inputs = tokenizer(
        enriched_text,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"  # "pt" = PyTorch Tensors
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

def run_batch_evaluation(model, tokenizer, device, vocabs, priority_list):
    """
    Führt die Batch-Prüfung auf 50 zufälligen Samples einer CSV-Datei durch.
    Akzeptiert eine dynamische 'priority_list'.
    """
    print("\n--- Batch-Prüfung (50 Samples) ---")

    # 1. Dynamische CSV-Abfrage (KORRIGIERTER Standardpfad)
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
        if not all(col in df.columns for col in ['subject', 'body', 'priority']):
            print("❌ FEHLER: CSV muss Spalten 'subject', 'body' und 'priority' enthalten.")
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

        # Vorhersage durchführen
        predicted_label, _ = predict_priority(subject, body, model, tokenizer, device, vocabs, priority_list)

        predictions.append(predicted_label)
        originals.append(original_label)

    print("✅ Batch-Prüfung abgeschlossen.")
    print("-" * 30)

    # 5. Ergebnisse auswerten (Text)
    accuracy = accuracy_score(originals, predictions)
    print(f"Gesamt-Genauigkeit (Accuracy) der 50 Samples: {accuracy:.2%}")
    print("\nDetail-Auswertung (Classification Report):")

    # KORREKTUR: Die 'labels' für den Report müssen die Labels sein,
    # die *tatsächlich* in den 'originals' (aus der CSV) UND den 'predictions' vorkommen.
    # Wir verwenden 'priority_list' als Wunschliste, aber sklearn filtert automatisch.
    # Das Problem war der Label-Mismatch (z.B. 'LABEL_0' vs 'critical').
    # Da 'priority_list' jetzt dank unseres Fallbacks korrekt ist ('critical' etc.),
    # wird dieser Report nun funktionieren.

    # Ermittle alle einzigartigen Labels, die *wirklich* in den Daten vorkommen
    # (Dies ist robuster als nur 'priority_list' zu übergeben)
    unique_labels = sorted(list(set(originals) | set(predictions)))

    report = classification_report(
        originals,
        predictions,
        labels=unique_labels,  # Verwende die *tatsächlich* vorhandenen Labels
        target_names=priority_list,  # Gib die *Wunschnamen* an (wenn sie übereinstimmen)
        zero_division=0
    )
    print(report)
    print("-" * 30)

    # 6. Ergebnisse auswerten (Grafik)
    try:
        # KORREKTUR: Auch hier müssen die 'labels' die sein,
        # die in den Daten (originals, predictions) vorkommen.
        cm = confusion_matrix(originals, predictions, labels=unique_labels)

        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=unique_labels,  # Zeige die Achsenbeschriftungen
            yticklabels=unique_labels  # basierend auf den gefundenen Labels
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

def interactive_chat_mode(model, tokenizer, device, vocabs, priority_list):
    """
    Startet die interaktive Chat-Schleife.
    Akzeptiert 'priority_list'.
    """
    print("\n--- Interaktiver KI-Priorisierungs-Chat ---")
    print(f"Modell: {model.config.name_or_path} ({len(priority_list)} Labels)")
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
            label, conf = predict_priority(subject, body, model, tokenizer, device, vocabs, priority_list)

            print("-" * 30)
            print(f"🤖 Vorhergesagte Priorität: >> {label.upper()} <<")
            print(f"   (Konfidenz: {conf:.2%})")
            print("-" * 30)

        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}")

    print("Interaktiver Modus beendet.")


def main_menu(model, tokenizer, device, vocabs, priority_list):
    """
    Zeigt das Hauptmenü NACHDEM ein Modell geladen wurde.
    Akzeptiert alle geladenen Komponenten.
    """
    while True:
        print("\n--- Hauptmenü KI-Priorisierung ---")
        print(f"✅ Geladenes Modell: {os.path.basename(model.config.name_or_path)} ({len(priority_list)} Labels)")
        print("1: Interaktiver Chat-Modus")
        print("2: Batch-Prüfung (50 zufällige Samples + Grafik)")
        print("q: Beenden (und neues Modell wählen)")

        choice = input("Wähle eine Option: ").strip().lower()

        if choice == '1':
            interactive_chat_mode(model, tokenizer, device, vocabs, priority_list)
        elif choice == '2':
            run_batch_evaluation(model, tokenizer, device, vocabs, priority_list)
        elif choice in ['q', 'exit', 'quit']:
            print("Zurück zur Modellauswahl.")
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

    KORRIGIERT: Sucht 'trainer_state.json' im letzten Checkpoint-Ordner.
    PATCH: Fängt generische 'LABEL_0'-Fehler ab.
    """
    print("Suche nach trainierten Modellen (Ordner 'ergebnisse_*')...")
    model_dirs = glob.glob("ergebnisse_*")
    available_models = []

    for path in model_dirs:
        if not os.path.isdir(path):
            continue

        config_path = os.path.join(path, "config.json")

        # 1. Finde alle Checkpoint-Ordner
        checkpoints = [
            d for d in os.listdir(path)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(path, d))
        ]

        if not checkpoints:
            print(f"ℹ️  '{path}' enthält keine Checkpoints. Überspringe.")
            continue

        # 2. Sortiere sie numerisch (nach Schrittzahl)
        try:
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        except ValueError:
            print(f"⚠️  Warnung: Ungültiger Checkpoint-Name in '{path}'. Überspringe.")
            continue

        # 3. Wähle den letzten Checkpoint
        last_checkpoint_dir = checkpoints[-1]
        state_path = os.path.join(path, last_checkpoint_dir, "trainer_state.json")

        if os.path.exists(config_path) and os.path.exists(state_path):
            try:
                # 1. Konfiguration (Labels) lesen (aus dem Root-Ordner)
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                id2label = config.get("id2label", {})
                if not id2label:
                    print(f"⚠️ Warnung: 'config.json' in {path} enthält keine 'id2label'-Info. Überspringe.")
                    continue

                priority_list = [v for k, v in sorted(id2label.items(), key=lambda item: int(item[0]))]

                # --- NEUER PATCH ---
                # Prüfe, ob die Labels generisch sind (z.B. 'LABEL_0')
                is_generic_label = False
                if not priority_list or priority_list[0].startswith("LABEL_"):
                    is_generic_label = True
                    # Wenn ja, überschreibe sie mit der Standard-Liste
                    if len(priority_list) == len(DEFAULT_PRIORITY_ORDER):
                        priority_list = DEFAULT_PRIORITY_ORDER
                    else:
                        # Fallback für Modelle mit anderer Label-Anzahl (z.B. 3)
                        # Wir können die Namen nicht erraten, also behalten wir 'LABEL_0'
                        print(
                            f"⚠️ Warnung: {path} hat generische Labels UND eine unerwartete Label-Anzahl ({len(priority_list)}).")
                        # In diesem Fall werden 'LABEL_0' etc. beibehalten.
                        is_generic_label = False  # Zurücksetzen, da wir den Patch nicht anwenden konnten
                # --- ENDE PATCH ---

                # 2. Metriken (Trainer-Status) lesen (aus dem Checkpoint-Ordner)
                with open(state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)

                best_metric_val = state.get("best_metric", 0.0)
                metric_name = state.get("metric_for_best_model", "best_metric (unbekannt)")

                available_models.append({
                    "path": path,
                    "metric_val": best_metric_val,
                    "metric_name": metric_name,
                    "labels": priority_list,
                    "is_generic_label": is_generic_label  # Speichern, ob wir den Patch angewendet haben
                })
            except Exception as e:
                print(f"Fehler beim Lesen der Konfiguration von {path}: {e}")

    return available_models


def select_model(models):
    """
    Zeigt dem Benutzer die gefundenen Modelle an und lässt ihn eines auswählen.

    KORRIGIERT: Stürzt nicht mehr ab, wenn 'models' leer ist.
    """
    if not models:
        print("\n" + "=" * 50)
        print("❌ FEHLER: Keine gültigen, trainierten Modelle gefunden.")
        print("Stelle sicher, dass die Ordner 'ergebnisse_*' existieren und")
        print("dass 'config.json' (im Root) und 'trainer_state.json' (in einem checkpoint-Ordner) vorhanden sind.")
        print("\nWarte 10 Sekunden und versuche es erneut...")
        print("=" * 50 + "\n")
        time.sleep(10)
        return None  # Signalisiert dem main-loop, es erneut zu versuchen

    print("\n--- Verfügbare trainierte Modelle ---")
    for i, model_info in enumerate(models):
        label_count = len(model_info['labels'])
        metric_str = f"{model_info['metric_name']} = {model_info['metric_val']:.4f}"
        labels_str = ", ".join(model_info['labels'])

        print(f"\n  [{i + 1}] {model_info['path']}")

        # NEU: Warnung anzeigen, wenn Labels gepatcht wurden
        if model_info.get("is_generic_label"):
            print(
                f"      Labels ({label_count}): {labels_str}  <- [!] WARNUNG: config.json war fehlerhaft, Labels wurden ersetzt.")
        else:
            print(f"      Labels ({label_count}): {labels_str}")

        print(f"      Metrik: {metric_str}")

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
    1. Lädt Vokabulare (wird als geteilt angenommen).
    2. Sucht und lässt den Benutzer ein Modell auswählen.
    3. Lädt das Modell.
    4. Startet das Untermenü (Chat / Validierung).
    5. Kehrt zur Modellauswahl zurück, wenn das Untermenü verlassen wird.
    """
    # 1. Vokabulare einmalig laden (wird von allen Modellen geteilt)
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

        # 3. Ausgewähltes Modell laden
        try:
            model, tokenizer, device = load_model_and_tokenizer(MODEL_PATH)
        except Exception as e:
            print(f"Kritischer Fehler beim Laden des Modells {MODEL_PATH}: {e}")
            continue

            # 4. Starte das Untermenü mit den geladenen, modellspezifischen Daten
        main_menu(model, tokenizer, device, vocabs, PRIORITY_ORDER)


if __name__ == "__main__":
    main()