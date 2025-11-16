# chat_with_modell.py (Modular / Multi-Modell / Validierung V2.1 / Batch V1.3)

import os
import re
import sys
import torch
import pandas as pd
import numpy as np
import glob  # Zum Finden von Modell-Ordnern
import json  # Zum Lesen von config.json und trainer_state.json
import time  # Für Wartezeit und Matrix-Timestamp
from datetime import datetime  # Für Datums-Ordner

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Importe für die Validierungs-Funktion
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm  # Für den Fortschrittsbalken

# --- Globale Konfiguration für Vokabulare ---
BASE_DIR = "trainingsdaten"
NEG_CSV = os.path.join(BASE_DIR, "neg_arguments_multilingual.csv")
POS_CSV = os.path.join(BASE_DIR, "pos_arguments_multilingual.csv")
SLA_CSV = os.path.join(BASE_DIR, "sla_arguments_multilingual.csv")

DEFAULT_DATA_FILE = os.path.join(BASE_DIR, "5_prio_stufen/dataset-tickets-german_normalized_50_5_2.csv")
NEW_TOKENS = ['KEY_CORE_APP', 'KEY_CRITICAL', 'KEY_REQUEST', 'KEY_NORMAL']

SLA_WEIGHT = 5
NEG_WEIGHT = 4
POS_WEIGHT = 1

DEFAULT_PRIORITY_ORDER = [
    "critical",
    "high",
    "medium",
    "low",
    "very_low"
]


# ==============================================================================
# HILFSFUNKTIONEN (Vokabular & Preprocessing)
# ==============================================================================

def load_vocab_from_csvs() -> (list, list, list):
    """Lädt die Vokabularlisten aus den CSV-Dateien."""
    print("Lade Vokabular-Listen aus CSV-Dateien...")
    try:
        df_neg = pd.read_csv(NEG_CSV)
        df_pos = pd.read_csv(POS_CSV)
        df_sla = pd.read_csv(SLA_CSV)

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
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except OSError:
        print(f"❌ FEHLER: Modell-Ordner '{model_path}' nicht gefunden oder korrupt.")
        print("Stelle sicher, dass 'config.json' und 'pytorch_model.bin' vorhanden sind.")
        return None, None, None
    except Exception as e:
        print(f"❌ FEHLER beim Laden des Modells: {e}")
        return None, None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"✅ Modell erfolgreich geladen und auf '{device}' verschoben.")
    return model, tokenizer, device


def predict_priority(subject, body, model, tokenizer, device, vocabs, priority_list, max_len):
    """
    Führt eine einzelne Vorhersage für ein neues Ticket durch.
    Gibt das *native* Modell-Label zurück (z.B. 'critical').
    """
    neg_vocab, pos_vocab, sla_vocab = vocabs
    raw_text = str(body) + " " + str(subject)

    enriched_text = preprocess_with_vocab(
        raw_text,
        neg_vocab, pos_vocab, sla_vocab,
        sla_weight=SLA_WEIGHT,
        neg_weight=NEG_WEIGHT,
        pos_weight=POS_WEIGHT
    )

    inputs = tokenizer(
        enriched_text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)[0]
    predicted_index = torch.argmax(probabilities).item()

    # Das ist das native Label des Modells (z.B. "high")
    predicted_label = priority_list[predicted_index]
    confidence = probabilities[predicted_index].item()

    return predicted_label, confidence


# ==============================================================================
# VALIDIERUNGS- & BATCH-FUNKTIONEN (MODIFIZIERT)
# ==============================================================================

def map_csv_labels(df, original_label_col, target_labels):
    """
    Frägt den Benutzer, ob CSV-Labels (z.B. '1') auf Ziel-Labels (z.B. 'P1' oder 'high')
    gemappt werden sollen.

    Gibt den DataFrame (potenziell gefiltert) und den Namen der neuen,
    gemappten Label-Spalte zurück.
    """
    print("\n" + "-" * 70)
    print("--- Label-Abgleich (Mapping) für Genauigkeitsberechnung ---")

    df[original_label_col] = df[original_label_col].astype(str)
    unique_csv_labels = sorted(df[original_label_col].unique())

    print(f"Ihre CSV enthält die folgenden Labels in '{original_label_col}':")
    print(f"  -> {unique_csv_labels}")
    print(f"Das Zielsystem (Modell-Output oder Ihr Mapping) erwartet diese Labels:")
    print(f"  -> {target_labels}")

    mapping_dict = {}
    mapped_col_name = "mapped_label"

    while True:
        choice = input(
            "\nSollen diese CSV-Labels für den Vergleich auf die Ziel-Labels gemappt werden? (j/n): ").strip().lower()

        if choice == 'n':
            print("  -> OK. Labels werden als reine Strings verglichen (z.B. '1' vs 'P1').")
            df[mapped_col_name] = df[original_label_col].astype(str)
            return df, mapped_col_name

        elif choice == 'j':
            print("  -> OK. Bitte weisen Sie die CSV-Labels den Ziel-Labels zu.")
            print(f"   Ziel-Optionen: {target_labels} (oder 'skip', um ein Label zu ignorieren)")

            for csv_label in unique_csv_labels:
                while True:
                    prompt = f"    CSV-Label '{csv_label}' -> Ziel-Label: "
                    user_map = input(prompt).strip()  # Nicht lower(), falls Ziel 'P1' heißt

                    if user_map in target_labels:
                        mapping_dict[csv_label] = user_map
                        print(f"      '{csv_label}' wird als '{user_map}' gezählt.")
                        break
                    elif user_map.lower() == 'skip' or user_map == '':
                        mapping_dict[csv_label] = pd.NA
                        print(f"      '{csv_label}' wird ignoriert (übersprungen).")
                        break
                    else:
                        print(f"    Ungültige Eingabe. Muss eines der folgenden sein: {target_labels} oder 'skip'")

            print("Wende Mapping an...")
            df[mapped_col_name] = df[original_label_col].map(mapping_dict)

            original_count = len(df)
            df.dropna(subset=[mapped_col_name], inplace=True)
            mapped_count = len(df)

            if original_count > mapped_count:
                print(
                    f"  INFO: {original_count - mapped_count} Zeilen wurden ignoriert (da sie 'skip' oder kein Mapping hatten).")

            if mapped_count == 0:
                print("  WARNUNG: Nach dem Mapping sind 0 Zeilen übrig.")
            else:
                print(f"  -> {len(df)} Zeilen werden für die Validierung verwendet.")
            return df, mapped_col_name

        else:
            print("  Ungültige Eingabe. Bitte 'j' oder 'n'.")


def get_output_mapping(model_labels):
    """
    Frägt den Benutzer, ob die 5 Modell-Labels auf ein Zielsystem (z.B. 3 Stufen)
    gemappt werden sollen.
    Gibt ein Mapping-Dict (z.B. {'critical': 'P1', ...}) und die Liste der
    neuen Ziel-Labels (z.B. ['P1', 'P2', 'P3']) zurück.
    """
    print("\n" + "-" * 70)
    print("--- Optionales Output-Mapping (Skalierung) ---")
    print("Das Modell gibt 5 Stufen aus:", model_labels)

    mapping_dict = {}

    while True:
        choice = input(
            "Möchten Sie diese 5 Stufen auf ein anderes System (z.B. 3 Stufen) mappen? (j/n): ").strip().lower()

        if choice == 'n':
            print("  -> OK. Die 5 nativen Modell-Labels werden verwendet.")
            # Rückgabe: Leeres Dict und die originalen Labels
            return None, model_labels

        elif choice == 'j':
            print("  -> OK. Bitte definieren Sie Ihr Zielsystem.")
            print("     Geben Sie für jedes Modell-Label das entsprechende Ziel-Label ein (z.B. 'P1', 'Hoch', '1')")

            final_target_labels = set()
            for model_label in model_labels:
                while True:
                    prompt = f"    Modell-Label '{model_label}' -> Ziel-Label: "
                    user_map = input(prompt).strip()

                    if user_map:
                        mapping_dict[model_label] = user_map
                        final_target_labels.add(user_map)
                        print(f"      '{model_label}' wird zu '{user_map}'")
                        break
                    else:
                        print("    Eingabe darf nicht leer sein.")

            print(f"\n✅ Output-Mapping erstellt. Neue Ziel-Labels sind: {sorted(list(final_target_labels))}")
            return mapping_dict, sorted(list(final_target_labels))

        else:
            print("  Ungültige Eingabe. Bitte 'j' oder 'n'.")


# --- MODIFIZIERT: Funktion für detaillierte Validierung ---
def run_detailed_validation(model, tokenizer, device, vocabs, model_info):
    """
    Führt eine detaillierte, interaktive Validierung durch (Konfusionsmatrix).
    (MODIFIZIERT: Nutzt jetzt Output-Mapping und dynamischen Speicherort)
    """
    print("\n" + "=" * 70)
    print("--- Detaillierte Validierung starten (Konfusionsmatrix) ---")

    model_path = model_info['path']
    model_labels = model_info['labels']  # (z.B. ['critical', ...])
    max_len = model_info['max_len']

    # --- 1. Validierungs-CSV auswählen ---
    default_file = model_info.get('training_csv', DEFAULT_DATA_FILE)
    filepath = input(f"Pfad zur Validierungs-CSV [Standard: {default_file}]: ").strip() or default_file

    try:
        df = pd.read_csv(filepath)
        print(f"Lade {len(df)} Zeilen aus {filepath}...")
    except FileNotFoundError:
        print(f"❌ FEHLER: Datei nicht gefunden: {filepath}")
        return
    except Exception as e:
        print(f"❌ FEHLER beim Lesen der CSV: {e}")
        return

    # --- 2. Spalten-Zuweisung ---
    print("\nWelche Spalten sollen für die Validierung verwendet werden?")
    default_subj = model_info.get('training_subject_col', 'subject')
    default_body = model_info.get('training_body_col', 'body')
    default_label = 'priority'

    subj_col = input(f"  Spaltenname für BETREFF [{default_subj}]: ").strip() or default_subj
    body_col = input(f"  Spaltenname für TEXT [{default_body}]: ").strip() or default_body
    label_col = input(f"  Spaltenname für LABEL [{default_label}]: ").strip() or default_label

    required_cols = [subj_col, body_col, label_col]
    if not all(col in df.columns for col in required_cols):
        print(f"❌ FEHLER: CSV muss die Spalten '{', '.join(required_cols)}' enthalten.")
        return

    # --- 3. NEU: Output-Mapping (Skalierung) (Goal 1) ---
    output_map, final_target_labels = get_output_mapping(model_labels)
    # final_target_labels ist z.B. ['P1', 'P2', 'P3'] oder ['critical', ...]

    # --- 4. Input-Label-Mapping (CSV -> Ziel-Labels) ---
    # map_csv_labels mappt jetzt die CSV-Labels (z.B. '1') auf die
    # final_target_labels (z.B. 'P1')
    df_mapped, mapped_label_col = map_csv_labels(df, label_col, final_target_labels)

    if df_mapped.empty:
        print("❌ FEHLER: Nach dem Label-Mapping sind keine Daten mehr übrig. Breche ab.")
        return

    # --- 5. Sampling-Modus ---
    print("\nWelche Tickets sollen validiert werden?")
    print("  [1] Alle Tickets (die gemappt werden konnten)")
    print("  [2] Manuelle Anzahl pro (gemappter) Prioritäts-Klasse")

    df_sample = None
    sample_mode = ""

    while True:
        mode_choice = input("Wähle Modus [1]: ").strip() or "1"
        if mode_choice == '1':
            df_sample = df_mapped.copy()
            sample_mode = "Alle"
            print(f"✅ Alle {len(df_sample)} gemappten Tickets werden validiert.")
            break
        elif mode_choice == '2':
            print(f"  Definiere die Anzahl der Tickets pro Klasse (Ziel-Labels: {final_target_labels}):")
            counts_dict = {}
            for label in final_target_labels:
                while True:
                    count_str = input(f"    Anzahl für '{label}': ").strip() or "0"
                    try:
                        counts_dict[label] = int(count_str)
                        break
                    except ValueError:
                        print("    Ungültige Zahl.")

            print("Sammle Samples...")
            sample_list = []
            df_grouped = df_mapped.groupby(mapped_label_col)
            for label, count in counts_dict.items():
                if count == 0:
                    continue
                try:
                    class_df = df_grouped.get_group(label)
                    n_available = len(class_df)
                    n_to_sample = min(n_available, count)
                    if n_to_sample < count:
                        print(
                            f"    WARNUNG: Für '{label}' nur {n_available} statt {count} verfügbar. Nehme {n_to_sample}.")
                    sample_list.append(class_df.sample(n=n_to_sample, random_state=42))
                except KeyError:
                    print(f"    INFO: Gemapptes Label '{label}' nicht in CSV gefunden. Überspringe.")

            if not sample_list:
                print("❌ FEHLER: Keine Tickets zum Validieren ausgewählt.")
                return

            df_sample = pd.concat(sample_list)
            sample_mode = f"Manuell ({len(df_sample)})"
            print(f"✅ {len(df_sample)} Tickets für die Validierung gesammelt.")
            break
        else:
            print("  Ungültige Eingabe. Bitte '1' oder '2'.")

    # --- 6. Vorhersage-Schleife (mit tqdm) ---
    print(f"\nStarte Vorhersage für {len(df_sample)} Tickets...")

    originals = df_sample[mapped_label_col].tolist()
    predictions = []

    disable_tqdm = len(df_sample) < 10

    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Validiere", disable=disable_tqdm):
        # 1. Hole native Modell-Vorhersage (z.B. 'critical')
        predicted_label, _ = predict_priority(
            row[subj_col], row[body_col],
            model, tokenizer, device, vocabs, model_labels, max_len  # Wichtig: model_labels (5)
        )

        # 2. Wende Output-Mapping an (z.B. 'critical' -> 'P1')
        if output_map:
            predicted_label = output_map.get(predicted_label, predicted_label)

        predictions.append(predicted_label)

    print("✅ Validierung abgeschlossen.")
    print("-" * 70)

    # --- 7. Ergebnisse auswerten (Text) ---
    accuracy = accuracy_score(originals, predictions)
    print(f"Gesamt-Genauigkeit (Accuracy) [{sample_mode} Samples]: {accuracy:.2%}")
    print("\nDetail-Auswertung (Classification Report):")

    all_present_labels = sorted(list(set(originals) | set(predictions)))

    report = classification_report(
        originals,
        predictions,
        labels=all_present_labels,
        zero_division=0
    )
    print(report)
    print("-" * 70)

    # --- 8. Ergebnisse auswerten (Grafik) ---
    try:
        # KORREKTUR (Goal 1): Verwende die final_target_labels für die Achsen
        cm = confusion_matrix(originals, predictions, labels=final_target_labels)

        plt.figure(figsize=(max(8, len(final_target_labels) * 2), max(6, len(final_target_labels) * 1.5)))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=final_target_labels,
            yticklabels=final_target_labels
        )
        plt.title(
            f'Konfusionsmatrix - {os.path.basename(model_path)}\n[{sample_mode} Samples / Genauigkeit: {accuracy:.2%}]')
        plt.ylabel('Wahres Label (aus CSV, gemappt)')
        plt.xlabel('Vorhergesagtes Label (Modell, gemappt)')

        # KORREKTUR (Goal 2): Dynamischer Speicherort
        timestamp = int(time.time())
        matrix_name = f"validierungs_matrix_{timestamp}.png"
        output_filename = ""

        # Prüfe, ob die Validierungs-CSV die Trainings-CSV ist
        is_training_csv = False
        training_csv_path = model_info.get('training_csv')
        if training_csv_path and training_csv_path != "N/A":
            try:
                # Vergleiche absolute Pfade
                is_training_csv = os.path.abspath(filepath) == os.path.abspath(training_csv_path)
            except Exception:
                pass  # Fehler beim Pfadvergleich

        if is_training_csv:
            # Speichere im Hauptordner
            output_filename = os.path.join(model_path, matrix_name)
            print(f"  -> Validierung mit Trainings-CSV erkannt. Speichere Matrix in: {output_filename}")
        else:
            # Speichere im 'processed_csv' Ordner
            timestamp_date = datetime.now().strftime("%Y-%m-%d")
            output_subdir = os.path.join(model_path, f"processed_csv_{timestamp_date}")
            os.makedirs(output_subdir, exist_ok=True)
            output_filename = os.path.join(output_subdir, matrix_name)
            print(f"  -> Validierung mit externer CSV. Speichere Matrix in: {output_filename}")

        plt.savefig(output_filename, bbox_inches='tight')
        print(f"✅ Grafik wurde gespeichert.")

    except Exception as e:
        print(f"❌ FEHLER beim Erstellen der Grafik: {e}")
        print("  (Stelle sicher, dass 'matplotlib' und 'seaborn' installiert sind: pip install matplotlib seaborn)")

    print("-" * 70)


# --- MODIFIZIERT: Funktion für die Batch-Klassifizierung ---
def run_batch_classification(model, tokenizer, device, vocabs, model_info):
    """
    Führt eine vollständige Klassifizierung einer CSV-Datei durch
    und speichert die Ergebnisse.
    """
    print("\n" + "=" * 70)
    print("--- Batch-Klassifizierung starten ---")

    model_path = model_info['path']
    model_labels = model_info['labels']  # (z.B. ['critical', ...])
    max_len = model_info['max_len']

    # --- 1. Quell-CSV auswählen ---
    default_file = model_info.get('training_csv', DEFAULT_DATA_FILE)
    filepath = input(
        f"Pfad zur Quell-CSV (die klassifiziert werden soll) [Standard: {default_file}]: ").strip() or default_file

    try:
        df = pd.read_csv(filepath)
        print(f"Lade {len(df)} Zeilen aus {filepath}...")
    except FileNotFoundError:
        print(f"❌ FEHLER: Datei nicht gefunden: {filepath}")
        return
    except Exception as e:
        print(f"❌ FEHLER beim Lesen der CSV: {e}")
        return

    # --- 2. Spalten-Zuweisung ---
    print("\nWelche Spalten sollen für die Klassifizierung verwendet werden?")
    default_subj = model_info.get('training_subject_col', 'subject')
    default_body = model_info.get('training_body_col', 'body')

    subj_col = input(f"  Spaltenname für BETREFF [{default_subj}]: ").strip() or default_subj
    body_col = input(f"  Spaltenname für TEXT [{default_body}]: ").strip() or default_body

    required_cols = [subj_col, body_col]
    if not all(col in df.columns for col in required_cols):
        print(f"❌ FEHLER: CSV muss die Spalten '{', '.join(required_cols)}' enthalten.")
        return

    # --- 3. Original-Priorität prüfen ---
    has_orig_priority = False
    original_label_col = None
    if 'priority' in df.columns:
        print("  -> Spalte 'priority' gefunden. Wird zu 'orig_priority' umbenannt.")
        df.rename(columns={'priority': 'orig_priority'}, inplace=True)
        original_label_col = 'orig_priority'
        has_orig_priority = True
    elif 'orig_priority' in df.columns:
        print("  -> Spalte 'orig_priority' bereits vorhanden.")
        original_label_col = 'orig_priority'
        has_orig_priority = True

    # --- 4. Vorhersage-Schleife ---
    print(f"\nStarte Klassifizierung für {len(df)} Tickets...")

    predictions = []
    confidences = []

    start_time = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Klassifiziere"):
        predicted_label, confidence = predict_priority(
            row[subj_col], row[body_col],
            model, tokenizer, device, vocabs, model_labels, max_len
        )
        predictions.append(predicted_label)
        confidences.append(confidence)

    end_time = time.time()
    duration_sec = end_time - start_time

    df['predicted_priority_native'] = predictions  # Speichere die native 5-Stufen-Antwort
    df['confidence'] = confidences

    print("✅ Klassifizierung abgeschlossen.")

    # --- 5. Optionales Output-Mapping (Skalierung) ---
    output_map, final_target_labels = get_output_mapping(model_labels)

    if output_map:
        print("Wende Output-Mapping an...")
        # Erstelle eine neue Spalte für die gemappte Prio
        df['predicted_priority'] = df['predicted_priority_native'].map(output_map)
        print("✅ Output-Mapping abgeschlossen.")
    else:
        # Wenn kein Mapping, ist die native Prio die finale Prio
        df['predicted_priority'] = df['predicted_priority_native']

    # --- 6. Genauigkeit berechnen (falls möglich) ---
    print("-" * 70)
    accuracy_str = "N/A (Keine 'orig_priority' Spalte gefunden)"
    if has_orig_priority:
        # Vergleiche die 'orig_priority' Spalte mit den *finalen* Ziel-Labels
        df_for_metrics, mapped_label_col = map_csv_labels(
            df.copy(),
            original_label_col,
            final_target_labels
        )

        if df_for_metrics.empty:
            accuracy_str = "N/A (Keine Labels konnten gemappt werden)"
        else:
            try:
                # Vergleiche die gemappten Original-Labels mit den (potenziell) gemappten Vorhersagen
                accuracy = accuracy_score(df_for_metrics[mapped_label_col], df_for_metrics['predicted_priority'])
                accuracy_str = f"{accuracy:.4f} ({accuracy:.2%}) (basiert auf {len(df_for_metrics)} gemappten Zeilen)"
            except Exception as e:
                accuracy_str = f"Fehler bei Berechnung: {e}"

    # --- 7. Ergebnisse speichern ---
    try:
        timestamp_date = datetime.now().strftime("%Y-%m-%d")
        timestamp_full = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_subdir = os.path.join(model_path, f"processed_csv_{timestamp_date}")
        os.makedirs(output_subdir, exist_ok=True)

        source_csv_name = os.path.basename(filepath)
        output_csv_name = f"{timestamp_full}_{source_csv_name.replace('.csv', '')}_classified.csv"
        output_summary_name = f"{timestamp_full}_{source_csv_name.replace('.csv', '')}_summary.txt"

        output_csv_path = os.path.join(output_subdir, output_csv_name)
        output_summary_path = os.path.join(output_subdir, output_summary_name)

        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

        with open(output_summary_path, "w", encoding="utf-8") as f:
            f.write("=== Batch-Klassifizierungs-Zusammenfassung ===\n")
            f.write(f"Modell: {model_path}\n")
            f.write(f"Quelle-CSV: {filepath}\n")
            f.write(f"Startzeit: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Endzeit: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dauer: {duration_sec:.2f} Sekunden\n")
            f.write(f"Anzahl Zeilen: {len(df)}\n")
            f.write(f"Genauigkeit (vs 'orig_priority', nach Mapping): {accuracy_str}\n")

        print("✅ Ergebnisse erfolgreich gespeichert.")
        print(f"  -> CSV-Datei: {output_csv_path}")
        print(f"  -> Summary:   {output_summary_path}")
        print(f"  -> Genauigkeit: {accuracy_str}")

    except Exception as e:
        print(f"❌ FEHLER beim Speichern der Ergebnisse: {e}")

    print("-" * 70)


# ==============================================================================
# HAUPT-MENÜS
# ==============================================================================

def interactive_chat_mode(model, tokenizer, device, vocabs, priority_list, max_len):
    """
    Startet die interaktive Chat-Schleife.
    """
    print("\n--- Interaktiver KI-Priorisierungs-Chat ---")
    print(f"Modell: {os.path.basename(model.config.name_or_path)} ({len(priority_list)} Labels, max_len={max_len})")
    print("Gib 'exit' oder 'quit' ein, um zum Hauptmenü zurückzukehren.")
    print("HINWEIS: Dieser Chat gibt die *nativen* 5 Modell-Labels aus (z.B. 'critical').")
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


# --- MODIFIZIERT: Das Menü, das erscheint, NACHDEM ein Modell ausgewählt wurde ---
def show_model_action_menu(model_info, vocabs):
    """
    Zeigt das Menü für ein ausgewähltes Modell (Chat, Validierung, Batch, Zurück).
    """
    model_path = model_info['path']
    priority_list = model_info['labels']
    max_len = model_info['max_len']

    while True:
        print("\n" + "=" * 70)
        print(f"--- Modell ausgewählt ---")
        print(f"Ordner: {model_path}")
        print(f"Labels: {', '.join(priority_list)}")
        print(f"Max. Tokenlänge: {max_len}")
        print(f"Trainings-CSV: {model_info.get('training_csv', 'N/A')}")
        print("-" * 70)
        print("  [1] Interaktiven Chat starten (nutzt native 5 Labels)")
        print("  [2] Detaillierte Validierung starten (Konfusionsmatrix & Skalierung)")
        print("  [3] Batch-Klassifizierung einer CSV-Datei (mit Skalierung)")
        print("  [b] Zurück zur Modellauswahl")

        choice = input("Wähle eine Option: ").strip().lower()

        model, tokenizer, device = None, None, None

        try:
            if choice == '1':
                model, tokenizer, device = load_model_and_tokenizer(model_path)
                if model:
                    interactive_chat_mode(model, tokenizer, device, vocabs, priority_list, max_len)

            elif choice == '2':
                model, tokenizer, device = load_model_and_tokenizer(model_path)
                if model:
                    run_detailed_validation(model, tokenizer, device, vocabs, model_info)

            elif choice == '3':
                model, tokenizer, device = load_model_and_tokenizer(model_path)
                if model:
                    run_batch_classification(model, tokenizer, device, vocabs, model_info)

            elif choice in ['b', 'q', 'back', 'quit', 'exit']:
                print("Zurück zur Modellauswahl.")
                return

            else:
                print("Ungültige Eingabe, bitte '1', '2', '3' oder 'b' wählen.")

        finally:
            if model:
                del model
                del tokenizer
                if device == "cuda":
                    torch.cuda.empty_cache()
                    print(f"(Modell '{os.path.basename(model_path)}' aus VRAM entfernt.)")


# ==============================================================================
# MODELLAUSWAHL (MODIFIZIERT: Liest Trainings-Infos)
# ==============================================================================

def parse_summary_txt(filepath):
    """Liest die training_setup_summary.txt und extrahiert die wichtigsten Infos."""
    info = {
        "training_csv": "N/A",
        "training_subject_col": "N/A",
        "training_body_col": "N/A"
    }
    if not os.path.exists(filepath):
        return info

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith("CSV-Datei:"):
                    info["training_csv"] = line.split(":", 1)[1].strip()
                elif line.strip().startswith("Spalte (Betreff):"):
                    info["training_subject_col"] = line.split(":", 1)[1].strip()
                elif line.strip().startswith("Spalte (Text):"):
                    info["training_body_col"] = line.split(":", 1)[1].strip()
        return info
    except Exception:
        return info


def find_available_models():
    """
    Sucht nach allen 'ergebnisse_*' Ordnern und extrahiert Metriken
    und Label-Konfigurationen.
    """
    print("Suche nach trainierten Modellen (Ordner './ergebnisse_*')...")
    model_dirs = glob.glob("./ergebnisse_*")
    available_models = []

    for path in model_dirs:
        if not os.path.isdir(path):
            continue

        config_path = os.path.join(path, "config.json")
        summary_path = os.path.join(path, "training_setup_summary.txt")

        if not os.path.exists(config_path):
            # print(f"ℹ️  '{path}' enthält keine 'config.json'. Überspringe.") # Weniger störend
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

            max_len = config.get("model_max_length", 512)
            if "max_position_embeddings" in config:
                max_len = config.get("max_position_embeddings", 512)

            # PATCH: Generische 'LABEL_0' Fehler abfangen
            is_generic_label = False
            if not priority_list or priority_list[0].startswith("LABEL_"):
                is_generic_label = True
                if len(priority_list) == len(DEFAULT_PRIORITY_ORDER):
                    priority_list = DEFAULT_PRIORITY_ORDER
                else:
                    print(
                        f"⚠️ Warnung: {path} hat generische Labels UND eine unerwartete Label-Anzahl ({len(priority_list)}).")
                    is_generic_label = False

                    # 2. Metriken (Optional) aus dem letzten Checkpoint
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
                    metric_name = "Fehler (Metrik)"

            # 3. Trainings-Setup (Optional) aus der summary.txt lesen (NEU)
            summary_info = parse_summary_txt(summary_path)

            model_info_dict = {
                "path": path,
                "metric_val": best_metric_val,
                "metric_name": metric_name,
                "labels": priority_list,
                "is_generic_label": is_generic_label,
                "max_len": max_len
            }
            model_info_dict.update(summary_info)
            available_models.append(model_info_dict)

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
        return None

    print("\n--- Verfügbare trainierte Modelle ---")

    try:
        models.sort(key=lambda x: x['path'], reverse=True)
    except Exception:
        pass

    for i, model_info in enumerate(models):
        label_count = len(model_info['labels'])

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

        print(f"      Trainings-CSV: {model_info.get('training_csv', 'N/A')}")
        print(
            f"      Trainings-Spalten: [Body] {model_info.get('training_body_col', 'N/A')}, [Subject] {model_info.get('training_subject_col', 'N/A')}")

    while True:
        try:
            choice_str = input(f"\nWähle ein Modell (1-{len(models)}) [oder 'q' zum Beenden]: ")
            if choice_str.lower() in ['q', 'exit', 'quit']:
                sys.exit("Programm beendet.")

            choice_idx = int(choice_str) - 1
            if 0 <= choice_idx < len(models):
                return models[choice_idx]
            else:
                print(f"Ungültige Zahl. Bitte 1-{len(models)} wählen.")
        except ValueError:
            print("Ungültige Eingabe. Bitte eine Zahl wählen.")


# ==============================================================================
# HAUPT-FUNKTION (Startpunkt)
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
            continue

            # 3. Starte das Untermenü für das ausgewählte Modell
        show_model_action_menu(selected_model_info, vocabs)


if __name__ == "__main__":
    main()