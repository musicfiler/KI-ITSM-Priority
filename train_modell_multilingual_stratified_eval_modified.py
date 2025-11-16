# train_modell_multilingual-stratified_eval_modified.py

# Erforderliche Bibliotheken importieren
import os
import sys
import io  # Erforderlich für das Logging
import time
import re
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import shutil
import math  # Für die Rundung der max_length
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    PreTrainedTokenizer
)
from datasets import load_dataset, ClassLabel, Features, Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- Globale Konfiguration ---
BASE_DIR = "trainingsdaten"
DEFAULT_DATA_FILE = os.path.join(BASE_DIR, "5_prio_stufen/dataset-tickets-german_normalized_50_5_2.csv")

# ==============================================================================
# FELDZUORDNUNGEN (STANDARDWERTE)
# ==============================================================================
DEFAULT_FIELD_LABEL = "priority"
DEFAULT_FIELD_SUBJECT = "subject"
DEFAULT_FIELD_BODY = "body"
FIELDS_TO_REMOVE = ['queue', 'language']
# ==============================================================================


# Vokabular-Dateien (Pfade bleiben global)
NEG_CSV = os.path.join(BASE_DIR, "neg_arguments_multilingual.csv")
POS_CSV = os.path.join(BASE_DIR, "pos_arguments_multilingual.csv")
SLA_CSV = os.path.join(BASE_DIR, "sla_arguments_multilingual.csv")
STOPWORDS_CSV = os.path.join(BASE_DIR, "stopwords.csv")

# Diese Reihenfolge ist jetzt KRITISCH für die Per-Klassen-Metriken
PRIORITY_ORDER = ["critical", "high", "medium", "low", "very_low"]
HIGH_PRIO_CLASSES_STR = ["critical", "high"]
TOP_N_TERMS = 75
MIN_DF = 5
NEW_TOKENS = ['KEY_CORE_APP', 'KEY_CRITICAL', 'KEY_REQUEST', 'KEY_NORMAL']

# ==============================================================================
# KEYWORD-GEWICHTUNG (Anpassbar)
# ==============================================================================
KEYWORD_WEIGHTS = {
    "sla_weight": 5,  # (KEY_CORE_APP)
    "neg_weight": 4,  # (KEY_CRITICAL)
    "pos_weight": 1  # (KEY_REQUEST)
}
# ==============================================================================


# --- Stopwörter-Management (unverändert) ---
try:
    from stop_words import get_stop_words

    GERMAN_STOPWORDS = get_stop_words('de')
except ImportError:
    print("WARNUNG: Paket 'stop-words' nicht gefunden. (pip install stop-words)")
    print("Fahre ohne deutsche Stopwörter fort.")
    GERMAN_STOPWORDS = []

if os.path.exists(STOPWORDS_CSV):
    try:
        print(f"Lade benutzerdefinierte Stopwörter aus {STOPWORDS_CSV}...")
        df_stop = pd.read_csv(STOPWORDS_CSV)
        custom_stopwords_from_csv = df_stop['term'].dropna().astype(str).tolist()
        GERMAN_STOPWORDS.extend(custom_stopwords_from_csv)
        print(f"✅ Stopwörter-Liste um {len(custom_stopwords_from_csv)} Wörter aus {STOPWORDS_CSV} erweitert.")
    except Exception as e:
        print(f"⚠️ WARNUNG: Konnte {STOPWORDS_CSV} nicht laden oder verarbeiten: {e}")
else:
    print(f"ℹ️  Keine {STOPWORDS_CSV} gefunden.")
    print("   (Wird bei Bedarf in Phase 1 (Vokabular-Generierung) erstellt.)")


# -------------------------------------------


# ==============================================================================
# COMPUTE_METRICS FUNKTION (unverändert)
# ==============================================================================

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    class_indices = list(range(len(PRIORITY_ORDER)))
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0,
        labels=class_indices
    )
    metrics = {
        'accuracy': acc,
        'f1_weighted': weighted_f1,
        'precision_weighted': weighted_precision,
        'recall_weighted': weighted_recall
    }
    print("\n--- Per-Klassen-Evaluierung (Recall = 'Trefferquote' der Klasse) ---")
    for i, label_name in enumerate(PRIORITY_ORDER):
        recall_val = recall_per_class[i]
        precision_val = precision_per_class[i]
        f1_val = f1_per_class[i]
        support_val = support_per_class[i]
        metrics[f'recall_{label_name}'] = recall_val
        metrics[f'precision_{label_name}'] = precision_val
        metrics[f'f1_{label_name}'] = f1_val
        print(f"  [{label_name.upper():<8}]: "
              f"Recall: {recall_val:<7.2%}, "
              f"Precision: {precision_val:<7.2%}, "
              f"F1: {f1_val:<7.2%}, "
              f"Support: {int(support_val)} Tickets")
    print("---------------------------------------------------------------------")
    f1_critical = f1_per_class[0]
    f1_high = f1_per_class[1]
    f1_crit_high_avg = (f1_critical + f1_high) / 2.0
    metrics['f1_critical_high_avg'] = f1_crit_high_avg
    print(f"  [SPEZIAL-METRIK]: f1_critical_high_avg = {f1_crit_high_avg:<7.2%}")
    print("=====================================================================")
    return metrics


# ==============================================================================
# HILFSFUNKTIONEN (unverändert)
# ==============================================================================

def generate_vocab_files_if_needed(data_file_path: str, field_subject: str, field_body: str, field_label: str):
    os.makedirs(BASE_DIR, exist_ok=True)
    if os.path.exists(NEG_CSV) and os.path.exists(POS_CSV) and os.path.exists(SLA_CSV):
        print("✅ Phase 1: Vokabular-Dateien (pos/neg/sla) gefunden. Überspringe automatische Generierung.")
        return

    print("⚠️ Phase 1: Vokabular-Dateien nicht gefunden. Starte automatische Extraktion...")
    if not os.path.exists(data_file_path):
        print(f"❌ FEHLER: Trainingsdatensatz {data_file_path} nicht gefunden.")
        sys.exit()
    try:
        df = pd.read_csv(data_file_path)
        df['text'] = df[field_subject].fillna('') + ' ' + df[field_body].fillna('')
        df = df.dropna(subset=['text', field_label])
    except KeyError as e:
        print(f"❌ FEHLER: Die Spalte {e} wurde in {data_file_path} nicht gefunden.")
        sys.exit()

    print(f"Analysiere {len(df)} Tickets für Vokabular-Extraktion...")
    df_high_prio = df[df[field_label].isin(HIGH_PRIO_CLASSES_STR)]
    df_low_prio = df[~df[field_label].isin(HIGH_PRIO_CLASSES_STR)]

    if df_high_prio.empty or df_low_prio.empty:
        print(f"❌ FEHLER: Konnte keine Tickets für hohe Priorität (Werte: {HIGH_PRIO_CLASSES_STR}) finden.")
        sys.exit()

    vectorizer = TfidfVectorizer(
        stop_words=GERMAN_STOPWORDS,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=MIN_DF
    )
    print("Passe TfidfVectorizer an alle Texte an...")
    vectorizer.fit(df['text'])

    if not os.path.exists(STOPWORDS_CSV):
        print(f"Generiere Stopword-Vorschlagsliste unter {STOPWORDS_CSV}...")
        try:
            feature_names_idf = vectorizer.get_feature_names_out()
            idf_scores = vectorizer.idf_
            idf_df = pd.DataFrame({'term': feature_names_idf, 'idf_score': idf_scores})
            idf_df = idf_df.sort_values(by='idf_score', ascending=True)
            idf_df = idf_df[~idf_df['term'].str.match(r'^\d+$')]
            idf_df = idf_df[idf_df['term'].str.len() > 2]
            top_stopwords = idf_df.head(200)['term'].tolist()
            pd.DataFrame(top_stopwords, columns=['term']).to_csv(STOPWORDS_CSV, index=False)
            print(f"✅ Stopword-Vorschlagsliste mit {len(top_stopwords)} Wörtern gespeichert.")
            print("   Bitte überprüfen Sie diese Liste manuell und starten Sie das Skript erneut.")
        except Exception as e:
            print(f"❌ FEHLER beim Generieren der Stopword-Liste: {e}")

    print("Transformiere Hoch- und Niedrig-Prio-Texte für TF-IDF-Vergleich...")
    tfidf_high = vectorizer.transform(df_high_prio['text'])
    tfidf_low = vectorizer.transform(df_low_prio['text'])
    mean_tfidf_high = np.asarray(tfidf_high.mean(axis=0)).ravel()
    mean_tfidf_low = np.asarray(tfidf_low.mean(axis=0)).ravel()
    score_diff = mean_tfidf_high - mean_tfidf_low
    feature_names = np.array(vectorizer.get_feature_names_out())
    results_df = pd.DataFrame({'term': feature_names, 'score_diff': score_diff})
    top_neg_terms = results_df.sort_values(by='score_diff', ascending=False).head(TOP_N_TERMS)['term'].tolist()
    top_pos_terms = results_df.sort_values(by='score_diff', ascending=True).head(TOP_N_TERMS)['term'].tolist()
    pd.DataFrame(top_neg_terms, columns=['term']).to_csv(NEG_CSV, index=False)
    pd.DataFrame(top_pos_terms, columns=['term']).to_csv(POS_CSV, index=False)
    if not os.path.exists(SLA_CSV):
        pd.DataFrame(columns=['term']).to_csv(SLA_CSV, index=False)
    print(f"✅ Phase 1: Extraktion abgeschlossen. Dateien gespeichert in '{BASE_DIR}'.")
    print("👉 WICHTIG: Bitte bearbeite nun die CSV-Dateien und starte das Skript danach erneut.")
    print("-" * 70)


def load_vocab_from_csvs() -> (list, list, list):
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
        sys.exit()
    except KeyError:
        print("❌ FEHLER: CSV-Dateien müssen eine Spalte namens 'term' haben.")
        sys.exit()


def preprocess_with_vocab(
        text: str,
        neg_vocab: list,
        pos_vocab: list,
        sla_vocab: list,
        sla_weight: int,
        neg_weight: int,
        pos_weight: int
) -> str:
    """Reichert einen Text mit speziellen Signal-Wörtern (KEY_...) an."""
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


def add_new_tokens_to_tokenizer(tokenizer):
    tokenizer.add_special_tokens({'additional_special_tokens': NEW_TOKENS})
    print(f"Neue Tokens zum Tokenizer hinzugefügt: {NEW_TOKENS}")
    return tokenizer


# ==============================================================================
# Logger-Klasse (unverändert)
# ==============================================================================
class ConsoleLogger(object):
    def __init__(self, log_path: str):
        self.terminal_stdout = sys.stdout
        self.terminal_stderr = sys.stderr
        self.log_file = io.open(log_path, 'w', encoding='utf-8')

    def hook_stdout(self): sys.stdout = self

    def hook_stderr(self): sys.stderr = self

    def write(self, message):
        self.terminal_stdout.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal_stdout.flush()
        self.log_file.flush()

    def close_logs(self):
        sys.stdout = self.terminal_stdout
        sys.stderr = self.terminal_stderr
        self.log_file.close()


# ==============================================================================
# Checkpoint-Metadaten Callback (unverändert)
# ==============================================================================
class CheckpointMetadataCallback(TrainerCallback):
    def __init__(self, num_train_tickets=0, steps_per_epoch=0):
        super().__init__()
        self.num_train_tickets = num_train_tickets
        self.steps_per_epoch = steps_per_epoch

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        metadata_path = os.path.join(checkpoint_dir, "training_setup.txt")
        current_epoch = int(round(state.epoch))
        data_to_save = [
            f"=== Metadaten für Checkpoint ===",
            f"Aktueller Schritt (Global Step): {state.global_step}",
            f"Aktuelle Epoche (ca.): {current_epoch}", "",
            f"--- Trainings-Setup-Parameter ---",
            f"Gesamtanzahl Trainingsepochen (Wiederholungen): {int(args.num_train_epochs)}",
            f"Gesamtanzahl Trainingsschritte (Total Steps): {int(state.max_steps)}",
            f"Schritte pro Epoche (Steps per Epoch): {self.steps_per_epoch}",
            f"Anzahl Trainings-Tickets: {self.num_train_tickets}",
            f"Batch-Größe (pro Gerät): {args.per_device_train_batch_size}"
        ]
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write("\n".join(data_to_save))
            print(f"\n[Callback] ✅ Metadaten in '{metadata_path}' gespeichert.")
        except Exception as e:
            print(f"\n[Callback-Fehler] ❌ Konnte Metadaten-Datei nicht in '{metadata_path}' schreiben: {e}\n")


# ==============================================================================
# Bereinigungsfunktion (KORRIGIERT)
# ==============================================================================
def cleanup_checkpoints(output_dir: str, best_model_path: str, save_limit: int = None):
    # HINWEIS: Diese Funktion wird jetzt nur noch für Strategie 2 und 3 aufgerufen,
    # bei denen save_limit=None ist. Strategie 1 (save_limit=2) nutzt die
    # eingebaute Bereinigung des Trainers.

    print(f"\n--- Starte manuelle Bereinigung der Checkpoints in '{output_dir}' ---")
    best_checkpoint_name = None

    if save_limit is None:  # Sollte immer True sein, wenn diese Funktion aufgerufen wird
        if best_model_path and os.path.isdir(best_model_path):
            best_checkpoint_name = os.path.basename(best_model_path.rstrip(os.sep))
            print(f"Das beste Modell ist in: '{best_checkpoint_name}'. Dieser Ordner wird behalten.")
        else:
            print(f"❌ FEHLER: 'best_model_path' ('{best_model_path}') ist ungültig. Breche Bereinigung ab.")
            return
    else:
        # Dieser Fall sollte nicht mehr eintreten, aber als Sicherheitsnetz
        print("WARNUNG: cleanup_checkpoints wurde mit save_limit != None aufgerufen. Der Trainer sollte dies tun.")
        return

    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        print("Keine Checkpoint-Ordner zum Bereinigen gefunden.")
        return

    print(f"Gefunden: {len(checkpoint_dirs)} Checkpoint-Ordner. Beginne Löschvorgang...")
    deleted_count = 0
    kept_count = 0

    # KORREKTUR: Robusteres Löschen mit mehr Wiederholungen bei [WinError 5]
    for folder_path in checkpoint_dirs:
        folder_name = os.path.basename(folder_path)
        if folder_name == best_checkpoint_name:
            print(f"  ✅ '{folder_name}' wird BEHALTEN.")
            kept_count += 1
            continue

        if os.path.isdir(folder_path):
            attempts = 3
            for i in range(attempts):
                try:
                    shutil.rmtree(folder_path)
                    print(f"  🗑️  '{folder_name}' gelöscht.")
                    deleted_count += 1
                    break  # Erfolg, nächste Schleife
                except FileNotFoundError:
                    print(f"  ℹ️  '{folder_name}' wurde bereits an anderer Stelle gelöscht.")
                    break  # Erfolg, nächste Schleife
                except OSError as e:
                    print(f"  ❌ FEHLER (Versuch {i + 1}/{attempts}) beim Löschen von '{folder_name}': {e}")
                    if i < attempts - 1:
                        print(f"     ... warte 2 Sekunden und versuche es erneut ...")
                        time.sleep(2)  # Warte, falls das System die Datei noch sperrt
                    else:
                        print(f"     ... konnte '{folder_name}' nach {attempts} Versuchen nicht löschen.")

    print(f"--- Bereinigung abgeschlossen. {deleted_count} Ordner entfernt, {kept_count} behalten. ---")

    # Finaler Sweep (unverändert, ist meist unproblematisch)
    print("Starte finalen Sweep für leere Checkpoint-Ordner...")
    try:
        remaining_checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        final_deleted = 0
        for folder_path in remaining_checkpoint_dirs:
            if os.path.isdir(folder_path):
                try:
                    if not os.listdir(folder_path):
                        os.rmdir(folder_path)
                        print(f"  🗑️  Leerer Ordner '{os.path.basename(folder_path)}' im Sweep entfernt.")
                        final_deleted += 1
                except OSError:
                    pass
        if final_deleted > 0:
            print(f"  {final_deleted} leere Ordner im Sweep entfernt.")
    except Exception as e:
        print(f"  ❌ FEHLER im finalen Sweep: {e}")
    print("--- Finaler Sweep abgeschlossen. ---")


# ==============================================================================
# FUNKTIONEN ZUR STRATEGIEAUSWAHL (MODIFIZIERT)
# ==============================================================================

def select_data_file():
    print("\n" + "=" * 70)
    print("--- 0a. Wähle die Trainings-CSV-Datei ---")
    print(f"Standard: '{DEFAULT_DATA_FILE}'")
    while True:
        data_file_path = input("Enter (zum Bestätigen) oder gib einen anderen Pfad an: ").strip() or DEFAULT_DATA_FILE
        if os.path.isfile(data_file_path) and data_file_path.endswith(".csv"):
            print(f"✅ Trainings-CSV gesetzt auf: {data_file_path}")
            return data_file_path
        else:
            print(f"❌ FEHLER: Datei '{data_file_path}' nicht gefunden oder keine .csv-Datei.")


def select_column_mappings():
    print("\n" + "=" * 70)
    print("--- 0b. Definiere die Spalten-Zuweisungen ---")
    print("Gib die exakten Spaltennamen aus deiner CSV an.")
    label_col = input(f"Spaltenname für das LABEL (Priorität) [{DEFAULT_FIELD_LABEL}]: ").strip() or DEFAULT_FIELD_LABEL
    subject_col = input(
        f"Spaltenname für den BETREFF (Subject) [{DEFAULT_FIELD_SUBJECT}]: ").strip() or DEFAULT_FIELD_SUBJECT
    body_col = input(f"Spaltenname für den TEXT (Body) [{DEFAULT_FIELD_BODY}]: ").strip() or DEFAULT_FIELD_BODY
    mappings = {"label": label_col, "subject": subject_col, "body": body_col}
    print("✅ Spalten-Zuweisung:")
    print(f"   Label    -> {mappings['label']}")
    print(f"   Betreff  -> {mappings['subject']}")
    print(f"   Text     -> {mappings['body']}")
    return mappings


# --- MODIFIZIERT: Diese Funktion ist jetzt interaktiv ---
def select_training_strategy():
    """Frägt den Benutzer, welche Trainingsstrategie verwendet werden soll."""
    print("\n" + "=" * 70)
    print("--- 1. Wähle eine Trainingsstrategie ---")
    print(" [1] Optimiert (Standard): Early Stopping (anpassbar), spart Speicher (save_total_limit=2).")
    print(" [2] Vollständig (Progressiv): Kein Early Stopping, speichert nur das beste Modell am Ende.")
    print(" [3] Rewind and Retry (Experimentell): Kein Early Stopping, lädt nach jeder Epoche das beste Modell neu.")
    print("=" * 70)

    while True:
        choice = input("Wähle Strategie [1]: ").strip() or "1"

        if choice == "1":
            print("✅ Optimierte Strategie gewählt.")
            print("--- Passe Early Stopping an ---")

            default_patience = 4
            patience_val = default_patience
            while True:
                print("\nErklärung 'Patience' (Geduld):")
                print(f"Wie viele Epochen darf das Modell *keine* Verbesserung zeigen, bevor das Training stoppt?")
                patience_str = input(f"  Patience (Geduld) in Epochen [{default_patience}]: ").strip() or str(
                    default_patience)
                try:
                    patience_val = int(patience_str)
                    if patience_val > 0:
                        print(f"  -> Training stoppt nach {patience_val} Epochen ohne Verbesserung.")
                        break
                    else:
                        print("  Bitte eine Zahl größer 0 eingeben.")
                except ValueError:
                    print("  Ungültige Eingabe. Bitte eine ganze Zahl eingeben.")

            default_threshold = 0.0
            threshold_val = default_threshold
            while True:
                print("\nErklärung 'Threshold' (Schwelle):")
                print(
                    f"Welche *minimale Verbesserung* (z.B. 0.001) muss die Metrik (z.B. F1-Score) erreichen, um als 'Verbesserung' zu zählen?")
                print(f" (Standard {default_threshold} = Jede beliebige Verbesserung zählt.)")
                threshold_str = input(f"  Minimale Verbesserung (Threshold) [{default_threshold}]: ").strip() or str(
                    default_threshold)
                try:
                    threshold_val = float(threshold_str)
                    if threshold_val >= 0.0:
                        if threshold_val == 0.0:
                            print("  -> Jede (auch 0.00001) Verbesserung zählt.")
                        else:
                            print(f"  -> Metrik muss sich um mind. {threshold_val} verbessern.")
                        break
                    else:
                        print("  Bitte eine positive Zahl (z.B. 0.0 or 0.001) eingeben.")
                except ValueError:
                    print("  Ungültige Eingabe. Bitte eine Fließkommazahl eingeben.")

            return 1, 2, patience_val, threshold_val

        elif choice == "2":
            print("✅ Vollständige (Progressive) Strategie gewählt.")
            return 2, None, None, 0.0

        elif choice == "3":
            print("✅ Rewind and Retry (Experimentell) Strategie gewählt.")
            return 3, None, None, 0.0

        else:
            print("Ungültige Eingabe. Bitte '1', '2' oder '3' wählen.")


def select_optimization_metric(priority_order_list):
    print("\n" + "=" * 70)
    print("--- 2. Wähle die Optimierungs-Metrik ---")
    options = [
        ("f1_critical_high_avg", "(Durchschnitt von Critical/High) - EMPFOHLEN"),
        ("f1_weighted", "(Gesamtdurchschnitt aller Klassen)"),
        ("accuracy", "(Genauigkeit - Nicht empfohlen bei Imbalance)")]
    for label in priority_order_list:
        options.append((f"f1_{label}", f"(Nur F1-Score für '{label}')"))
    for i, (metric_name, description) in enumerate(options):
        print(f" [{i + 1}] {metric_name.ljust(24)} {description}")
    print("=" * 70)
    while True:
        choice = input(f"Wähle Metrik [1]: ").strip() or "1"
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(options):
                chosen_metric = options[choice_idx][0]
                print(f"✅ Optimierung auf '{chosen_metric}' gesetzt.")
                return chosen_metric
            else:
                print(f"Ungültige Zahl. Bitte 1-{len(options)} wählen.")
        except ValueError:
            print("Ungültige Eingabe. Bitte eine Zahl wählen.")


def select_evaluation_strategy(priority_order_list):
    print("\n" + "=" * 70)
    print("--- 3. Wähle die Evaluierungs-Strategie ---")
    print("\n [1] Prozentualer Split (Stratifiziert): (Default: 10%)")
    print("\n [2] Absoluter Split (Stratifiziert): (z.B. 500 Tickets)")
    print("\n [3] Balancierter Split (Gleiche Anzahl pro Klasse): (z.B. 50 pro Klasse)")
    print("\n [4] Separate CSV-Datei verwenden:")
    print("\n [5] Manueller Split (Anzahl pro Klasse definieren):")
    print("=" * 70)
    while True:
        choice = input("Wähle Evaluierungs-Modus [1]: ").strip() or "1"
        if choice == "1":
            while True:
                val_str = input("  Welcher Anteil (Prozent als Dezimalzahl)? [0.1]: ").strip() or "0.1"
                try:
                    val_float = float(val_str)
                    if 0.01 <= val_float < 1.0:
                        print(f"✅ Modus 'Prozentual' gewählt ({val_float * 100:.0f}%).")
                        return "percentage", val_float
                except ValueError:
                    print("  Ungültige Eingabe. Bitte eine Zahl eingeben (z.B. 0.1).")
        elif choice == "2":
            while True:
                val_str = input("  Wieviele Tickets *insgesamt*? [500]: ").strip() or "500"
                try:
                    val_int = int(val_str)
                    if val_int > 0:
                        print(f"✅ Modus 'Absolut' gewählt ({val_int} Tickets).")
                        return "absolute", val_int
                except ValueError:
                    print("  Ungültige Eingabe. Bitte eine ganze Zahl eingeben.")
        elif choice == "3":
            while True:
                val_str = input(f"  Wieviele Tickets *pro Klasse*? [50]: ").strip() or "50"
                try:
                    val_int = int(val_str)
                    if val_int > 0:
                        print(f"✅ Modus 'Balanciert' gewählt (Ziel: {val_int} Tickets pro Klasse).")
                        return "balanced", val_int
                except ValueError:
                    print("  Ungültige Eingabe. Bitte eine ganze Zahl eingeben.")
        elif choice == "4":
            while True:
                val_str = input("  Pfad zur externen Evaluierungs-CSV-Datei: ").strip()
                if os.path.isfile(val_str) and val_str.endswith(".csv"):
                    print(f"✅ Modus 'Externe CSV' gewählt.")
                    return "external_csv", val_str
                else:
                    print(f"  FEHLER: Datei '{val_str}' nicht gefunden oder keine .csv-Datei.")
        elif choice == "5":
            print("  Definiere die Anzahl der Evaluierungs-Tickets pro Klasse:")
            counts_dict = {}
            for label in priority_order_list:
                while True:
                    count_str = input(f"    Anzahl für '{label}': ").strip()
                    try:
                        count_int = int(count_str)
                        if count_int >= 0:
                            counts_dict[label] = count_int
                            break
                        else:
                            print("    Bitte eine positive Zahl eingeben.")
                    except ValueError:
                        print("    Ungültige Eingabe. Bitte eine ganze Zahl eingeben.")
            print(f"✅ Modus 'Manueller Split' gewählt.")
            print(f"   Ziel-Anzahlen: {counts_dict}")
            return "manual_per_class", counts_dict
        else:
            print("Ungültige Eingabe. Bitte '1', '2', '3', '4' oder '5' wählen.")


def select_batch_sizes():
    """Frägt den Benutzer nach den Batch-Größen."""
    print("\n" + "=" * 70)
    print("--- 4. Wähle die Batch-Größen (VRAM) ---")
    default_train = 32
    default_eval = 64
    train_batch_size = default_train
    eval_batch_size = default_eval

    try:
        train_str = input(f"Trainings-Batch-Größe (pro Gerät) [{default_train}]: ").strip() or str(default_train)
        train_batch_size = int(train_str)
    except ValueError:
        print(f"Ungültige Eingabe, verwende Standard: {default_train}")
        train_batch_size = default_train

    try:
        eval_str = input(f"Evaluierungs-Batch-Größe (pro Gerät) [{default_eval}]: ").strip() or str(default_eval)
        eval_batch_size = int(eval_str)
    except ValueError:
        print(f"Ungültige Eingabe, verwende Standard: {default_eval}")
        eval_batch_size = default_eval

    print(f"✅ Training Batch: {train_batch_size}, Evaluierung Batch: {eval_batch_size}")
    return train_batch_size, eval_batch_size


# ==============================================================================
# FUNKTION: MANUELLER SPLIT (unverändert)
# ==============================================================================

def create_manual_split(full_dataset: Dataset, label_column: str, split_counts: dict, seed: int = 42) -> (Dataset,
                                                                                                          Dataset):
    print(f"Starte manuellen Split (gemäß Vorgabe) für Spalte '{label_column}'...")
    try:
        class_label_feature = full_dataset.features[label_column]
        if not isinstance(class_label_feature, ClassLabel):
            print(f"❌ FEHLER: Spalte '{label_column}' ist kein ClassLabel-Typ.")
            return None, None
        label_to_id = {name: i for i, name in enumerate(class_label_feature.names)}
        print(f"  Label-zu-ID-Zuordnung erkannt: {label_to_id}")
    except Exception as e:
        print(f"❌ FEHLER beim Abrufen der ClassLabel-Zuordnung: {e}")
        return None, None

    try:
        print("  Konvertiere zu Pandas für Split-Logik...")
        df = full_dataset.to_pandas()
    except Exception as e:
        print(f"  FEHLER bei Konvertierung zu Pandas: {e}")
        return None, None

    eval_df_list = []
    np.random.seed(seed)
    all_eval_indices = set()

    for label_name, n_to_sample_requested in split_counts.items():
        label_id = label_to_id.get(label_name)
        if label_id is None:
            print(f"  INFO: Label-Name '{label_name}' aus split_counts nicht in PRIORITY_ORDER. Überspringe.")
            continue

        class_df = df[df[label_column] == label_id]
        n_available = len(class_df)
        n_to_sample_actual = min(n_available, n_to_sample_requested)

        if n_available == 0:
            if n_to_sample_requested > 0: print(f"  WARNUNG: Klasse '{label_name}' (ID: {label_id}) hat 0 Tickets.")
            continue
        if n_to_sample_actual == 0:
            if n_to_sample_requested > 0: print(
                f"  INFO: Klasse '{label_name}' (ID: {label_id}) - 0 für Eval ausgewählt.")
            continue
        if n_available < n_to_sample_requested:
            print(
                f"  WARNUNG: Klasse '{label_name}' (ID: {label_id}) hat nur {n_available} Tickets (< {n_to_sample_requested}). Verwende alle.")
        elif n_available == n_to_sample_actual:
            print(
                f"  WARNUNG: Klasse '{label_name}' (ID: {label_id}) hat exakt {n_available} Tickets. Alle für Eval, 0 für Training.")
        else:
            print(
                f"  Klasse '{label_name}' (ID: {label_id}): {n_to_sample_actual} von {n_available} Tickets für Eval ausgewählt.")

        eval_sample = class_df.sample(n=n_to_sample_actual, random_state=seed)
        eval_df_list.append(eval_sample)
        all_eval_indices.update(eval_sample.index)

    if not eval_df_list:
        print("FEHLER: Evaluierungsset ist leer (vielleicht alle Anzahlen auf 0 gesetzt?).")
        return None, None

    eval_df = pd.concat(eval_df_list).sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df = df.drop(all_eval_indices).sample(frac=1, random_state=seed).reset_index(drop=True)

    if train_df.empty:
        print("FEHLER: Trainingsset ist leer nach dem Split.")
        return None, None

    print(f"  Split abgeschlossen: {len(train_df)} Trainings-Tickets, {len(eval_df)} Evaluierungs-Tickets.")
    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(eval_df)

    try:
        train_ds = train_ds.cast_column(label_column, class_label_feature)
        eval_ds = eval_ds.cast_column(label_column, class_label_feature)
        print("  ✅ Datasets erfolgreich zurück in ClassLabel-Typ konvertiert.")
    except Exception as e:
        print(f"❌ FEHLER beim Zurück-Konvertieren der Datasets: {e}")
        return None, None

    return train_ds, eval_ds


# --- MODIFIZIERT: Funktion zur Analyse und interaktiven Auswahl der Token-Länge ---
def calculate_dynamic_max_length(
        train_dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        field_subject: str,
        field_body: str,
        neg_vocab: list,
        pos_vocab: list,
        sla_vocab: list
) -> int:
    """
    Analysiert das Trainingsset, um eine VRAM-optimierte max_length zu finden,
    und lässt den Benutzer diese Einstellung bestätigen oder anpassen.
    """
    print("\n" + "=" * 70)
    print("--- 6a. Analysiere Token-Längen für VRAM-Optimierung ---")

    def get_token_length(examples):
        texts = [str(body) + " " + str(subject) for body, subject in
                 zip(examples[field_body], examples[field_subject])]

        enriched_texts = [
            preprocess_with_vocab(
                text, neg_vocab, pos_vocab, sla_vocab,
                sla_weight=KEYWORD_WEIGHTS["sla_weight"],
                neg_weight=KEYWORD_WEIGHTS["neg_weight"],
                pos_weight=KEYWORD_WEIGHTS["pos_weight"]
            )
            for text in texts
        ]

        tokenized_inputs = tokenizer(enriched_texts, truncation=False, padding=False)
        return {"token_length": [len(x) for x in tokenized_inputs["input_ids"]]}

    print("Analysiere Längen im Trainings-Set (kann einen Moment dauern)...")

    all_lengths_np = None
    try:
        dataset_with_lengths = train_dataset.map(get_token_length, batched=True)
        all_lengths_np = np.array(dataset_with_lengths['token_length'])
    except Exception as e:
        print(f"❌ FEHLER bei der Token-Analyse: {e}.")
        print(f"   -> Verwende Standard max_length von 512.")
        import traceback
        traceback.print_exc()
        return 512

    if all_lengths_np is None or len(all_lengths_np) == 0:
        print("❌ FEHLER: Keine Token-Längen gefunden. Verwende Standard 512.")
        return 512

    # Metriken berechnen
    try:
        total_tickets = len(all_lengths_np)
        max_len = int(all_lengths_np.max())
        p95_len = int(np.percentile(all_lengths_np, 95))
        avg_len = int(all_lengths_np.mean())
        model_max = tokenizer.model_max_length  # (z.B. 512 für DistilBERT)

        print(f"  Statistiken (aus {total_tickets} Tickets):")
        print(f"    - Längstes Ticket:      {max_len} Tokens")
        print(f"    - 95%-Perzentil:        {p95_len} Tokens")
        print(f"    - Durchschnitt:         {avg_len} Tokens")
        print(f"    - Modell-Limit:         {model_max} Tokens")

        # Empfehlung berechnen (95%-Perzentil, auf 64 aufgerundet, max. 512)
        target_len = p95_len
        if target_len % 64 != 0:
            recommended_length = int(math.ceil(target_len / 64.0)) * 64
        else:
            recommended_length = target_len

        recommended_length = min(recommended_length, model_max)

        # Interaktive Schleife
        while True:
            print("\n--- Empfehlung für 'max_length' ---")
            print(
                f"Basierend auf dem 95%-Perzentil ({p95_len}) wird eine 'max_length' von **{recommended_length}** empfohlen.")

            # Berechne Ausreißer basierend auf der Empfehlung
            outliers = np.sum(all_lengths_np > recommended_length)
            outlier_percent = (outliers / total_tickets) * 100
            print(
                f" -> Bei {recommended_length} Tokens werden {outliers} Tickets ({outlier_percent:.2f}%) abgeschnitten (truncated).")
            print(f"    (Das Modell 'sieht' bei diesen Tickets nur die ersten {recommended_length} Tokens.)")

            user_input = input(f"\nFinale 'max_length' bestätigen [{recommended_length}]: ").strip()

            if not user_input:
                final_chosen_length = recommended_length
                print(f"✅ Empfehlung ({final_chosen_length}) übernommen.")
                break

            try:
                final_chosen_length = int(user_input)
                if final_chosen_length <= 0:
                    print("  Bitte eine positive Zahl eingeben.")
                elif final_chosen_length > model_max:
                    print(f"  WARNUNG: Eingabe ({final_chosen_length}) ist größer als das Modell-Limit ({model_max}).")
                    print(f"  Wert wird auf {model_max} begrenzt.")
                    final_chosen_length = model_max
                    break
                else:
                    # Zeige die Konsequenz der manuellen Wahl
                    final_outliers = np.sum(all_lengths_np > final_chosen_length)
                    final_percent = (final_outliers / total_tickets) * 100
                    print(f"  -> Manuell auf {final_chosen_length} gesetzt.")
                    print(f"     {final_outliers} Tickets ({final_percent:.2f}%) werden nun abgeschnitten.")
                    break
            except ValueError:
                print("  Ungültige Eingabe. Bitte eine ganze Zahl eingeben.")

        return final_chosen_length

    except Exception as e:
        print(f"❌ FEHLER bei der Metrik-Berechnung: {e}.")
        print(f"   -> Verwende Standard max_length von 512.")
        return 512


# --- NEU: Funktion zum Speichern der Konfiguration ---
def save_config_summary(filepath: str, config_data: dict):
    """Speichert die Konfigurations-Parameter in einer Textdatei."""
    print(f"\nSpeichere Konfigurationsübersicht in '{filepath}'...")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=== Zusammenfassung der Trainings-Konfiguration ===\n")
            f.write(f"Durchlauf gestartet: {config_data.pop('start_timestamp', 'N/A')}\n")
            f.write(f"Log-Datei: {config_data.pop('log_file', 'N/A')}\n")
            f.write(f"Ausgabeordner: {config_data.pop('output_dir', 'N/A')}\n")

            f.write("\n--- 1. Datenquellen ---\n")
            f.write(f"CSV-Datei: {config_data.pop('data_file_path', 'N/A')}\n")
            f.write(f"Spalte (Label): {config_data.pop('field_label', 'N/A')}\n")
            f.write(f"Spalte (Betreff): {config_data.pop('field_subject', 'N/A')}\n")
            f.write(f"Spalte (Text): {config_data.pop('field_body', 'N/A')}\n")

            f.write("\n--- 2. Trainings-Strategie ---\n")
            strategy_id = config_data.pop('strategy_id', 'N/A')
            f.write(f"Strategie-ID: {strategy_id}\n")
            if strategy_id == 1:
                f.write(f"  -> Modus: Optimiert (Early Stopping)\n")
                f.write(f"  -> Patience: {config_data.pop('strategy_patience', 'N/A')} Epochen\n")
                f.write(f"  -> Threshold: {config_data.pop('strategy_threshold', 'N/A')}\n")
            elif strategy_id == 2:
                f.write(f"  -> Modus: Vollständig (Progressiv)\n")
            elif strategy_id == 3:
                f.write(f"  -> Modus: Rewind and Retry\n")
            f.write(f"Checkpoint-Limit: {config_data.pop('strategy_save_limit', 'N/A')}\n")
            f.write(f"Optimierungs-Metrik: {config_data.pop('chosen_metric', 'N/A')}\n")

            f.write("\n--- 3. Evaluierungs-Split ---\n")
            f.write(f"Modus: {config_data.pop('eval_mode', 'N/A')}\n")
            f.write(f"Wert: {str(config_data.pop('eval_value', 'N/A'))}\n")  # str() für Dicts

            f.write("\n--- 4. VRAM / Batch-Parameter ---\n")
            f.write(f"Trainings-Batch-Größe: {config_data.pop('train_batch_size', 'N/A')}\n")
            f.write(f"Evaluierungs-Batch-Größe: {config_data.pop('eval_batch_size', 'N/A')}\n")
            f.write(f"Dynamische max_length (gewählt): {config_data.pop('dynamic_max_len', 'N/A')}\n")

            if config_data:
                f.write("\n--- 5. Weitere Parameter ---\n")
                for key, val in config_data.items():
                    if key not in ['strategy_patience', 'strategy_threshold']:
                        f.write(f"{key}: {val}\n")

        print("✅ Konfiguration gespeichert.")
    except Exception as e:
        print(f"❌ WARNUNG: Konnte Konfigurationsübersicht nicht speichern: {e}")


# ==============================================================================
# HAUPT-TRAININGSFUNKTION (main) (MODIFIZIERT)
# ==============================================================================

def main():
    # --- 0. Logging & Ausgabeordner ---
    base_log_dir = "logs_multilingual_stratified_512"
    timestamp_date = datetime.now().strftime("%Y-%m-%d")
    default_output_dir = f"./ergebnisse_multilingual_stratified_eval_modified_{timestamp_date}"

    print("\n" + "=" * 70)
    print("--- 0c. Wähle den Ausgabeordner ---")
    print(f"Standardmäßig wird: '{default_output_dir}' vorgeschlagen.")
    output_dir_input = input("Enter (zum Bestätigen) oder gib einen anderen Pfad an: ").strip()
    output_dir = output_dir_input or default_output_dir
    print(f"✅ Ausgabeordner gesetzt auf: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)

    # Logging-Konfiguration
    if os.path.isfile(base_log_dir):
        print(f"⚠️  Warnung: Datei '{base_log_dir}' blockiert Log-Verzeichnis.")
        backup_name = f"logs_multilingual_als_datei_{int(time.time())}.txt"
        os.rename(base_log_dir, backup_name)

    timestamp_log = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_name = "train_modell_multilingual_stratified_eval_modified"
    run_log_dir = os.path.join(base_log_dir, f"{timestamp_log}_{script_name}")
    os.makedirs(run_log_dir, exist_ok=True)
    console_log_path = os.path.join(run_log_dir, f"{timestamp_log}_{script_name}_console.log")

    logger = ConsoleLogger(console_log_path)
    logger.hook_stdout()
    logger.hook_stderr()

    # --- try...finally-Block ---
    try:
        # === 0a/0b. Datendatei und Spalten wählen ===
        data_file_path = select_data_file()
        field_mappings = select_column_mappings()
        field_label = field_mappings['label']
        field_subject = field_mappings['subject']
        field_body = field_mappings['body']

        print(f"Starte den HYBRID-Trainingsprozess...")
        print(f"Alle Konsolenausgaben werden in '{console_log_path}' gespeichert.")
        print("=" * 70)

        # === 1, 2, 3. Strategien wählen ===
        strategy_id, strategy_save_limit, strategy_patience, strategy_threshold = select_training_strategy()
        chosen_metric = select_optimization_metric(PRIORITY_ORDER)
        eval_mode, eval_value = select_evaluation_strategy(PRIORITY_ORDER)

        # === 4. Batch-Größen wählen ===
        train_batch_size, eval_batch_size = select_batch_sizes()
        print("=" * 70)

        # === 5. Vokabular-Management ===
        generate_vocab_files_if_needed(data_file_path, field_subject, field_body, field_label)
        neg_vocab, pos_vocab, sla_vocab = load_vocab_from_csvs()

        # === 6. GPU-Prüfung und Setup ===
        is_gpu_available = torch.cuda.is_available()
        if is_gpu_available:
            print("✅ GPU gefunden! Das Training wird auf der GPU ausgeführt. 🚀")
        else:
            print("⚠️ Keine GPU gefunden. Das Training wird auf der CPU ausgeführt (deutlich langsamer).")
        print(f"➡️  Aktuelles Arbeitsverzeichnis: {os.getcwd()}")

        # --- Logik für bestehenden Ausgabeordner ---
        overwrite_output = False
        resume_from_checkpoint = False
        if os.path.isdir(output_dir) and os.listdir(output_dir):
            print(f"⚠️  Es sind bereits Daten im Ausgabeverzeichnis '{output_dir}' vorhanden.")
            print("   [1] Training fortsetzen (weitere Epochen trainieren)")
            print("   [2] ALLES ÜBERSCHREIBEN (bisheriges Modell löschen)")
            print("   [3] Abbrechen")
            while True:
                choice = input("Wähle eine Option [1, 2, 3]: ").strip()
                if choice == '1':
                    print("✅ Training wird fortgesetzt. Lade letzten Checkpoint...")
                    overwrite_output = False;
                    resume_from_checkpoint = True;
                    break
                elif choice == '2':
                    print("✅ Vorhandene Daten werden überschrieben.")
                    overwrite_output = True;
                    resume_from_checkpoint = False;
                    break
                elif choice == '3':
                    print("❌ Vorgang vom Benutzer abgebrochen.");
                    sys.exit()
                else:
                    print("Ungültige Eingabe. Bitte '1', '2' oder '3' eingeben.")
        else:
            overwrite_output = False;
            resume_from_checkpoint = False

        print(f"Logs für diesen Durchlauf werden in '{run_log_dir}' gespeichert.")

        # === 5. Labels (Prioritäten) umwandeln (Definition) ===
        print("Definiere 'priority'-Spalte als ClassLabel mit fester Reihenfolge...")
        class_label_feature = ClassLabel(names=PRIORITY_ORDER)
        num_unique_labels = len(PRIORITY_ORDER)

        # === 4. Dataset laden und aufteilen (Dynamisch) ===
        dataset = None
        try:
            if eval_mode == "external_csv":
                print(f"Lade Trainings-Dataset von: {data_file_path}")
                train_ds = load_dataset('csv', data_files=data_file_path, split="train")
                eval_file_path = eval_value
                print(f"Lade externes Evaluierungs-Dataset von: {eval_file_path}")
                eval_ds = load_dataset('csv', data_files=eval_file_path, split="train")
                print("Wandle 'priority'-Spalte für externes Set in ClassLabel um...")
                try:
                    train_ds = train_ds.cast_column(field_label, class_label_feature)
                    eval_ds = eval_ds.cast_column(field_label, class_label_feature)
                    print(f"✅ '{field_label}'-Spalte erfolgreich in {num_unique_labels} Labels umgewandelt.")
                except (ValueError, KeyError) as e:
                    print(f"❌ FEHLER beim Umwandeln der '{field_label}'-Spalte im externen Set: {e}")
                    sys.exit()
                dataset = DatasetDict({"train": train_ds, "validation": eval_ds})

            else:
                print(f"Lade gesamtes Dataset von: {data_file_path}")
                full_dataset = load_dataset('csv', data_files=data_file_path, split="train")
                print("Wandle 'priority'-Spalte für internen Split in ClassLabel um...")
                try:
                    full_dataset = full_dataset.cast_column(field_label, class_label_feature)
                    print(f"✅ '{field_label}'-Spalte erfolgreich in {num_unique_labels} Labels umgewandelt.")
                except (ValueError, KeyError) as e:
                    print(f"❌ FEHLER beim Umwandeln der '{field_label}'-Spalte im Haupt-Dataset: {e}")
                    sys.exit()

                if eval_mode == "percentage":
                    print(f"Erstelle {eval_value * 100:.0f}% prozentualen, stratifizierten Split...")
                    split = full_dataset.train_test_split(test_size=eval_value, seed=42, stratify_by_column=field_label)
                    dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

                elif eval_mode == "absolute":
                    print(f"Erstelle {eval_value} absoluten, stratifizierten Split...")
                    split_size = eval_value
                    if eval_value > len(full_dataset):
                        print(f"WARNUNG: Größe ({eval_value}) > Dataset ({len(full_dataset)}). Nutze 10% Fallback.")
                        split_size = 0.1
                    split = full_dataset.train_test_split(test_size=split_size, seed=42, stratify_by_column=field_label)
                    dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

                elif eval_mode == "balanced":
                    print(f"Erstelle 'Balancierten' Split (Ziel: {eval_value} pro Klasse)...")
                    split_counts = {label: eval_value for label in PRIORITY_ORDER}
                    train_ds, eval_ds = create_manual_split(full_dataset, field_label, split_counts, seed=42)
                    if train_ds is None or eval_ds is None: sys.exit()
                    dataset = DatasetDict({"train": train_ds, "validation": eval_ds})

                elif eval_mode == "manual_per_class":
                    print("Erstelle 'Manuellen (pro Klasse)' Split...")
                    split_counts = eval_value
                    train_ds, eval_ds = create_manual_split(full_dataset, field_label, split_counts, seed=42)
                    if train_ds is None or eval_ds is None: sys.exit()
                    dataset = DatasetDict({"train": train_ds, "validation": eval_ds})

        except FileNotFoundError as e:
            print(f"❌ FEHLER: Datei nicht gefunden: {e}");
            sys.exit()
        except Exception as e:
            print(f"❌ FEHLER beim Laden oder Aufteilen der Daten: {e}")
            import traceback;
            traceback.print_exc();
            sys.exit()

        print(
            f"✅ Dataset-Aufteilung abgeschlossen: {len(dataset['train'])} Trainings-, {len(dataset['validation'])} Validierungs-Beispiele.")

        # === 6. Modell und Tokenizer laden ===
        print("Lade das Basis-Modell und den Tokenizer...")
        modell_name = "distilbert/distilbert-base-multilingual-cased"
        try:
            tokenizer = AutoTokenizer.from_pretrained(modell_name)
            model = AutoModelForSequenceClassification.from_pretrained(modell_name, num_labels=num_unique_labels)
        except OSError:
            print(f"❌ FEHLER: Modell '{modell_name}' nicht gefunden. (Internetverbindung?)");
            sys.exit()

        tokenizer = add_new_tokens_to_tokenizer(tokenizer)
        model.resize_token_embeddings(len(tokenizer))
        print("✅ Tokenizer und Modell um neue Signal-Tokens erweitert.")

        # === 6a. Dynamische max_length berechnen (INTERAKTIV) ===
        dynamic_max_len = calculate_dynamic_max_length(
            dataset['train'], tokenizer, field_subject, field_body,
            neg_vocab, pos_vocab, sla_vocab
        )

        # === 6b. Konfiguration speichern ===
        config_to_save = {
            "start_timestamp": timestamp_log,
            "log_file": console_log_path,
            "output_dir": output_dir,
            "data_file_path": data_file_path,
            "field_label": field_label,
            "field_subject": field_subject,
            "field_body": field_body,
            "strategy_id": strategy_id,
            "strategy_patience": strategy_patience,
            "strategy_threshold": strategy_threshold,
            "strategy_save_limit": strategy_save_limit,
            "chosen_metric": chosen_metric,
            "eval_mode": eval_mode,
            "eval_value": eval_value,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "dynamic_max_len": dynamic_max_len
        }
        config_summary_path = os.path.join(output_dir, "training_setup_summary.txt")
        save_config_summary(config_summary_path, config_to_save)

        print("=" * 70)

        # === 7. Tokenize-Funktion definieren und anwenden ===

        def tokenize_and_enrich_function(examples):
            raw_texts = [str(body) + " " + str(subject) for body, subject in
                         zip(examples[field_body], examples[field_subject])]

            enriched_texts = [
                preprocess_with_vocab(
                    text, neg_vocab, pos_vocab, sla_vocab,
                    sla_weight=KEYWORD_WEIGHTS["sla_weight"],
                    neg_weight=KEYWORD_WEIGHTS["neg_weight"],
                    pos_weight=KEYWORD_WEIGHTS["pos_weight"]
                )
                for text in raw_texts
            ]

            return tokenizer(
                enriched_texts,
                padding="max_length",
                truncation=True,
                max_length=dynamic_max_len  # Verwendet dynamischen Wert
            )

        print(f"Starte Anreicherung und Tokenisierung des Datasets (dynamische max_length={dynamic_max_len})...")
        tokenized_datasets = dataset.map(tokenize_and_enrich_function, batched=True)

        # === 8. Finale Vorbereitung (Spalten aufräumen) ===
        print(f"Benenne die '{field_label}'-Spalte in 'labels' um...")
        tokenized_datasets = tokenized_datasets.rename_column(field_label, "labels")
        try:
            columns_to_remove = [field_subject, field_body] + FIELDS_TO_REMOVE
            final_columns_to_remove = [col for col in columns_to_remove if
                                       col in tokenized_datasets["train"].column_names]
            print(f"Entferne nicht benötigte Spalten: {final_columns_to_remove}")
            tokenized_datasets = tokenized_datasets.remove_columns(final_columns_to_remove)
        except (ValueError, KeyError) as e:
            print(f"Hinweis: Einige Spalten zum Entfernen ({e}) wurden nicht gefunden, fahre fort.")
            pass

        # === 9. Trainings-Argumente definieren (Dynamisch) ===
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=chosen_metric,
            save_total_limit=strategy_save_limit,
            greater_is_better=True,
            num_train_epochs=30,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_dir=run_log_dir,
            logging_strategy="epoch",
            overwrite_output_dir=overwrite_output,
            report_to="none",
            fp16=is_gpu_available,
        )

        # === 9.1 Metadaten für den Callback ===
        num_train_tickets_for_callback = len(tokenized_datasets["train"])
        batch_size_for_callback = training_args.per_device_train_batch_size
        steps_per_epoch_for_callback = (num_train_tickets_for_callback // batch_size_for_callback) + \
                                       (1 if num_train_tickets_for_callback % batch_size_for_callback > 0 else 0)

        # === 10. Trainer initialisieren ===
        callbacks_list = [
            CheckpointMetadataCallback(
                num_train_tickets=num_train_tickets_for_callback,
                steps_per_epoch=steps_per_epoch_for_callback
            )
        ]
        if strategy_id == 1:
            print(
                f"Aktiviere Early Stopping (Strategie 1) mit Geduld={strategy_patience} und Schwelle={strategy_threshold}...")
            callbacks_list.append(
                EarlyStoppingCallback(
                    early_stopping_patience=strategy_patience,
                    early_stopping_threshold=strategy_threshold
                )
            )
        else:
            print(f"Strategie {strategy_id}: Early Stopping ist DEAKTIViert.")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            compute_metrics=compute_metrics,
            callbacks=callbacks_list
        )

        # ==================================================================
        # === SCHRITT 11: Training starten (unverändert) ===
        # ==================================================================
        train_result = None
        if strategy_id == 3:  # Rewind and Retry
            print(f"\n--- Starte 'Rewind and Retry' Training (optimiert auf '{chosen_metric}') ---")
            current_best_checkpoint = None
            num_epochs = int(training_args.num_train_epochs)
            if resume_from_checkpoint:
                print("Info: 'Fortsetzen' ist aktiv. Der Trainer wird den letzten Checkpoint suchen.")
                current_best_checkpoint = True
            for epoch in range(num_epochs):
                print(f"\n" + "=" * 80)
                print(f"--- Starte Epoche {epoch + 1} / {num_epochs} (Rewind-Modus) ---")
                trainer.args.max_steps = (epoch + 1) * steps_per_epoch_for_callback
                print(f"Max steps gesetzt auf: {trainer.args.max_steps}")
                print(f"Starte Training von: {current_best_checkpoint or 'Anfang (Epoche 0)'}")
                train_result = trainer.train(resume_from_checkpoint=current_best_checkpoint)
                current_best_checkpoint = trainer.state.best_model_checkpoint
                if current_best_checkpoint:
                    print(
                        f"✅ Bester Checkpoint nach Epoche {epoch + 1} ist: {os.path.basename(current_best_checkpoint)}")
                else:
                    print(f"⚠️ Warnung: Konnte besten Checkpoint nicht finden nach Epoche {epoch + 1}.")
            print("=" * 80)
            print("--- 'Rewind and Retry' Training abgeschlossen ---")
        else:  # Standard oder Vollständig (Strategie 1 oder 2)
            print(f"\n--- Starte Standard-Training (optimiert auf '{chosen_metric}') ---")
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # === Schritt 12: Modell explizit speichern (unverändert) ===
        print(f"Speichere das finale *beste* Modell (basierend auf '{chosen_metric}')...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"\n🎉 Training erfolgreich abgeschlossen! Das beste Modell wurde im Ordner '{output_dir}' gespeichert.")

        # ==================================================================
        # === SCHRITT 13: Redundante Checkpoints bereinigen (KORRIGIERT) ===
        # ==================================================================
        best_model_path = trainer.state.best_model_checkpoint
        if best_model_path is None and train_result is not None:
            best_model_path = train_result.best_model_checkpoint  # Fallback

        # --- KORREKTUR: Cleanup nur für Strategie 2 & 3 ---
        if strategy_id == 1:
            print("\n--- Checkpoint-Bereinigung (Strategie 1) ---")
            print(f"Trainer hat 'save_total_limit={strategy_save_limit}' genutzt und automatisch bereinigt.")
            print("Keine manuelle Bereinigung notwendig.")
            if best_model_path:
                print(f"Bestes Modell (geladen): {os.path.basename(best_model_path)}")
            else:
                print("WARNUNG: Konnte das beste Modell nicht identifizieren (Trainer-State).")
        else:
            # Nur für Strategie 2 und 3, wo save_total_limit=None war
            print("\n--- Checkpoint-Bereinigung (Strategie 2/3) ---")
            cleanup_checkpoints(output_dir, best_model_path, strategy_save_limit)


    except Exception as e:
        print(f"❌ EIN SCHWERWIEGENDER FEHLER IST AUFGETRETEN:")
        import traceback
        traceback.print_exc()
    finally:
        logger.close_logs()
        print(f"Konsolen-Logging beendet. Log-Datei gespeichert in: '{console_log_path}'")


if __name__ == "__main__":
    main()