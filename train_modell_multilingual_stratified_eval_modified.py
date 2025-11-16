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
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from datasets import load_dataset, ClassLabel, Features, Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- Globale Konfiguration ---
BASE_DIR = "trainingsdaten"
DATA_FILE = os.path.join(BASE_DIR, "5_prio_stufen/dataset-tickets-german_normalized_50_5_2.csv")

# Vokabular-Dateien (werden automatisch generiert, falls nicht vorhanden)
NEG_CSV = os.path.join(BASE_DIR, "neg_arguments_multilingual.csv")
POS_CSV = os.path.join(BASE_DIR, "pos_arguments_multilingual.csv")
SLA_CSV = os.path.join(BASE_DIR, "sla_arguments_multilingual.csv")
# MODIFIZIERT: Pfad zur Stopwords-CSV
STOPWORDS_CSV = os.path.join(BASE_DIR, "stopwords.csv")

# Diese Reihenfolge ist jetzt KRITISCH für die Per-Klassen-Metriken
PRIORITY_ORDER = ["critical", "high", "medium", "low", "very_low"]
HIGH_PRIO_CLASSES_STR = ["critical", "high"]  # Für Vokabular-Generierung
TOP_N_TERMS = 75  # Anzahl der Top-Begriffe für pos/neg Vokabular
MIN_DF = 5  # Minimale Dokumentfrequenz für Begriffe bei Vokabular-Generierung
NEW_TOKENS = ['KEY_CORE_APP', 'KEY_CRITICAL', 'KEY_REQUEST', 'KEY_NORMAL']

# ==============================================================================
# FELDZUORDNUNGEN (CSV-Spaltennamen)
# ==============================================================================
FIELD_LABEL = "priority"  # Spalte, die die Priorität (Label) enthält
FIELD_SUBJECT = "subject"  # Spalte, die den Betreff enthält
FIELD_BODY = "body"  # Spalte, die den Textkörper enthält
FIELDS_TO_REMOVE = ['queue', 'language']

# ==============================================================================
# KEYWORD-GEWICHTUNG (Anpassbar)
# ==============================================================================
KEYWORD_WEIGHTS = {
    "sla_weight": 5,  # (KEY_CORE_APP)
    "neg_weight": 4,  # (KEY_CRITICAL)
    "pos_weight": 1  # (KEY_REQUEST)
}
# ==============================================================================


# --- Stopwörter-Management (MODIFIZIERT) ---
try:
    from stop_words import get_stop_words

    # Lade Standard-Stopwörter für Deutsch
    GERMAN_STOPWORDS = get_stop_words('de')
except ImportError:
    print("WARNUNG: Paket 'stop-words' nicht gefunden. (pip install stop-words)")
    print("Fahre ohne deutsche Stopwörter fort.")
    GERMAN_STOPWORDS = []

# --- NEU: Lade Stopwörter aus CSV (Request 1) ---
# Die manuelle CUSTOM_STOPWORDS-Liste wird entfernt.
# Das Skript lädt stattdessen die generierte/bearbeitete CSV.
if os.path.exists(STOPWORDS_CSV):
    try:
        print(f"Lade benutzerdefinierte Stopwörter aus {STOPWORDS_CSV}...")
        df_stop = pd.read_csv(STOPWORDS_CSV)
        # Lese alle Begriffe aus der 'term'-Spalte
        custom_stopwords_from_csv = df_stop['term'].dropna().astype(str).tolist()

        # Füge sie zur Hauptliste hinzu
        GERMAN_STOPWORDS.extend(custom_stopwords_from_csv)
        print(f"✅ Stopwörter-Liste um {len(custom_stopwords_from_csv)} Wörter aus {STOPWORDS_CSV} erweitert.")
    except Exception as e:
        print(f"⚠️ WARNUNG: Konnte {STOPWORDS_CSV} nicht laden oder verarbeiten: {e}")
        print("   Stelle sicher, dass die Datei eine Spalte 'term' enthält.")
else:
    print(f"ℹ️  Keine {STOPWORDS_CSV} gefunden.")
    print("   Nur Standard-Stopwörter werden für die Vokabular-Generierung (Phase 1) verwendet.")
    print(f"   (Die Datei wird automatisch erstellt, wenn Phase 1 (Vokabular-Generierung) läuft.)")


# -------------------------------------------


# ==============================================================================
# COMPUTE_METRICS FUNKTION (unverändert)
# ==============================================================================

def compute_metrics(pred):
    """
    Berechnet Metriken für die Evaluierung (z.B. F1, Accuracy).
    Wird nach jeder Epoche auf dem Validierungs-Set aufgerufen.
    """
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

def generate_vocab_files_if_needed():
    """
    Phase 1: Vokabular-Generierung.
    Erstellt pos/neg/sla Listen UND die stopwords.csv, falls sie fehlen.
    """
    os.makedirs(BASE_DIR, exist_ok=True)

    # Prüfen, ob die *Haupt*-Vokabularlisten (pos/neg/sla) bereits existieren.
    if os.path.exists(NEG_CSV) and os.path.exists(POS_CSV) and os.path.exists(SLA_CSV):
        print("✅ Phase 1: Vokabular-Dateien (pos/neg/sla) gefunden. Überspringe automatische Generierung.")
        return

    # === Start der Generierung ===
    print("⚠️ Phase 1: Vokabular-Dateien nicht gefunden. Starte automatische Extraktion...")
    if not os.path.exists(DATA_FILE):
        print(f"❌ FEHLER: Trainingsdatensatz {DATA_FILE} nicht gefunden.")
        sys.exit()
    try:
        df = pd.read_csv(DATA_FILE)
        df['text'] = df[FIELD_SUBJECT].fillna('') + ' ' + df[FIELD_BODY].fillna('')
        df = df.dropna(subset=['text', FIELD_LABEL])
    except KeyError as e:
        print(f"❌ FEHLER: Die Spalte {e} wurde in {DATA_FILE} nicht gefunden.")
        print(f"   Stellen Sie sicher, dass die FELDZUORDNUNGEN korrekt sind.")
        sys.exit()

    print(f"Analysiere {len(df)} Tickets für Vokabular-Extraktion...")
    df_high_prio = df[df[FIELD_LABEL].isin(HIGH_PRIO_CLASSES_STR)]
    df_low_prio = df[~df[FIELD_LABEL].isin(HIGH_PRIO_CLASSES_STR)]

    if df_high_prio.empty or df_low_prio.empty:
        print(f"❌ FEHLER: Konnte keine Tickets für hohe Priorität (Werte: {HIGH_PRIO_CLASSES_STR}) finden.")
        sys.exit()

    # Initialisiere TF-IDF Vectorizer
    # Nutzt die erweiterte Stopwörter-Liste (aus CSV oder nur Standard)
    vectorizer = TfidfVectorizer(
        stop_words=GERMAN_STOPWORDS,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=MIN_DF
    )

    print("Passe TfidfVectorizer an alle Texte an...")
    vectorizer.fit(df['text'])

    # --- Stopword-Vorschlagsliste generieren ---
    if not os.path.exists(STOPWORDS_CSV):
        print(f"Generiere Stopword-Vorschlagsliste unter {STOPWORDS_CSV}...")
        try:
            feature_names_idf = vectorizer.get_feature_names_out()
            idf_scores = vectorizer.idf_
            idf_df = pd.DataFrame({'term': feature_names_idf, 'idf_score': idf_scores})

            # Sortiere nach niedrigstem IDF-Score (häufigste Wörter)
            idf_df = idf_df.sort_values(by='idf_score', ascending=True)
            idf_df = idf_df[~idf_df['term'].str.match(r'^\d+$')]
            idf_df = idf_df[idf_df['term'].str.len() > 2]

            top_stopwords = idf_df.head(200)['term'].tolist()
            pd.DataFrame(top_stopwords, columns=['term']).to_csv(STOPWORDS_CSV, index=False)
            print(f"✅ Stopword-Vorschlagsliste mit {len(top_stopwords)} Wörtern gespeichert.")
            print("   Bitte überprüfen Sie diese Liste manuell und starten Sie das Skript erneut.")
            print("   (Die geladenen Stopwörter werden dann beim nächsten Lauf verwendet.)")
        except Exception as e:
            print(f"❌ FEHLER beim Generieren der Stopword-Liste: {e}")
    else:
        # Diese Meldung wird jetzt am Skript-Anfang angezeigt
        pass
    # --- Ende Stopword-Generierung ---

    # Führe die Transformation für Hoch/Niedrig-Prio-Gruppen durch
    print("Transformiere Hoch- und Niedrig-Prio-Texte für TF-IDF-Vergleich...")
    tfidf_high = vectorizer.transform(df_high_prio['text'])
    tfidf_low = vectorizer.transform(df_low_prio['text'])

    mean_tfidf_high = np.asarray(tfidf_high.mean(axis=0)).ravel()
    mean_tfidf_low = np.asarray(tfidf_low.mean(axis=0)).ravel()
    score_diff = mean_tfidf_high - mean_tfidf_low

    feature_names = np.array(vectorizer.get_feature_names_out())
    results_df = pd.DataFrame({'term': feature_names, 'score_diff': score_diff})

    # Speichere die Listen
    top_neg_terms = results_df.sort_values(by='score_diff', ascending=False).head(TOP_N_TERMS)['term'].tolist()
    top_pos_terms = results_df.sort_values(by='score_diff', ascending=True).head(TOP_N_TERMS)['term'].tolist()

    pd.DataFrame(top_neg_terms, columns=['term']).to_csv(NEG_CSV, index=False)
    pd.DataFrame(top_pos_terms, columns=['term']).to_csv(POS_CSV, index=False)

    if not os.path.exists(SLA_CSV):
        pd.DataFrame(columns=['term']).to_csv(SLA_CSV, index=False)

    print(f"✅ Phase 1: Extraktion abgeschlossen. Dateien gespeichert in '{BASE_DIR}'.")
    print(f"👉 WICHTIG: Bitte bearbeite nun die CSV-Dateien (besonders {SLA_CSV})")
    print("   und füge dein Domänenwissen hinzu. Starte das Skript danach erneut.")
    print("-" * 70)


def load_vocab_from_csvs() -> (list, list, list):
    """Lädt die Vokabularlisten (neg, pos, sla) aus den CSV-Dateien."""
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
        sla_weight: int = 1,
        neg_weight: int = 1,
        pos_weight: int = 1
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
    """Fügt die neuen Signal-Wörter (NEW_TOKENS) als spezielle Tokens hinzu."""
    tokenizer.add_special_tokens({'additional_special_tokens': NEW_TOKENS})
    print(f"Neue Tokens zum Tokenizer hinzugefügt: {NEW_TOKENS}")
    return tokenizer


# ==============================================================================
# Logger-Klasse (unverändert)
# ==============================================================================
class ConsoleLogger(object):
    """Leitet print-Anweisungen in eine Log-Datei um."""

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
    """Speichert Metadaten (z.B. Epoche) in jeden Checkpoint-Ordner."""

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
            f"Aktuelle Epoche (ca.): {current_epoch}",
            "",
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
# Bereinigungsfunktion (unverändert)
# ==============================================================================
def cleanup_checkpoints(output_dir: str, best_model_path: str, save_limit: int = None):
    """
    Räumt Checkpoint-Ordner auf.
    - Strat 1 (save_limit=2): Löscht alle.
    - Strat 2/3 (save_limit=None): Löscht alle AUSSER dem besten.
    - Führt einen finalen Sweep für leere Ordner durch.
    """
    print(f"\n--- Starte Bereinigung der Checkpoints in '{output_dir}' ---")
    best_checkpoint_name = None

    if save_limit is None:
        if best_model_path:
            best_checkpoint_name = os.path.basename(best_model_path.rstrip(os.sep))
            print(f"Das beste Modell ist in: '{best_checkpoint_name}'. Dieser Ordner wird behalten.")
        else:
            print("❌ FEHLER: 'best_model_path' ist None. Breche Bereinigung ab, um Datenverlust zu verhindern.")
            return

    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        print("Keine Checkpoint-Ordner zum Bereinigen gefunden.")
        return

    print(f"Gefunden: {len(checkpoint_dirs)} Checkpoint-Ordner. Beginne Löschvorgang...")
    deleted_count = 0
    kept_count = 0

    # --- Haupt-Löschschleife ---
    for folder_path in checkpoint_dirs:
        folder_name = os.path.basename(folder_path)
        if folder_name == best_checkpoint_name:
            print(f"  ✅ '{folder_name}' wird BEHALTEN.")
            kept_count += 1
            continue
        if os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"  🗑️  '{folder_name}' gelöscht.")
                deleted_count += 1
            except FileNotFoundError:
                print(f"  ℹ️  '{folder_name}' wurde bereits an anderer Stelle gelöscht.")
            except OSError as e:
                print(f"  ❌ FEHLER beim Löschen von '{folder_name}': {e}")

    print(f"--- Bereinigung abgeschlossen. {deleted_count} Ordner entfernt, {kept_count} behalten. ---")

    # --- Finaler Sweep für leere Ordner ---
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
                except OSError as e:
                    if final_deleted == 0:
                        print(
                            f"  ℹ️  Ordner '{os.path.basename(folder_path)}' ist nicht leer oder konnte nicht entfernt werden.")
                    pass
        if final_deleted > 0:
            print(f"  {final_deleted} leere Ordner im Sweep entfernt.")
    except Exception as e:
        print(f"  ❌ FEHLER im finalen Sweep: {e}")
    print("--- Finaler Sweep abgeschlossen. ---")


# ==============================================================================
# FUNKTIONEN ZUR STRATEGIEAUSWAHL (unverändert)
# ==============================================================================

def select_training_strategy():
    """Frägt den Benutzer, welche Trainingsstrategie verwendet werden soll."""
    print("\n" + "=" * 70)
    print("--- 1. Wähle eine Trainingsstrategie ---")
    print(" [1] Optimiert (Standard):")
    print("     - Verwendet Early Stopping (patience=6).")
    print("     - Spart Speicherplatz (save_total_limit=2) -> Trainer löscht alte CPs.")
    print("     - Schnell, stoppt wenn das Modell nicht besser wird.")
    print("\n [2] Vollständig (Progressiv):")
    print("     - Deaktiviert Early Stopping. Trainiert *alle* Epochen.")
    print("     - Speichert temporär alle Checkpoints, ABER räumt am Ende")
    print("       bis auf den BESTEN auf (Spart Speicherplatz).")
    print("\n [3] Rewind and Retry (Experimentell):")
    print("     - Deaktiviert Early Stopping. Trainiert Epoche für Epoche.")
    print("     - Nach JEDER Epoche wird das BESTE BISHERIGE Modell geladen")
    print("       und das Training von dort fortgesetzt.")
    print("     - Speichert temporär alle Checkpoints, ABER räumt am Ende")
    print("       bis auf den BESTEN auf (Spart Speicherplatz).")
    print("=" * 70)
    while True:
        choice = input("Wähle Strategie [1]: ").strip() or "1"
        if choice == "1":
            print("✅ Optimierte Strategie gewählt.")
            return 1, 2, 6, 0.001
        elif choice == "2":
            print("✅ Vollständige (Progressive) Strategie gewählt.")
            return 2, None, None, 0.0
        elif choice == "3":
            print("✅ Rewind and Retry (Experimentell) Strategie gewählt.")
            return 3, None, None, 0.0
        else:
            print("Ungültige Eingabe. Bitte '1', '2' oder '3' wählen.")


def select_optimization_metric(priority_order_list):
    """Frägt den Benutzer, auf welche Metrik das Modell optimiert werden soll."""
    print("\n" + "=" * 70)
    print("--- 2. Wähle die Optimierungs-Metrik ---")
    print("   (Das Modell wird basierend auf dieser Metrik als 'bestes' ausgewählt)")
    options = [
        ("f1_critical_high_avg", "(Durchschnitt von Critical/High) - EMPFOHLEN"),
        ("f1_weighted", "(Gesamtdurchschnitt aller Klassen)"),
        ("accuracy", "(Genauigkeit - Nicht empfohlen bei Imbalance)")
    ]
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
    """Frägt den Benutzer, wie das Evaluierungs-Set erstellt werden soll."""
    print("\n" + "=" * 70)
    print("--- 3. Wähle die Evaluierungs-Strategie ---")
    print("\n [1] Prozentualer Split (Stratifiziert): (Default: 10%)")
    print("\n [2] Absoluter Split (Stratifiziert): (z.B. 500 Tickets)")
    print("\n [3] Balancierter Split (Feste Anzahl pro Klasse): (z.B. 50 pro Klasse)")
    print("\n [4] Separate CSV-Datei verwenden:")
    print("=" * 70)
    while True:
        choice = input("Wähle Evaluierungs-Modus [1]: ").strip() or "1"
        if choice == "1":
            while True:
                val_str = input(
                    "  Welcher Anteil (Prozent als Dezimalzahl) für die Evaluierung? [0.1]: ").strip() or "0.1"
                try:
                    val_float = float(val_str)
                    if 0.01 <= val_float < 1.0:
                        print(f"✅ Modus 'Prozentual' gewählt ({val_float * 100:.0f}%).")
                        return "percentage", val_float
                except ValueError:
                    print("  Ungültige Eingabe. Bitte eine Zahl eingeben (z.B. 0.1 für 10%).")
        elif choice == "2":
            while True:
                val_str = input("  Wieviele Tickets *insgesamt* für die Evaluierung? [500]: ").strip() or "500"
                try:
                    val_int = int(val_str)
                    if val_int > 0:
                        print(f"✅ Modus 'Absolut' gewählt ({val_int} Tickets).")
                        return "absolute", val_int
                except ValueError:
                    print("  Ungültige Eingabe. Bitte eine ganze Zahl eingeben.")
        elif choice == "3":
            while True:
                val_str = input(f"  Wieviele Tickets *pro Klasse* für die Evaluierung? [50]: ").strip() or "50"
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
        else:
            print("Ungültige Eingabe. Bitte '1', '2', '3' oder '4' wählen.")


# ==============================================================================
# FUNKTION: BALANCIERTER SPLIT (unverändert)
# ==============================================================================

def create_balanced_split(full_dataset: Dataset, label_column: str, n_per_class: int, seed: int = 42) -> (Dataset,
                                                                                                          Dataset):
    """Erstellt manuell einen balancierten Split (Modus 3)."""
    print(f"Starte manuellen 'balancierten' Split (max. {n_per_class} pro Klasse) für Spalte '{label_column}'...")
    try:
        print("  Konvertiere zu Pandas für Split-Logik...")
        df = full_dataset.to_pandas()
    except Exception as e:
        print(f"  FEHLER bei Konvertierung zu Pandas: {e}")
        return None, None
    unique_labels = df[label_column].unique()
    print(f"  Gefundene Prioritäten: {list(unique_labels)}")
    eval_df_list = []
    train_df_list = []
    np.random.seed(seed)
    for label in unique_labels:
        class_df = df[df[label_column] == label]
        n_available = len(class_df)
        n_to_sample = min(n_available, n_per_class)
        if n_available == 0:
            continue
        elif n_available < n_per_class:
            print(
                f"  WARNUNG: Klasse '{label}' hat nur {n_available} Tickets (< {n_per_class}). Verwende alle für Eval.")
        elif n_available == n_to_sample:
            print(
                f"  WARNUNG: Klasse '{label}' hat exakt {n_available} Tickets. Alle für Eval, 0 für Training dieser Klasse.")
        else:
            print(f"  Klasse '{label}': {n_to_sample} von {n_available} Tickets für Eval ausgewählt.")
        eval_sample = class_df.sample(n=n_to_sample, random_state=seed)
        eval_df_list.append(eval_sample)
        train_sample = class_df.drop(eval_sample.index)
        train_df_list.append(train_sample)
    if not eval_df_list:
        print("FEHLER: Evaluierungsset ist leer.")
        return None, None
    train_df = pd.concat(train_df_list).sample(frac=1, random_state=seed).reset_index(drop=True)
    eval_df = pd.concat(eval_df_list).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"  Split abgeschlossen: {len(train_df)} Trainings-Tickets, {len(eval_df)} Evaluierungs-Tickets.")
    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(eval_df)
    return train_ds, eval_ds


# ==============================================================================
# HAUPT-TRAININGSFUNKTION (main) (MODIFIZIERT)
# ==============================================================================

def main():
    # --- 0. Logging & Ausgabeordner (MODIFIZIERT) ---
    base_log_dir = "logs_multilingual_stratified_512"

    # --- NEU: Ausgabeordner definieren (Request 2 & 3) ---
    # Schlage Standard-Ausgabeordner mit Datum vor
    timestamp_date = datetime.now().strftime("%Y-%m-%d")
    default_output_dir = f"./ergebnisse_multilingual_stratified_eval_modified_{timestamp_date}"

    print("\n" + "=" * 70)
    print("--- 0. Wähle den Ausgabeordner ---")
    print(f"Standardmäßig wird: '{default_output_dir}' vorgeschlagen.")
    output_dir_input = input("Enter (zum Bestätigen) oder gib einen anderen Pfad an: ").strip()

    output_dir = output_dir_input or default_output_dir
    print(f"✅ Ausgabeordner gesetzt auf: {output_dir}")
    # Stelle sicher, dass der Ordner existiert (wichtig für 'resume' Check)
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)

    # Logging-Konfiguration (nutzt detaillierten Timestamp)
    if os.path.isfile(base_log_dir):
        print(f"⚠️  Warnung: Datei '{base_log_dir}' blockiert Log-Verzeichnis.")
        backup_name = f"logs_multilingual_als_datei_{int(time.time())}.txt"
        print(f"✅ Datei wird umbenannt in '{backup_name}'.")
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
        print(f"Starte den HYBRID-Trainingsprozess...")
        print(f"Alle Konsolenausgaben werden in '{console_log_path}' gespeichert.")
        print("=" * 70)

        # === 1. Strategie-Auswahl (Benutzer-Input) ===
        strategy_id, strategy_save_limit, strategy_patience, strategy_threshold = select_training_strategy()
        chosen_metric = select_optimization_metric(PRIORITY_ORDER)
        eval_mode, eval_value = select_evaluation_strategy(PRIORITY_ORDER)
        print("=" * 70)

        # === 2. Vokabular-Management ===
        generate_vocab_files_if_needed()
        neg_vocab, pos_vocab, sla_vocab = load_vocab_from_csvs()

        # === 3. GPU-Prüfung und Setup ===
        is_gpu_available = torch.cuda.is_available()
        if is_gpu_available:
            print("✅ GPU gefunden! Das Training wird auf der GPU ausgeführt. 🚀")
        else:
            print("⚠️ Keine GPU gefunden. Das Training wird auf der CPU ausgeführt (deutlich langsamer).")
        print(f"➡️  Aktuelles Arbeitsverzeichnis: {os.getcwd()}")

        # --- NEU: Logik für bestehenden Ausgabeordner (Request 4) ---
        overwrite_output = False
        resume_from_checkpoint = False  # Neuer Flag für den Trainer

        # Prüfe, ob der Ordner existiert UND nicht leer ist
        if os.path.isdir(output_dir) and os.listdir(output_dir):
            print(f"⚠️  Es sind bereits Daten im Ausgabeverzeichnis '{output_dir}' vorhanden.")
            print("   [1] Training fortsetzen (weitere Epochen trainieren)")
            print("   [2] ALLES ÜBERSCHREIBEN (bisheriges Modell löschen)")
            print("   [3] Abbrechen")

            while True:
                choice = input("Wähle eine Option [1, 2, 3]: ").strip()

                if choice == '1':
                    print("✅ Training wird fortgesetzt. Lade letzten Checkpoint...")
                    overwrite_output = False  # Nicht überschreiben
                    resume_from_checkpoint = True  # Sage dem Trainer, er soll fortsetzen
                    break
                elif choice == '2':
                    print("✅ Vorhandene Daten werden überschrieben.")
                    overwrite_output = True  # Sage dem Trainer, er soll überschreiben
                    resume_from_checkpoint = False
                    break
                elif choice == '3':
                    print("❌ Vorgang vom Benutzer abgebrochen.")
                    sys.exit()
                else:
                    print("Ungültige Eingabe. Bitte '1', '2' oder '3' eingeben.")
        else:
            # Ordner ist leer (oder wurde gerade erst erstellt)
            overwrite_output = False
            resume_from_checkpoint = False
        # --- Ende der neuen Logik ---

        print(f"Logs für diesen Durchlauf werden in '{run_log_dir}' gespeichert.")

        # === 4. Dataset laden und aufteilen (Dynamisch) ===
        dataset = None
        try:
            if eval_mode == "external_csv":
                print(f"Lade Trainings-Dataset von: {DATA_FILE}")
                train_ds = load_dataset('csv', data_files=DATA_FILE, split="train")
                eval_file_path = eval_value
                print(f"Lade externes Evaluierungs-Dataset von: {eval_file_path}")
                eval_ds = load_dataset('csv', data_files=eval_file_path, split="train")
                dataset = DatasetDict({"train": train_ds, "validation": eval_ds})
            else:
                print(f"Lade gesamtes Dataset von: {DATA_FILE}")
                full_dataset = load_dataset('csv', data_files=DATA_FILE, split="train")
                if eval_mode == "percentage":
                    print(f"Erstelle {eval_value * 100:.0f}% prozentualen, stratifizierten Split...")
                    split = full_dataset.train_test_split(
                        test_size=eval_value, seed=42, stratify_by_column=FIELD_LABEL
                    )
                    dataset = DatasetDict({"train": split["train"], "validation": split["test"]})
                elif eval_mode == "absolute":
                    print(f"Erstelle {eval_value} absoluten, stratifizierten Split...")
                    split_size = eval_value
                    if eval_value > len(full_dataset):
                        print(
                            f"WARNUNG: Angeforderte Größe ({eval_value}) > Dataset ({len(full_dataset)}). Nutze 10% Fallback.")
                        split_size = 0.1
                    split = full_dataset.train_test_split(
                        test_size=split_size, seed=42, stratify_by_column=FIELD_LABEL
                    )
                    dataset = DatasetDict({"train": split["train"], "validation": split["test"]})
                elif eval_mode == "balanced":
                    train_ds, eval_ds = create_balanced_split(
                        full_dataset, FIELD_LABEL, eval_value, seed=42
                    )
                    if train_ds is None or eval_ds is None:
                        print("❌ FEHLER: Balancierter Split fehlgeschlagen.")
                        sys.exit()
                    dataset = DatasetDict({"train": train_ds, "validation": eval_ds})
        except FileNotFoundError as e:
            print(f"❌ FEHLER: Datei nicht gefunden: {e}")
            sys.exit()
        except Exception as e:
            print(f"❌ FEHLER beim Laden oder Aufteilen der Daten: {e}")
            import traceback
            traceback.print_exc()
            sys.exit()
        print(
            f"✅ Dataset-Aufteilung abgeschlossen: {len(dataset['train'])} Trainings-, {len(dataset['validation'])} Validierungs-Beispiele.")

        # === 5. Labels (Prioritäten) umwandeln ===
        print("Wandle die 'priority'-Spalte in Klassen-Labels mit fester Reihenfolge um...")
        class_label_feature = ClassLabel(names=PRIORITY_ORDER)
        num_unique_labels = len(PRIORITY_ORDER)
        try:
            dataset = dataset.cast_column(FIELD_LABEL, class_label_feature)
            print(f"✅ '{FIELD_LABEL}'-Spalte erfolgreich in {num_unique_labels} Labels (ClassLabel-Typ) umgewandelt.")
        except (ValueError, KeyError) as e:
            print(f"❌ FEHLER beim Umwandeln der '{FIELD_LABEL}'-Spalte: {e}")
            print(f"Stelle sicher, dass die Spalte nur Werte enthält aus: {PRIORITY_ORDER}")
            sys.exit()

        # === 6. Modell und Tokenizer laden ===
        print("Lade das Basis-Modell und den Tokenizer...")
        modell_name = "distilbert/distilbert-base-multilingual-cased"
        try:
            tokenizer = AutoTokenizer.from_pretrained(modell_name)
            model = AutoModelForSequenceClassification.from_pretrained(modell_name, num_labels=num_unique_labels)
        except OSError:
            print(f"❌ FEHLER: Modell '{modell_name}' nicht gefunden. (Internetverbindung?)")
            sys.exit()

        tokenizer = add_new_tokens_to_tokenizer(tokenizer)
        model.resize_token_embeddings(len(tokenizer))
        print("✅ Tokenizer und Modell um neue Signal-Tokens erweitert.")

        # === 7. Tokenize-Funktion definieren und anwenden ===
        def tokenize_and_enrich_function(examples):
            raw_texts = [str(body) + " " + str(subject) for body, subject in
                         zip(examples[FIELD_BODY], examples[FIELD_SUBJECT])]
            enriched_texts = [
                preprocess_with_vocab(
                    text, neg_vocab, pos_vocab, sla_vocab,
                    sla_weight=KEYWORD_WEIGHTS["sla_weight"],
                    neg_weight=KEYWORD_WEIGHTS["neg_weight"],
                    pos_weight=KEYWORD_WEIGHTS["pos_weight"]
                )
                for text in raw_texts
            ]
            return tokenizer(enriched_texts, padding="max_length", truncation=True, max_length=512)

        print("Starte Anreicherung und Tokenisierung des Datasets (max_length=512)...")
        tokenized_datasets = dataset.map(tokenize_and_enrich_function, batched=True)

        # === 8. Finale Vorbereitung (Spalten aufräumen) ===
        print(f"Benenne die '{FIELD_LABEL}'-Spalte in 'labels' um...")
        tokenized_datasets = tokenized_datasets.rename_column(FIELD_LABEL, "labels")
        try:
            columns_to_remove = [FIELD_SUBJECT, FIELD_BODY] + FIELDS_TO_REMOVE
            final_columns_to_remove = [col for col in columns_to_remove if
                                       col in tokenized_datasets["train"].column_names]
            print(f"Entferne nicht benötigte Spalten: {final_columns_to_remove}")
            tokenized_datasets = tokenized_datasets.remove_columns(final_columns_to_remove)
        except ValueError:
            print("Hinweis: Einige Spalten zum Entfernen wurden nicht gefunden, fahre fort.")
            pass

        # === 9. Trainings-Argumente definieren (Dynamisch) ===
        training_args = TrainingArguments(
            output_dir=output_dir,  # MODIFIZIERT: Dynamischer Pfad
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=chosen_metric,
            save_total_limit=strategy_save_limit,
            greater_is_better=True,
            num_train_epochs=30,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_dir=run_log_dir,
            logging_strategy="epoch",
            overwrite_output_dir=overwrite_output,  # MODIFIZIERT: Dynamisch
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
            print(f"Aktiviere Early Stopping (Strategie 1) mit Geduld={strategy_patience}...")
            callbacks_list.append(
                EarlyStoppingCallback(
                    early_stopping_patience=strategy_patience,
                    early_stopping_threshold=strategy_threshold
                )
            )
        else:
            print(f"Strategie {strategy_id} ('Vollständig' or 'Rewind'): Early Stopping ist DEAKTIViert.")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            compute_metrics=compute_metrics,
            callbacks=callbacks_list
        )

        # ==================================================================
        # === SCHRITT 11: Training starten (MODIFIZIERT) ===
        # ==================================================================

        train_result = None

        if strategy_id == 3:  # Rewind and Retry
            print(f"\n--- Starte 'Rewind and Retry' Training (optimiert auf '{chosen_metric}') ---")
            current_best_checkpoint = None
            num_epochs = int(training_args.num_train_epochs)

            # NEU: Übertrage den globalen Resume-Flag auf den ersten Loop
            if resume_from_checkpoint:
                print("Info: 'Fortsetzen' ist aktiv. Der Trainer wird den letzten Checkpoint suchen.")
                # 'True' sagt trainer.train(), den letzten Checkpoint aus dem output_dir zu finden
                current_best_checkpoint = True

            for epoch in range(num_epochs):
                print(f"\n" + "=" * 80)
                print(f"--- Starte Epoche {epoch + 1} / {num_epochs} (Rewind-Modus) ---")
                trainer.args.max_steps = (epoch + 1) * steps_per_epoch_for_callback

                # Beim ersten Loop ist current_best_checkpoint (None oder True)
                # Bei späteren Loops ist es der Pfad (z.B. ".../checkpoint-1500")
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
            # Übergebe den globalen resume_from_checkpoint Flag (True oder False)
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # === Schritt 12: Modell explizit speichern ===
        print(f"Speichere das finale *beste* Modell (basierend auf '{chosen_metric}')...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"\n🎉 Training erfolgreich abgeschlossen! Das beste Modell wurde im Ordner '{output_dir}' gespeichert.")

        # === SCHRITT 13: Redundante Checkpoints bereinigen ===
        best_model_path = trainer.state.best_model_checkpoint
        if best_model_path is None and train_result is not None:
            best_model_path = train_result.best_model_checkpoint  # Fallback

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