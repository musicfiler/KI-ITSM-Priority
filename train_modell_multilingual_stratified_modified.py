# train_modell_multilingual_stratified.py

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
from datasets import load_dataset, ClassLabel, Features
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- Globale Konfiguration ---
BASE_DIR = "trainingsdaten"
DATA_FILE = os.path.join(BASE_DIR, "5_prio_stufen/dataset-tickets-german_normalized_50_5_2.csv")
NEG_CSV = os.path.join(BASE_DIR, "neg_arguments_multilingual.csv")
POS_CSV = os.path.join(BASE_DIR, "pos_arguments_multilingual.csv")
SLA_CSV = os.path.join(BASE_DIR, "sla_arguments_multilingual.csv")

# Diese Reihenfolge ist jetzt KRITISCH für die Per-Klassen-Metriken
PRIORITY_ORDER = ["critical", "high", "medium", "low", "very_low"]
HIGH_PRIO_CLASSES_STR = ["critical", "high"]
TOP_N_TERMS = 75
MIN_DF = 5
NEW_TOKENS = ['KEY_CORE_APP', 'KEY_CRITICAL', 'KEY_REQUEST', 'KEY_NORMAL']

# Versuche, deutsche Stopwörter zu laden
try:
    from stop_words import get_stop_words

    GERMAN_STOPWORDS = get_stop_words('de')
except ImportError:
    print("WARNUNG: Paket 'stop-words' nicht gefunden. (pip install stop-words)")
    print("Fahre ohne deutsche Stopwörter fort.")
    GERMAN_STOPWORDS = []


# ==============================================================================
# COMPUTE_METRICS FUNKTION
# ==============================================================================

def compute_metrics(pred):
    """
    Berechnet Metriken für die Evaluierung.
    Berechnet eine spezielle Metrik 'f1_critical_high_avg'
    als Durchschnitt der F1-Scores der beiden wichtigsten Klassen.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # 1. Berechne die gewichteten Gesamtmetriken (fürs Logging)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)

    # 2. NEU: Berechne die Per-Klasse-Metriken (average=None)
    class_indices = list(range(len(PRIORITY_ORDER)))
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0,
        labels=class_indices
    )

    # 3. Erstelle das finale Metrik-Wörterbuch für den Trainer
    metrics = {
        'accuracy': acc,
        'f1_weighted': weighted_f1,
        'precision_weighted': weighted_precision,
        'recall_weighted': weighted_recall
    }

    # 4. Füge Per-Klasse-Metriken hinzu und gib sie in der Konsole aus
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

    # 5. Greife auf die spezifischen F1-Scores zu
    f1_critical = f1_per_class[0]  # PRIORITY_ORDER[0]
    f1_high = f1_per_class[1]  # PRIORITY_ORDER[1]

    # 6. Berechne die neue Spezial-Metrik
    f1_crit_high_avg = (f1_critical + f1_high) / 2.0

    # 7. Füge die neue Metrik zum Dictionary hinzu
    metrics['f1_critical_high_avg'] = f1_crit_high_avg

    print(f"  [SPEZIAL-METRIK]: f1_critical_high_avg = {f1_crit_high_avg:<7.2%}")
    print("=====================================================================")

    return metrics


# ==============================================================================
# HILFSFUNKTIONEN (unverändert)
# ==============================================================================

def generate_vocab_files_if_needed():
    """Prüft, ob die Vokabular-CSVs existieren... (Code unverändert)"""
    os.makedirs(BASE_DIR, exist_ok=True)
    if os.path.exists(NEG_CSV) and os.path.exists(POS_CSV) and os.path.exists(SLA_CSV):
        print("✅ Phase 1: Vokabular-Dateien gefunden. Überspringe automatische Generierung.")
        return
    # ... (Rest der Funktion unverändert) ...
    print("⚠️ Phase 1: Vokabular-Dateien nicht gefunden. Starte automatische Extraktion...")
    if not os.path.exists(DATA_FILE):
        print(f"❌ FEHLER: Trainingsdatensatz {DATA_FILE} nicht gefunden.")
        sys.exit()
    try:
        df = pd.read_csv(DATA_FILE)
        df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
        df = df.dropna(subset=['text', 'priority'])
    except KeyError as e:
        print(f"❌ FEHLER: Die Spalte {e} wurde in {DATA_FILE} nicht gefunden.")
        sys.exit()
    print(f"Analysiere {len(df)} Tickets für Vokabular-Extraktion...")
    df_high_prio = df[df['priority'].isin(HIGH_PRIO_CLASSES_STR)]
    df_low_prio = df[~df['priority'].isin(HIGH_PRIO_CLASSES_STR)]
    if df_high_prio.empty or df_low_prio.empty:
        print(f"❌ FEHLER: Konnte keine Tickets für hohe Priorität (Werte: {HIGH_PRIO_CLASSES_STR}) finden.")
        sys.exit()
    vectorizer = TfidfVectorizer(
        stop_words=GERMAN_STOPWORDS,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=MIN_DF
    )
    vectorizer.fit(df['text'])
    tfidf_high = vectorizer.transform(df_high_prio['text'])
    tfidf_low = vectorizer.transform(df_low_prio['text'])
    mean_tfidf_high = np.asarray(tfidf_high.mean(axis=0)).ravel()
    mean_tfidf_low = np.asarray(tfidf_low.mean(axis=0)).ravel()
    feature_names = np.array(vectorizer.get_feature_names_out())
    score_diff = mean_tfidf_high - mean_tfidf_low
    results_df = pd.DataFrame({'term': feature_names, 'score_diff': score_diff})
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
    """Lädt die Vokabularlisten aus den CSV-Dateien. (Code unverändert)"""
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
    """Reichert einen Text mit speziellen Signal-Wörtern (KEY_...) an. (Code unverändert)"""
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
    """Fügt die neuen Signal-Wörter als spezielle Tokens hinzu. (Code unverändert)"""
    tokenizer.add_special_tokens({'additional_special_tokens': NEW_TOKENS})
    print(f"Neue Tokens zum Tokenizer hinzugefügt: {NEW_TOKENS}")
    return tokenizer


# ==============================================================================
# Logger-Klasse (unverändert)
# ==============================================================================
class ConsoleLogger(object):
    """
    Leitet print-Anweisungen (stdout) und Fehler (stderr)
    sowohl an die Konsole als auch in eine Log-Datei um.
    """

    def __init__(self, log_path: str):
        self.terminal_stdout = sys.stdout
        self.terminal_stderr = sys.stderr
        self.log_file = io.open(log_path, 'w', encoding='utf-8')

    def hook_stdout(self):
        sys.stdout = self

    def hook_stderr(self):
        sys.stderr = self

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
    """
    Ein Custom Callback, der eine Textdatei mit den Trainings-Setup-Metadaten
    in jeden Checkpoint-Ordner speichert.
    """

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
# Bereinigungsfunktion (MODIFIZIERT)
# ==============================================================================
def cleanup_checkpoints(output_dir: str, best_model_path: str, save_limit: int = None):
    """
    Löscht alle 'checkpoint-*' Unterordner, außer dem besten.
    Wenn save_limit gesetzt ist (z.B. 2), löscht es einfach alle.
    """
    print(f"\n--- Starte Bereinigung der Checkpoints in '{output_dir}' ---")

    # Standard-Verhalten (Strategie 1 & 2):
    # 'load_best_model_at_end' kopiert das beste Modell bereits in den Root.
    # Wir können alle Checkpoints löschen.
    best_checkpoint_name = None

    # 'Rewind and Retry' (Strategie 3):
    # Das beste Modell ist *nur* im Checkpoint-Ordner. Wir müssen ihn behalten.
    if save_limit is None:  # Nur im 'Vollständig' oder 'Rewind' Modus
        if best_model_path:
            best_checkpoint_name = os.path.basename(best_model_path.rstrip(os.sep))
            print(f"Das beste Modell ist in: '{best_checkpoint_name}'. Dieser Ordner wird behalten.")
        else:
            print("❌ FEHLER: 'best_model_path' ist None. Breche Bereinigung ab, um Datenverlust zu verhindern.")
            return

    # Finde alle Ordner, die mit "checkpoint-" beginnen
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))

    if not checkpoint_dirs:
        print("Keine Checkpoint-Ordner zum Bereinigen gefunden.")
        return

    print(f"Gefunden: {len(checkpoint_dirs)} Checkpoint-Ordner. Beginne Löschvorgang...")

    deleted_count = 0
    kept_count = 0
    for folder_path in checkpoint_dirs:
        folder_name = os.path.basename(folder_path)

        # MODIFIZIERTE LOGIK: Lösche nur, wenn es NICHT der beste Ordner ist
        if folder_name == best_checkpoint_name:
            print(f"  ✅ '{folder_name}' wird BEHALTEN.")
            kept_count += 1
            continue

        # Andernfalls, lösche den Ordner
        if os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"  🗑️  '{folder_name}' gelöscht.")
                deleted_count += 1
            except OSError as e:
                print(f"  ❌ FEHLER beim Löschen von '{folder_name}': {e}")
                print("     (Möglicherweise blockiert OneDrive oder ein anderer Prozess die Datei.)")

    print(f"--- Bereinigung abgeschlossen. {deleted_count} Ordner entfernt, {kept_count} behalten. ---")


# ==============================================================================
# NEUE FUNKTIONEN ZUR STRATEGIEAUSWAHL
# ==============================================================================

def select_training_strategy():
    """
    Frägt den Benutzer, welche Trainingsstrategie verwendet werden soll.

    Gibt (strategie_id, save_limit, patience, threshold) zurück.
    """
    print("\n--- 1. Wähle eine Trainingsstrategie ---")
    print(" [1] Optimiert (Standard):")
    print("     - Verwendet Early Stopping (patience=6).")
    print("     - Spart Speicherplatz (save_total_limit=2).")
    print("     - Schnell, stoppt wenn das Modell nicht besser wird.")
    print("\n [2] Vollständig (Progressiv):")
    print("     - Deaktiviert Early Stopping. Trainiert *alle* Epochen.")
    print("     - Speichert *alle* Checkpoints (Hoher Speicherbedarf!).")
    print("     - Wählt am Ende das beste Modell aus allen Epochen.")
    print("\n [3] Rewind and Retry (Experimentell):")
    print("     - Deaktiviert Early Stopping. Trainiert Epoche für Epoche.")
    print("     - Nach JEDER Epoche wird das BESTE BISHERIGE Modell geladen")
    print("       und das Training von dort fortgesetzt.")
    print("     - Speichert *alle* Checkpoints (Hoher Speicherbedarf!).")

    while True:
        choice = input("Wähle Strategie [1]: ").strip() or "1"
        if choice == "1":
            print("✅ Optimierte Strategie gewählt.")
            # (id=1, save_limit=2, patience=6, threshold=0.001)
            return 1, 2, 6, 0.001
        elif choice == "2":
            print("✅ Vollständige (Progressive) Strategie gewählt.")
            # (id=2, save_limit=None, patience=None (kein Early Stopping))
            return 2, None, None, 0.0
        elif choice == "3":
            print("✅ Rewind and Retry (Experimentell) Strategie gewählt.")
            # (id=3, save_limit=None, patience=None (kein Early Stopping))
            return 3, None, None, 0.0
        else:
            print("Ungültige Eingabe. Bitte '1', '2' oder '3' wählen.")


def select_optimization_metric(priority_order_list):
    """
    Frägt den Benutzer, auf welche Metrik das Modell optimiert werden soll.
    """
    print("\n--- 2. Wähle die Optimierungs-Metrik ---")
    print("   (Das Modell wird basierend auf dieser Metrik als 'bestes' ausgewählt)")

    # Metriken, die immer verfügbar sind
    options = [
        ("f1_critical_high_avg", "(Durchschnitt von Critical/High) - EMPFOHLEN"),
        ("f1_weighted", "(Gesamtdurchschnitt aller Klassen)"),
        ("accuracy", "(Genauigkeit - Nicht empfohlen bei Imbalance)")
    ]

    # Füge die F1-Scores für jede einzelne Klasse hinzu
    for label in priority_order_list:
        options.append((f"f1_{label}", f"(Nur F1-Score für '{label}')"))

    # Zeige die Optionen an
    for i, (metric_name, description) in enumerate(options):
        print(f" [{i + 1}] {metric_name.ljust(24)} {description}")

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


# ==============================================================================
# HAUPT-TRAININGSFUNKTION (main)
# ==============================================================================

def main():
    # --- Logging wird zuerst konfiguriert ---
    base_log_dir = "logs_multilingual_stratified_512"
    output_dir = "./ergebnisse_multilingual_stratified_512"

    if os.path.isfile(base_log_dir):
        print(f"⚠️  Warnung: Datei '{base_log_dir}' blockiert Log-Verzeichnis.")
        backup_name = f"logs_multilingual_als_datei_{int(time.time())}.txt"
        print(f"✅ Datei wird umbenannt in '{backup_name}'.")
        os.rename(base_log_dir, backup_name)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_name = "train_modell_multilingual_stratified"
    run_log_dir = os.path.join(base_log_dir, f"{timestamp}_{script_name}")
    os.makedirs(run_log_dir, exist_ok=True)
    console_log_path = os.path.join(run_log_dir, f"{timestamp}_{script_name}_console.log")

    logger = ConsoleLogger(console_log_path)
    logger.hook_stdout()
    logger.hook_stderr()

    # --- try...finally-Block ---
    try:
        print(f"Starte den HYBRID-Trainingsprozess (OPTIMIERT & STRATIFIZIERT)...")
        print(f"Alle Konsolenausgaben werden in '{console_log_path}' gespeichert.")

        # ==================================================================
        # === NEU: Strategie-Auswahl vor dem Setup ===
        # ==================================================================
        strategy_id, strategy_save_limit, strategy_patience, strategy_threshold = select_training_strategy()
        chosen_metric = select_optimization_metric(PRIORITY_ORDER)

        # === Schritt 1: Vokabular-Management ===
        generate_vocab_files_if_needed()
        neg_vocab, pos_vocab, sla_vocab = load_vocab_from_csvs()

        # === Schritt 2: Konfiguration, Diagnose und Geräte-Prüfung ===
        is_gpu_available = torch.cuda.is_available()
        if is_gpu_available:
            print("✅ GPU gefunden! Das Training wird auf der GPU ausgeführt. 🚀")
        else:
            print("⚠️ Keine GPU gefunden. Das Training wird auf der CPU ausgeführt (deutlich langsamer).")

        print(f"➡️  Aktuelles Arbeitsverzeichnis: {os.getcwd()}")

        overwrite_output = False
        if os.path.isdir(output_dir) and os.listdir(output_dir):
            print(f"⚠️  Es sind bereits Daten im Ausgabeverzeichnis '{output_dir}' vorhanden.")
            while True:
                choice = input("Möchten Sie die vorhandenen Ergebnisse überschreiben? (j/n): ").lower()
                print(f"Benutzereingabe: {choice}")
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

        print(f"Logs für diesen Durchlauf werden in '{run_log_dir}' gespeichert.")

        # === Schritt 3: Dataset laden ===
        print("Lade das Dataset...")
        try:
            dataset = load_dataset('csv', data_files=DATA_FILE)
        except FileNotFoundError:
            print(f"❌ FEHLER: {DATA_FILE} nicht gefunden.")
            sys.exit()

        # === SCHRITT 4: Label-Spalte vorbereiten und casten ===
        print("Wandle die 'priority'-Spalte in Klassen-Labels mit fester Reihenfolge um...")
        class_label_feature = ClassLabel(names=PRIORITY_ORDER)
        num_unique_labels = len(PRIORITY_ORDER)
        try:
            dataset = dataset.cast_column("priority", class_label_feature)
            print(f"✅ 'priority'-Spalte erfolgreich in {num_unique_labels} Labels (ClassLabel-Typ) umgewandelt.")
        except ValueError as e:
            print(f"❌ FEHLER beim Umwandeln der 'priority'-Spalte: {e}")
            print("Stelle sicher, dass die 'priority'-Spalte in deiner CSV nur folgende Werte enthält:")
            print(PRIORITY_ORDER)
            sys.exit()

        # === SCHRITT 4.1: Stratified Split ===
        print("Teile das Dataset in STRATIFIZIERTE Trainings- und Validierungs-Sets auf (90/10 Split)...")
        train_test_split = dataset["train"].train_test_split(
            test_size=0.1,
            seed=42,
            stratify_by_column="priority"
        )
        dataset["train"] = train_test_split["train"]
        dataset["validation"] = train_test_split["test"]
        print(
            f"✅ Stratifizierte Aufteilung erfolgt: {len(dataset['train'])} Trainings-, {len(dataset['validation'])} Validierungs-Beispiele.")

        # === Schritt 5: Modell und Tokenizer laden und erweitern ===
        print("Lade das Basis-Modell und den Tokenizer...")
        modell_name = "distilbert/distilbert-base-multilingual-cased"
        try:
            tokenizer = AutoTokenizer.from_pretrained(modell_name)
            model = AutoModelForSequenceClassification.from_pretrained(modell_name, num_labels=num_unique_labels)
        except OSError:
            print(f"❌ FEHLER: Modell '{modell_name}' nicht gefunden.")
            print("Stelle sicher, dass du eine Internetverbindung hast und der Modellname korrekt ist.")
            sys.exit()

        tokenizer = add_new_tokens_to_tokenizer(tokenizer)
        model.resize_token_embeddings(len(tokenizer))
        print("✅ Tokenizer und Modell um neue Signal-Tokens erweitert.")

        # === Schritt 6: Tokenize-Funktion definieren und anwenden ===
        def tokenize_and_enrich_function(examples):
            raw_texts = [str(body) + " " + str(subject) for body, subject in
                         zip(examples["body"], examples["subject"])]
            enriched_texts = [
                preprocess_with_vocab(
                    text, neg_vocab, pos_vocab, sla_vocab,
                    sla_weight=5,
                    neg_weight=4,
                    pos_weight=1
                )
                for text in raw_texts
            ]
            return tokenizer(enriched_texts, padding="max_length", truncation=True, max_length=512)

        print("Starte Anreicherung und Tokenisierung des Datasets (max_length=512)...")
        tokenized_datasets = dataset.map(tokenize_and_enrich_function, batched=True)

        # === Schritt 7: Finale Vorbereitung ===
        print("Benenne die 'priority'-Spalte in 'labels' um...")
        tokenized_datasets = tokenized_datasets.rename_column("priority", "labels")
        try:
            tokenized_datasets = tokenized_datasets.remove_columns(['subject', 'body', 'queue', 'language'])
        except ValueError:
            print("Hinweis: Einige Spalten zum Entfernen wurden nicht gefunden, fahre fort.")
            pass

        # === Schritt 8: Trainings-Argumente definieren (Dynamisch) ===
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,  # Wichtig für alle Strategien

            # MODIFIZIERT: Basierend auf Benutzerauswahl
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
            overwrite_output_dir=overwrite_output,
            report_to="none",
            fp16=is_gpu_available,
        )

        # === Schritt 8.1: Metadaten für den Callback berechnen ===
        num_train_tickets_for_callback = len(tokenized_datasets["train"])
        batch_size_for_callback = training_args.per_device_train_batch_size
        steps_per_epoch_for_callback = (num_train_tickets_for_callback // batch_size_for_callback) + \
                                       (1 if num_train_tickets_for_callback % batch_size_for_callback > 0 else 0)

        # === Schritt 9: Trainer initialisieren (Dynamisch) ===

        # Stelle die Liste der Callbacks zusammen
        callbacks_list = [
            CheckpointMetadataCallback(
                num_train_tickets=num_train_tickets_for_callback,
                steps_per_epoch=steps_per_epoch_for_callback
            )
        ]

        # Füge EarlyStopping nur hinzu, wenn Strategie 1 (Optimiert) gewählt wurde
        if strategy_id == 1:
            print(
                f"Aktiviere Early Stopping (Strategie 1) mit Geduld={strategy_patience} und Schwellenwert={strategy_threshold}")
            callbacks_list.append(
                EarlyStoppingCallback(
                    early_stopping_patience=strategy_patience,
                    early_stopping_threshold=strategy_threshold
                )
            )
        else:
            print(f"Strategie {strategy_id} ('Vollständig' or 'Rewind'): Early Stopping ist DEAKTIVIERT.")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            compute_metrics=compute_metrics,
            callbacks=callbacks_list
        )

        # ==================================================================
        # === SCHRITT 10 (MODIFIZIERT): Training basierend auf Strategie starten ===
        # ==================================================================

        train_result = None

        if strategy_id == 3:  # Rewind and Retry
            print(f"\n--- Starte 'Rewind and Retry' Training (optimiert auf '{chosen_metric}') ---")
            current_best_checkpoint = None
            num_epochs = int(training_args.num_train_epochs)

            for epoch in range(num_epochs):
                print(f"\n" + "=" * 80)
                print(f"--- Starte Epoche {epoch + 1} / {num_epochs} (Rewind-Modus) ---")

                # Setze das neue Ziel: Trainiere genau EINE weitere Epoche
                # WICHTIG: `max_steps` ist die *absolute* Anzahl der Schritte seit Beginn
                trainer.args.max_steps = (epoch + 1) * steps_per_epoch_for_callback

                print(f"Max steps gesetzt auf: {trainer.args.max_steps}")
                print(f"Starte Training von: {current_best_checkpoint or 'Anfang (Epoche 0)'}")

                # Starte das Training. Es wird *immer* vom besten Punkt fortgesetzt
                # und trainiert, bis es `max_steps` erreicht, dann stoppt es.
                train_result = trainer.train(resume_from_checkpoint=current_best_checkpoint)

                # Das Wichtigste: Finde das neue beste Modell und speichere es für den nächsten Loop
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
            train_result = trainer.train()

        # === Schritt 11: Modell explizit speichern ===
        print(f"Speichere das finale *beste* Modell (basierend auf '{chosen_metric}')...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"\n🎉 Training erfolgreich abgeschlossen! Das beste Modell wurde im Ordner '{output_dir}' gespeichert.")

        # === SCHRITT 12: Redundante Checkpoints bereinigen ===
        best_model_path = train_result.best_model_checkpoint
        cleanup_checkpoints(output_dir, best_model_path, strategy_save_limit)

    except Exception as e:
        # Stellt sicher, dass auch Abstürze im Log landen
        print(f"❌ EIN SCHWERWIEGENDER FEHLER IST AUFGETRETEN:")
        import traceback
        traceback.print_exc()  # Druckt den vollen Stacktrace in die Log-Datei

    finally:
        # WICHTIG: Am Ende die Standard-Ausgabe wiederherstellen und Datei schließen
        logger.close_logs()
        # Diese letzte Anweisung wird wieder normal in der Konsole angezeigt
        print(f"Konsolen-Logging beendet. Log-Datei gespeichert in: '{console_log_path}'")


if __name__ == "__main__":
    main()