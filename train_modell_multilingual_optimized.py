# train_modell_multilingual_optimized.py

# Erforderliche Bibliotheken importieren
import os
import sys
import io  # NEU: Erforderlich für das Logging mit korrekter Kodierung
import time
import re
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_dataset, ClassLabel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(pred):
    """
    Berechnet Metriken für die Evaluierung.
    Diese Funktion wird vom Trainer nach jeder Evaluierung aufgerufen.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Berechne F1, Precision, Recall. (average='weighted' ist korrekt)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# Versuche, deutsche Stopwörter zu laden
try:
    from stop_words import get_stop_words

    GERMAN_STOPWORDS = get_stop_words('de')
except ImportError:
    print("WARNUNG: Paket 'stop-words' nicht gefunden. (pip install stop-words)")
    print("Fahre ohne deutsche Stopwörter fort.")
    GERMAN_STOPWORDS = []

# --- Globale Konfiguration für Hybrid-Modell ---
BASE_DIR = "trainingsdaten"
DATA_FILE = os.path.join(BASE_DIR, "5_prio_stufen/dataset-tickets-german_normalized_50_5_2.csv")
NEG_CSV = os.path.join(BASE_DIR, "neg_arguments_multilingual.csv")
POS_CSV = os.path.join(BASE_DIR, "pos_arguments_multilingual.csv")
SLA_CSV = os.path.join(BASE_DIR, "sla_arguments_multilingual.csv")

PRIORITY_ORDER = ["critical", "high", "medium", "low", "very_low"]
HIGH_PRIO_CLASSES_STR = ["critical", "high"]
TOP_N_TERMS = 75
MIN_DF = 5
NEW_TOKENS = ['KEY_CORE_APP', 'KEY_CRITICAL', 'KEY_REQUEST', 'KEY_NORMAL']


# ==============================================================================
# HILFSFUNKTIONEN FÜR DAS HYBRID-MODELL
# (Funktionen generate_vocab_files_if_needed, load_vocab_from_csvs,
#  preprocess_with_vocab und add_new_tokens_to_tokenizer bleiben
#  exakt gleich.)
# ==============================================================================
# NOTE: Die folgenden Funktionen sind hier zur Vollständigkeit eingefügt.
# Sie sind identisch mit Ihrem Original-Code.

def generate_vocab_files_if_needed():
    """
    Prüft, ob die Vokabular-CSVs existieren.
    Wenn nicht, generiert es sie automatisch mittels TF-IDF-Analyse
    aus der Roh-CSV-Datei.
    """
    os.makedirs(BASE_DIR, exist_ok=True)
    if os.path.exists(NEG_CSV) and os.path.exists(POS_CSV) and os.path.exists(SLA_CSV):
        print("✅ Phase 1: Vokabular-Dateien gefunden. Überspringe automatische Generierung.")
        return
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
    # 1. SLA / Core App
    if any(re.search(r'\b' + re.escape(p) + r'\b', text_lower) for p in sla_vocab):
        feature_string = " ".join(["KEY_CORE_APP"] * sla_weight)
        return f"{feature_string} [SEP] {text}"
    # 2. Negatives Vokabular
    if any(re.search(r'\b' + re.escape(p) + r'\b', text_lower) for p in neg_vocab):
        feature_string = " ".join(["KEY_CRITICAL"] * neg_weight)
        return f"{feature_string} [SEP] {text}"
    # 3. Positives Vokabular
    if any(re.search(r'\b' + re.escape(p) + r'\b', text_lower) for p in pos_vocab):
        feature_string = " ".join(["KEY_REQUEST"] * pos_weight)
        return f"{feature_string} [SEP] {text}"
    # 4. Fallback
    feature_string = "KEY_NORMAL"
    return f"{feature_string} [SEP] {text}"


def add_new_tokens_to_tokenizer(tokenizer):
    """Fügt die neuen Signal-Wörter als spezielle Tokens hinzu."""
    tokenizer.add_special_tokens({'additional_special_tokens': NEW_TOKENS})
    print(f"Neue Tokens zum Tokenizer hinzugefügt: {NEW_TOKENS}")
    return tokenizer


# ==============================================================================
# NEU: Logger-Klasse, um die Konsole in eine Datei umzuleiten
# ==============================================================================
class ConsoleLogger(object):
    """
    Leitet print-Anweisungen (stdout) und Fehler (stderr)
    sowohl an die Konsole als auch in eine Log-Datei um.
    """

    def __init__(self, log_path: str):
        self.terminal_stdout = sys.stdout
        self.terminal_stderr = sys.stderr
        # 'w' (write) modus, um die Log-Datei bei jedem Lauf neu zu erstellen
        # 'utf-8' encoding ist wichtig für Umlaute und Sonderzeichen
        self.log_file = io.open(log_path, 'w', encoding='utf-8')

    def hook_stdout(self):
        """Leitet sys.stdout auf diese Klasse um."""
        sys.stdout = self

    def hook_stderr(self):
        """Leitet sys.stderr auf diese Klasse um."""
        sys.stderr = self

    def write(self, message):
        """
        Wird von sys.stdout/sys.stderr aufgerufen.
        Schreibt die Nachricht sowohl ins Terminal als auch in die Datei.
        """
        self.terminal_stdout.write(message)
        self.log_file.write(message)

    def flush(self):
        """
        flush() ist wichtig, damit Ausgaben sofort geschrieben werden
        und nicht im Puffer hängen bleiben.
        """
        self.terminal_stdout.flush()
        self.log_file.flush()

    def close_logs(self):
        """Stellt die originalen stdout/stderr wieder her und schließt die Datei."""
        sys.stdout = self.terminal_stdout
        sys.stderr = self.terminal_stderr
        self.log_file.close()


# ==============================================================================
# HAUPT-TRAININGSFUNKTION (main)
# ==============================================================================

def main():
    # --- MODIFIZIERTER START: Logging wird zuerst konfiguriert ---

    # 1. Log-Verzeichnisse und Dateinamen definieren
    base_log_dir = "logs_multilingual_optimized"
    output_dir = "./ergebnisse_multilingual_optimized"

    # 1a. Blockier-Check (muss VOR dem Erstellen des run_log_dir passieren)
    if os.path.isfile(base_log_dir):
        # Diese print-Anweisungen werden (einmalig) NICHT geloggt,
        # da der Logger noch nicht aktiv ist. Das ist OK.
        print(f"⚠️  Warnung: Datei '{base_log_dir}' blockiert Log-Verzeichnis.")
        backup_name = f"logs_multilingual_als_datei_{int(time.time())}.txt"
        print(f"✅ Datei wird umbenannt in '{backup_name}'.")
        os.rename(base_log_dir, backup_name)

    # 1b. Log-Pfade definieren
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_name = "train_modell_multilingual_optimized"

    # Das Verzeichnis, in dem ALLE Logs für diesen Lauf landen
    # (sowohl Trainer-Metriken als auch die Konsolen-Logdatei)
    run_log_dir = os.path.join(base_log_dir, f"{timestamp}_{script_name}")

    # Sicherstellen, dass das Log-Verzeichnis existiert
    os.makedirs(run_log_dir, exist_ok=True)

    # Der Dateipfad für die Konsolen-Logdatei
    console_log_path = os.path.join(run_log_dir, f"{timestamp}_{script_name}_console.log")

    # 2. Logger starten und sys.stdout/stderr umleiten
    logger = ConsoleLogger(console_log_path)
    logger.hook_stdout()  # Leitet alle print() Anweisungen um
    logger.hook_stderr()  # Leitet alle Warnungen und Fehler um

    # --- ENDE MODIFIZIERTER START ---

    # 3. try...finally-Block, um sauberes Schließen sicherzustellen
    try:
        """
        Diese Funktion steuert den gesamten Prozess:
        ... (wie im Original-Skript) ...
        """
        print(f"Starte den HYBRID-Trainingsprozess (OPTIMIERT)...")
        print(f"Alle Konsolenausgaben werden in '{console_log_path}' gespeichert.")

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

        # (Pfade wurden bereits oben definiert)

        overwrite_output = False
        if os.path.isdir(output_dir) and os.listdir(output_dir):
            print(f"⚠️  Es sind bereits Daten im Ausgabeverzeichnis '{output_dir}' vorhanden.")
            while True:
                # WICHTIG: input() wird durch das Logging korrekt verarbeitet
                choice = input("Möchten Sie die vorhandenen Ergebnisse überschreiben? (j/n): ").lower()
                print(f"Benutzereingabe: {choice}")  # Loggt die Eingabe
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

        # (run_log_dir wurde bereits oben definiert und erstellt)
        print(f"Logs für diesen Durchlauf werden in '{run_log_dir}' gespeichert.")

        # === Schritt 3: Dataset laden ===
        print("Lade das Dataset...")
        try:
            dataset = load_dataset('csv', data_files=DATA_FILE)
        except FileNotFoundError:
            print(f"❌ FEHLER: {DATA_FILE} nicht gefunden.")
            sys.exit()

        # === Schritt 4: Label-Spalte vorbereiten ===
        print("Wandle die 'priority'-Spalte in Klassen-Labels mit fester Reihenfolge um...")
        class_label_feature = ClassLabel(names=PRIORITY_ORDER)
        try:
            dataset = dataset.map(
                lambda examples: {"priority": class_label_feature.str2int(examples["priority"])},
                batched=True
            )
        except ValueError as e:
            print(f"❌ FEHLER beim Mappen der Labels: {e}")
            print("Stelle sicher, dass die 'priority'-Spalte in deiner CSV nur folgende Werte enthält:")
            print(PRIORITY_ORDER)
            sys.exit()

        dataset['train'].features['priority'] = class_label_feature
        num_unique_labels = len(PRIORITY_ORDER)
        print(f"✅ 'priority'-Spalte erfolgreich in {num_unique_labels} Labels umgewandelt.")

        # === Schritt 4.1: Dataset in Trainings- und Validierungs-Set aufteilen ===
        print("Teile das Dataset in Trainings- und Validierungs-Sets auf (90/10 Split)...")
        train_test_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset["train"] = train_test_split["train"]
        dataset["validation"] = train_test_split["test"]
        print(
            f"✅ Aufteilung erfolgt: {len(dataset['train'])} Trainings-, {len(dataset['validation'])} Validierungs-Beispiele.")

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
                    # --- HIER SIND DIE REGLER ---
                    sla_weight=5,
                    neg_weight=4,
                    pos_weight=1
                )
                for text in raw_texts
            ]

            # max_length=256 ist ein guter Kompromiss
            return tokenizer(enriched_texts, padding="max_length", truncation=True, max_length=256)

        print("Starte Anreicherung und Tokenisierung des Datasets...")
        tokenized_datasets = dataset.map(tokenize_and_enrich_function, batched=True)

        # === Schritt 7: Finale Vorbereitung ===
        print("Benenne die 'priority'-Spalte in 'labels' um...")
        tokenized_datasets = tokenized_datasets.rename_column("priority", "labels")
        try:
            tokenized_datasets = tokenized_datasets.remove_columns(['subject', 'body', 'queue', 'language'])
        except ValueError:
            print("Hinweis: Einige Spalten zum Entfernen wurden nicht gefunden, fahre fort.")
            pass

        # === Schritt 8: Trainings-Argumente definieren (Optimiert) ===
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            num_train_epochs=30,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_dir=run_log_dir,  # Der Trainer nutzt dasselbe Verzeichnis
            logging_strategy="epoch",
            overwrite_output_dir=overwrite_output,
            report_to="none",
            fp16=is_gpu_available,
        )

        # === Schritt 9: Trainer initialisieren (Optimiert) ===
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=5, # Standard war 3, Gemini empfiehlt zwischen 5 und 7
                    early_stopping_threshold=0.001 # Standard war 0.001, Gemini empfiehlt eine Veränderungsempfindlichkeit von 0.00 bei patience von 3
                )
            ]
        )

        # === Schritt 10: Training starten ===
        print("Starte das optimierte Training (mit Early Stopping)...")
        trainer.train()

        # === Schritt 11: Modell explizit speichern ===
        print("Speichere das finale *beste* Modell...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(
            f"\n🎉 Training erfolgreich abgeschlossen! Das beste Modell wurde im Ordner '{output_dir}' gespeichert.")

    except Exception as e:
        # Stellt sicher, dass auch Abstürze im Log landen
        print(f"❌ EIN SCHWERWIEGENDER FEHLER IST AUFGETRETEN:")
        import traceback
        traceback.print_exc()  # Druckt den vollen Stacktrace in die Log-Datei

    finally:
        # WICHTIG: Am Ende die Standard-Ausgabe wiederherstellen und Datei schließen
        logger.close_logs()
        # Diese letzte Anweisung wird wieder normal in der Konsole angezeigt
        # (und nicht mehr in die Datei geschrieben)
        print(f"Konsolen-Logging beendet. Log-Datei gespeichert in: '{console_log_path}'")


if __name__ == "__main__":
    main()