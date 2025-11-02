# train_modell_multilingual.py

# Erforderliche Bibliotheken importieren
import os
import sys
import time
import re
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,  # WICHTIG: Die korrekte Klasse für Klassifizierung
    TrainingArguments,
    Trainer
)
from datasets import load_dataset, ClassLabel

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(pred):
    """
    Berechnet Metriken für die Evaluierung.
    Diese Funktion wird vom Trainer nach jeder Evaluierung aufgerufen.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)  # Nimm die Klasse mit der höchsten Wahrscheinlichkeit

    # Berechne F1, Precision, Recall.
    # average='weighted':
    #   Berücksichtigt die "Imbalance" der Klassen. Eine Metrik für die
    #   seltene Klasse 'critical' zählt genauso viel wie für die
    #   häufige Klasse 'low'. Sehr wichtig für Ticket-Priorisierung.
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
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
# Optimierung: Diese Konstanten sind gut ausgelagert.
BASE_DIR = "trainingsdaten"
DATA_FILE = os.path.join(BASE_DIR, "aa_dataset-tickets-multi-lang-5-2-50-version.csv")
NEG_CSV = os.path.join(BASE_DIR, "neg_arguments_multilingual.csv")
POS_CSV = os.path.join(BASE_DIR, "pos_arguments_multilingual.csv")
SLA_CSV = os.path.join(BASE_DIR, "sla_arguments_multilingual.csv")

# Definition der Prioritäts-Reihenfolge (aus dem Original-Skript)
# WICHTIG: Diese Reihenfolge MUSS mit der Logik in generate_vocab...() übereinstimmen
PRIORITY_ORDER = [
    "critical",  # Wird automatisch zu Label ID 0
    "high",  # Wird automatisch zu Label ID 1
    "medium",  # ...
    "low",
    "very_low"  # Wird automatisch zu Label ID 4
]
# Für die Vokabular-Generierung: Welche String-Labels gelten als "Hohe Prio"
HIGH_PRIO_CLASSES_STR = ["critical", "high"]
# Optimierung: TOP_N_TERMS und MIN_DF sind gute "Hyperparameter" für die
# Vokabular-Extraktion. Man kann mit ihnen experimentieren.
TOP_N_TERMS = 75  # Anzahl der automatisch zu extrahierenden Begriffe
MIN_DF = 5  # Ein Begriff muss in min. 5 Tickets vorkommen (verhindert Rauschen)

# Neue Signal-Tokens, die wir dem Modell beibringen
NEW_TOKENS = ['KEY_CORE_APP', 'KEY_CRITICAL', 'KEY_REQUEST', 'KEY_NORMAL']


# ==============================================================================
# HILFSFUNKTIONEN FÜR DAS HYBRID-MODELL
# ==============================================================================

def generate_vocab_files_if_needed():
    """
    Prüft, ob die Vokabular-CSVs existieren.
    Wenn nicht, generiert es sie automatisch mittels TF-IDF-Analyse
    aus der Roh-CSV-Datei.
    """
    # Stellt sicher, dass das Verzeichnis existiert, ohne Fehler zu werfen.
    os.makedirs(BASE_DIR, exist_ok=True)

    # Prüft, ob ALLE drei Dateien vorhanden sind.
    if os.path.exists(NEG_CSV) and os.path.exists(POS_CSV) and os.path.exists(SLA_CSV):
        print("✅ Phase 1: Vokabular-Dateien gefunden. Überspringe automatische Generierung.")
        return

    print("⚠️ Phase 1: Vokabular-Dateien nicht gefunden. Starte automatische Extraktion...")

    # 1. Daten laden (nur für diese Analyse mit Pandas)
    if not os.path.exists(DATA_FILE):
        print(f"❌ FEHLER: Trainingsdatensatz {DATA_FILE} nicht gefunden.")
        sys.exit()

    try:
        df = pd.read_csv(DATA_FILE)
        # Textspalten kombinieren (genau wie im Original-Skript)
        df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
        # Tickets ohne Text oder Prio werden verworfen
        df = df.dropna(subset=['text', 'priority'])
    except KeyError as e:
        print(f"❌ FEHLER: Die Spalte {e} wurde in {DATA_FILE} nicht gefunden.")
        print("Stelle sicher, dass die Spalten 'subject', 'body' und 'priority' existieren.")
        sys.exit()

    print(f"Analysiere {len(df)} Tickets für Vokabular-Extraktion...")

    # 2. Daten aufteilen (basierend auf den STRING-Labels)
    df_high_prio = df[df['priority'].isin(HIGH_PRIO_CLASSES_STR)]
    df_low_prio = df[~df['priority'].isin(HIGH_PRIO_CLASSES_STR)]

    if df_high_prio.empty or df_low_prio.empty:
        print(f"❌ FEHLER: Konnte keine Tickets für hohe Priorität (Werte: {HIGH_PRIO_CLASSES_STR}) finden.")
        sys.exit()

    # 3. TF-IDF Vektorisierer
    # Die Stopwörter sind (korrekterweise) Deutsch, da die TF-IDF-Analyse
    # auf den deutschen Roh-Texten läuft, unabhängig vom späteren KI-Modell.
    vectorizer = TfidfVectorizer(
        stop_words=GERMAN_STOPWORDS,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=MIN_DF
    )
    vectorizer.fit(df['text'])

    # 4. TF-IDF Scores berechnen
    tfidf_high = vectorizer.transform(df_high_prio['text'])
    tfidf_low = vectorizer.transform(df_low_prio['text'])

    mean_tfidf_high = np.asarray(tfidf_high.mean(axis=0)).ravel()
    mean_tfidf_low = np.asarray(tfidf_low.mean(axis=0)).ravel()

    feature_names = np.array(vectorizer.get_feature_names_out())

    # 5. Differenz-Score
    score_diff = mean_tfidf_high - mean_tfidf_low
    results_df = pd.DataFrame({'term': feature_names, 'score_diff': score_diff})

    # 6. Listen extrahieren und speichern
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
    """
    Reichert einen Text mit speziellen Signal-Wörtern (KEY_...) an.

    Nutzt eine HIERARCHISCHE Logik, um Signal-Mischung zu verhindern
    und wendet "Gewichte" an (durch Wiederholung der Tokens), um den
    Einfluss (den "Ausschlag") zu verstärken.
    """
    if not isinstance(text, str):
        return ""

    text_lower = text.lower()

    # --- HIERARCHISCHE PRÜFUNG ---
    # 1. SLA / Core App (höchste Priorität)
    if any(re.search(r'\b' + re.escape(p) + r'\b', text_lower) for p in sla_vocab):
        feature_string = " ".join(["KEY_CORE_APP"] * sla_weight)
        return f"{feature_string} [SEP] {text}"

    # 2. Negatives Vokabular (zweit-höchste Priorität)
    if any(re.search(r'\b' + re.escape(p) + r'\b', text_lower) for p in neg_vocab):
        feature_string = " ".join(["KEY_CRITICAL"] * neg_weight)
        return f"{feature_string} [SEP] {text}"

    # 3. Positives Vokabular (dritt-höchste Priorität)
    if any(re.search(r'\b' + re.escape(p) + r'\b', text_lower) for p in pos_vocab):
        feature_string = " ".join(["KEY_REQUEST"] * pos_weight)
        return f"{feature_string} [SEP] {text}"

    # 4. Fallback: Nichts gefunden
    feature_string = "KEY_NORMAL"
    return f"{feature_string} [SEP] {text}"


def add_new_tokens_to_tokenizer(tokenizer):
    """Fügt die neuen Signal-Wörter als spezielle Tokens hinzu."""
    tokenizer.add_special_tokens({'additional_special_tokens': NEW_TOKENS})
    print(f"Neue Tokens zum Tokenizer hinzugefügt: {NEW_TOKENS}")
    return tokenizer


# ==============================================================================
# HAUPT-TRAININGSFUNKTION (main)
# ==============================================================================

def main():
    """
    Diese Funktion steuert den gesamten Prozess:
    1. Vokabular-Dateien generieren/laden
    2. Konfiguration und Vorab-Prüfungen durchführen
    3. Daten laden (mit datasets)
    4. Label-Spalte vorbereiten (mit ClassLabel)
    5. Modell und Tokenizer laden UND ERWEITERN
    6. Daten verarbeiten (Anreichern + Tokenisierung)
    7. Modell trainieren
    8. Modell speichern
    """
    print("Starte den HYBRID-Trainingsprozess...")

    # === Schritt 1: Vokabular-Management (Neu) ===
    generate_vocab_files_if_needed()
    neg_vocab, pos_vocab, sla_vocab = load_vocab_from_csvs()

    # === Schritt 2: Konfiguration, Diagnose und Geräte-Prüfung ===
    is_gpu_available = torch.cuda.is_available()
    if is_gpu_available:
        print("✅ GPU gefunden! Das Training wird auf der GPU ausgeführt. 🚀")
    else:
        print("⚠️ Keine GPU gefunden. Das Training wird auf der CPU ausgeführt (deutlich langsamer).")

    print(f"➡️  Aktuelles Arbeitsverzeichnis: {os.getcwd()}")

    # === HIER GIBT ES EINE ÄNDERUNG ===
    # Neue Ordnernamen, um das alte Modell nicht zu überschreiben
    output_dir = "./ergebnisse_multilingual"
    base_log_dir = "logs_multilingual"
    # ==================================

    if os.path.isfile(base_log_dir):
        backup_name = f"logs_multilingual_als_datei_{int(time.time())}.txt"
        print(f"⚠️  Warnung: Datei '{base_log_dir}' blockiert Log-Verzeichnis.")
        print(f"✅ Datei wird umbenannt in '{backup_name}'.")
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
    else:
        overwrite_output = False

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_name = "train_modell_multilingual"  # Name angepasst
    run_log_dir = os.path.join(base_log_dir, f"{timestamp}_{script_name}")
    print(f"Logs für diesen Durchlauf werden in '{run_log_dir}' gespeichert.")

    # === Schritt 3: Dataset laden (wie im Original) ===
    print("Lade das Dataset...")
    try:
        dataset = load_dataset('csv', data_files=DATA_FILE)
    except FileNotFoundError:
        print(f"❌ FEHLER: {DATA_FILE} nicht gefunden.")
        sys.exit()

    # === Schritt 4: Label-Spalte vorbereiten (wie im Original) ===
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

    # === Schritt 4.1: Dataset in Trainings- und Validierungs-Set aufteilen (KORRIGIERT) ===
    print("Teile das Dataset in Trainings- und Validierungs-Sets auf (90/10 Split)...")
    train_test_split = dataset["train"].train_test_split(test_size=0.1, seed=42)

    dataset["train"] = train_test_split["train"]
    dataset["validation"] = train_test_split["test"]

    print(
        f"✅ Aufteilung erfolgt: {len(dataset['train'])} Trainings-, {len(dataset['validation'])} Validierungs-Beispiele.")

    # ==============================================================================
    # === HIER BEGINNT DIE ÄNDERUNG ===
    # Der Modellname wird auf das mehrsprachige Modell geändert.
    # ==============================================================================
    print("Lade das Basis-Modell und den Tokenizer...")
    modell_name = "distilbert/distilbert-base-multilingual-cased"
    try:
        tokenizer = AutoTokenizer.from_pretrained(modell_name)
        model = AutoModelForSequenceClassification.from_pretrained(modell_name, num_labels=num_unique_labels)
    except OSError:
        # Fehlermeldung angepasst, da es kein lokales Modell mehr ist
        print(f"❌ FEHLER: Modell '{modell_name}' nicht gefunden.")
        print("Stelle sicher, dass du eine Internetverbindung hast und der Modellname korrekt ist.")
        sys.exit()
    # ==============================================================================
    # === HIER ENDET DIE ÄNDERUNG ===
    # ==============================================================================

    # (Neu) Tokenizer und Modell um die Signal-Tokens erweitern
    tokenizer = add_new_tokens_to_tokenizer(tokenizer)

    # (KRITISCH!) Embedding-Matrix des Modells anpassen
    model.resize_token_embeddings(len(tokenizer))
    print("✅ Tokenizer und Modell um neue Signal-Tokens erweitert.")

    # === Schritt 6: Tokenize-Funktion definieren und anwenden (Modifiziert) ===
    def tokenize_and_enrich_function(examples):
        raw_texts = [str(body) + " " + str(subject) for body, subject in
                     zip(examples["body"], examples["subject"])]

        # Wendet die HIERARCHISCHE Logik und die GEWICHTUNG an
        enriched_texts = [
            preprocess_with_vocab(
                text,
                neg_vocab, pos_vocab, sla_vocab,

                # --- HIER SIND DIE REGLER ---
                sla_weight=5,  # STARKER TRIGGER für SLA-Tickets
                neg_weight=4,  # MITTLERER TRIGGER für kritische Tickets
                pos_weight=1  # LEICHTER TRIGGER für Anfragen
            )
            for text in raw_texts
        ]

        # 3. Angereicherten Text tokenisieren
        # Optimierung: Dieses Modell (distilbert-multilingual) kann bis zu 512
        # Tokens verarbeiten. 128 ist schneller und speichereffizienter,
        # aber 256 oder 512 könnten mehr Kontext erfassen.
        return tokenizer(enriched_texts, padding="max_length", truncation=True,
                         max_length=256)  # Teste hier ggf. mit 256

    print("Starte Anreicherung und Tokenisierung des Datasets...")
    tokenized_datasets = dataset.map(tokenize_and_enrich_function, batched=True)

    # === Schritt 7: Finale Vorbereitung (wie im Original) ===
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
        num_train_epochs=80,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=run_log_dir,
        logging_strategy="epoch",
        overwrite_output_dir=overwrite_output,
        report_to="none",
        fp16=is_gpu_available,  # Aktiviert Mixed Precision, wenn GPU vorhanden
    )

    # === Schritt 9: Trainer initialisieren ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    # === Schritt 10: Training starten ===
    print("Starte das optimierte Training (mit Evaluierung)...")
    trainer.train()

    # === Schritt 11: Modell explizit speichern (Verbessert) ===
    print("Speichere das finale *beste* Modell...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(
        f"\n🎉 Training erfolgreich abgeschlossen! Das mehrsprachige Modell wurde im Ordner '{output_dir}' gespeichert.")


if __name__ == "__main__":
    main()