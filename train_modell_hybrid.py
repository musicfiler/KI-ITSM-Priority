# train_modell_hybrid.py

# Erforderliche Bibliotheken importieren
import os
import sys
import time
import re
import torch
import pandas as pd
import numpy as np  # Numpy wird hier und weiter unten importiert. Einer ist redundant.
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset, ClassLabel

# Dieser Import ist redundant, da 'numpy as np' bereits oben steht.
# import numpy as np
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
DATA_FILE = os.path.join(BASE_DIR, "dataset-tickets-german_normalized_50_5_2.csv")
NEG_CSV = os.path.join(BASE_DIR, "neg_arguments.csv")
POS_CSV = os.path.join(BASE_DIR, "pos_arguments.csv")
SLA_CSV = os.path.join(BASE_DIR, "sla_arguments.csv")

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
    # Optimierung: max_features=5000 ist ein Hyperparameter.
    #   Ein kleinerer Wert (z.B. 2000) könnte Rauschen reduzieren.
    #   Ein größerer Wert (z.B. 10000) könnte mehr Signale finden.
    vectorizer = TfidfVectorizer(
        stop_words=GERMAN_STOPWORDS,
        max_features=5000,
        ngram_range=(1, 2),  # Sucht nach einzelnen Wörtern (z.B. "Ausfall")
        # und Phrasen aus 2 Wörtern (z.B. "Server down")
        min_df=MIN_DF
    )
    # .fit() lernt nur das Vokabular aus allen Texten
    vectorizer.fit(df['text'])

    # 4. TF-IDF Scores berechnen
    # .transform() erstellt die Matrix, wie oft/wichtig jedes Wort vorkommt
    tfidf_high = vectorizer.transform(df_high_prio['text'])
    tfidf_low = vectorizer.transform(df_low_prio['text'])

    # Berechne den *durchschnittlichen* TF-IDF Score für jedes Wort
    mean_tfidf_high = np.asarray(tfidf_high.mean(axis=0)).ravel()
    mean_tfidf_low = np.asarray(tfidf_low.mean(axis=0)).ravel()

    feature_names = np.array(vectorizer.get_feature_names_out())

    # 5. Differenz-Score
    # Findet Wörter, die in 'high_prio' sehr wichtig sind
    # UND in 'low_prio' sehr unwichtig sind.
    score_diff = mean_tfidf_high - mean_tfidf_low

    results_df = pd.DataFrame({'term': feature_names, 'score_diff': score_diff})

    # 6. Listen extrahieren und speichern
    # Sortiere absteigend -> die besten "Negativ"-Wörter (hohe Prio)
    top_neg_terms = results_df.sort_values(by='score_diff', ascending=False).head(TOP_N_TERMS)['term'].tolist()
    # Sortiere aufsteigend -> die besten "Positiv"-Wörter (niedrige Prio)
    top_pos_terms = results_df.sort_values(by='score_diff', ascending=True).head(TOP_N_TERMS)['term'].tolist()

    # Speichere die Listen als CSV für die manuelle Nachbearbeitung
    pd.DataFrame(top_neg_terms, columns=['term']).to_csv(NEG_CSV, index=False)
    pd.DataFrame(top_pos_terms, columns=['term']).to_csv(POS_CSV, index=False)

    # Leere SLA-Datei als Vorlage erstellen
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

        # .dropna() entfernt leere Zeilen, falls du sie manuell erzeugt hast
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


# ==============================================================================
# === HIER BEGINNT DIE ÄNDERUNG (1 von 2) ===
# Die alte Funktion 'preprocess_with_vocab' wird durch diese ersetzt.
# ==============================================================================

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

    NEU: Nutzt eine HIERARCHISCHE Logik, um Signal-Mischung zu verhindern
         und wendet "Gewichte" an (durch Wiederholung der Tokens), um den 
         Einfluss (den "Ausschlag") zu verstärken.
    """
    if not isinstance(text, str):
        return ""

    text_lower = text.lower()

    # --- HIERARCHISCHE PRÜFUNG ---
    # Wir prüfen in der Reihenfolge der Wichtigkeit.
    # Sobald ein Treffer gefunden wird, wird die Funktion beendet
    # (return), um Signalmischung zu verhindern (z.B. "Anfrage" in
    # einem "kritischen" Ticket).

    # 1. SLA / Core App (höchste Priorität)
    #   Wenn ein SLA-Wort gefunden wird, ignorieren wir alle anderen
    #   negativen oder positiven Wörter. Dies ist der stärkste Trigger.
    if any(re.search(r'\b' + re.escape(p) + r'\b', text_lower) for p in sla_vocab):
        # Wiederhole das Token X-mal, um das "Gewicht" zu erhöhen
        feature_string = " ".join(["KEY_CORE_APP"] * sla_weight)
        return f"{feature_string} [SEP] {text}"

    # 2. Negatives Vokabular (zweit-höchste Priorität)
    #   Wird nur geprüft, wenn KEIN SLA-Wort gefunden wurde.
    if any(re.search(r'\b' + re.escape(p) + r'\b', text_lower) for p in neg_vocab):
        feature_string = " ".join(["KEY_CRITICAL"] * neg_weight)
        return f"{feature_string} [SEP] {text}"

    # 3. Positives Vokabular (dritt-höchste Priorität)
    #   Wird nur geprüft, wenn KEIN SLA- oder Negativ-Wort gefunden wurde.
    if any(re.search(r'\b' + re.escape(p) + r'\b', text_lower) for p in pos_vocab):
        feature_string = " ".join(["KEY_REQUEST"] * pos_weight)
        return f"{feature_string} [SEP] {text}"

    # 4. Fallback: Nichts gefunden
    #    Hier kein Gewicht nötig, da dies der Standardfall ist.
    feature_string = "KEY_NORMAL"
    return f"{feature_string} [SEP] {text}"


# ==============================================================================
# === HIER ENDET DIE ÄNDERUNG (1 von 2) ===
# ==============================================================================


def add_new_tokens_to_tokenizer(tokenizer):
    """Fügt die neuen Signal-Wörter als spezielle Tokens hinzu."""
    # .add_special_tokens informiert den Tokenizer, dass dies
    # "einzelne" Wörter sind, die nicht weiter zerlegt werden sollen.
    tokenizer.add_special_tokens({'additional_special_tokens': NEW_TOKENS})
    print(f"Neue Tokens zum Tokenizer hinzugefügt: {NEW_TOKENS}")
    return tokenizer


# ==============================================================================
# HAUPT-TRAININGSFUNKTION (main)
# ==============================================================================

def main():
    """
    Diese Funktion steuert den gesamten Prozess:
    NEU: 1. Vokabular-Dateien generieren/laden
    2. Konfiguration und Vorab-Prüfungen durchführen
    3. Daten laden (mit datasets)
    4. Label-Spalte vorbereiten (mit ClassLabel)
    NEU: 5. Modell und Tokenizer laden UND ERWEITERN
    NEU: 6. Daten verarbeiten (Anreichern + Tokenisierung)
    7. Modell trainieren
    8. Modell speichern
    """
    print("Starte den HYBRID-Trainingsprozess...")

    # === Schritt 1: Vokabular-Management (Neu) ===
    # Führt die Vokabular-Extraktion (Phase 1) nur aus, wenn nötig.
    generate_vocab_files_if_needed()

    # Lädt das (jetzt manuell bearbeitete) Vokabular für das Training.
    neg_vocab, pos_vocab, sla_vocab = load_vocab_from_csvs()

    # === Schritt 2: Konfiguration, Diagnose und Geräte-Prüfung ===
    # Prüft, ob eine CUDA-fähige GPU verfügbar ist
    is_gpu_available = torch.cuda.is_available()
    if is_gpu_available:
        print("✅ GPU gefunden! Das Training wird auf der GPU ausgeführt. 🚀")
    else:
        print("⚠️ Keine GPU gefunden. Das Training wird auf der CPU ausgeführt (deutlich langsamer).")

    print(f"➡️  Aktuelles Arbeitsverzeichnis: {os.getcwd()}")
    output_dir = "./ergebnisse_hybrid"  # Angepasster Ordnername
    base_log_dir = "logs_hybrid"  # Angepasster Ordnername

    # Dieser Block verhindert Fehler, falls 'logs_hybrid' als Datei existiert
    if os.path.isfile(base_log_dir):
        backup_name = f"logs_hybrid_als_datei_{int(time.time())}.txt"
        print(f"⚠️  Warnung: Datei '{base_log_dir}' blockiert Log-Verzeichnis.")
        print(f"✅ Datei wird umbenannt in '{backup_name}'.")
        os.rename(base_log_dir, backup_name)

    # Wichtige Sicherheitsabfrage, um versehentliches Überschreiben zu verhindern
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
        overwrite_output = False  # Keine Abfrage nötig, wenn Ordner leer ist

    # Erstellt ein eindeutiges Log-Verzeichnis für jeden Lauf
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_name = "train_modell_hybrid"
    run_log_dir = os.path.join(base_log_dir, f"{timestamp}_{script_name}")
    print(f"Logs für diesen Durchlauf werden in '{run_log_dir}' gespeichert.")

    # === Schritt 3: Dataset laden (wie im Original) ===
    print("Lade das Dataset...")
    try:
        # load_dataset ist sehr effizient und lädt die Daten "lazy" (bei Bedarf).
        # Es erstellt automatisch einen Cache (z.B. in ~/.cache/huggingface/datasets)
        # für viel schnelleres Laden beim nächsten Mal.
        dataset = load_dataset('csv', data_files=DATA_FILE)
    except FileNotFoundError:
        print(f"❌ FEHLER: {DATA_FILE} nicht gefunden.")
        sys.exit()

    # === Schritt 4: Label-Spalte vorbereiten (wie im Original) ===
    print("Wandle die 'priority'-Spalte in Klassen-Labels mit fester Reihenfolge um...")
    # ClassLabel ist entscheidend! Es stellt sicher, dass "critical" *immer* # die ID 0 bekommt und "very_low" *immer* die ID 4, unabhängig
    # von der Reihenfolge in der CSV.
    class_label_feature = ClassLabel(names=PRIORITY_ORDER)
    try:
        # .map() ist die Kernfunktion der `datasets`-Bibliothek.
        # 'batched=True' beschleunigt die Umwandlung massiv.
        dataset = dataset.map(
            lambda examples: {"priority": class_label_feature.str2int(examples["priority"])},
            batched=True
        )
    except ValueError as e:
        print(f"❌ FEHLER beim Mappen der Labels: {e}")
        print("Stelle sicher, dass die 'priority'-Spalte in deiner CSV nur folgende Werte enthält:")
        print(PRIORITY_ORDER)
        sys.exit()

    # Weist dem Datensatz-Feature die neuen Label-Informationen zu (wichtig für den Trainer)
    dataset['train'].features['priority'] = class_label_feature
    num_unique_labels = len(PRIORITY_ORDER)
    print(f"✅ 'priority'-Spalte erfolgreich in {num_unique_labels} Labels umgewandelt.")

    # === Schritt 4.1: Dataset in Trainings- und Validierungs-Set aufteilen (KORRIGIERT) ===
    # KORREKTUR: Wir splitten das 'dataset' (rohe Daten), *bevor* wir tokenisieren.
    print("Teile das Dataset in Trainings- und Validierungs-Sets auf (90/10 Split)...")
    # 'dataset["train"]' ist der Standard-Name des Splits von load_dataset
    train_test_split = dataset["train"].train_test_split(test_size=0.1, seed=42)

    # Wir erstellen ein neues DatasetDict mit beiden Splits
    dataset["train"] = train_test_split["train"]
    dataset["validation"] = train_test_split["test"]

    print(
        f"✅ Aufteilung erfolgt: {len(dataset['train'])} Trainings-, {len(dataset['validation'])} Validierungs-Beispiele.")

    # === Schritt 5: Basis-Modell und Tokenizer laden UND ERWEITERN (Modifiziert) ===
    print("Lade das Basis-Modell und den Tokenizer...")
    modell_name = "./distilbert-local"
    try:
        # Lädt den Tokenizer (wandelt Text in Zahlen um)
        tokenizer = AutoTokenizer.from_pretrained(modell_name)
        # Lädt das Modell (die "Intelligenz") und passt den Kopf an
        # unsere 5 Prioritätsklassen an (num_labels=5)
        model = AutoModelForSequenceClassification.from_pretrained(modell_name, num_labels=num_unique_labels)
    except OSError:
        print(f"❌ FEHLER: Lokales Modell '{modell_name}' nicht gefunden.")
        print("Stelle sicher, dass das Modell in diesem Verzeichnis existiert.")
        print("Alternativ: Ändere 'modell_name' auf ein HuggingFace-Modell, z.B. 'distilbert-base-german-cased'.")
        sys.exit()

    # (Neu) Tokenizer und Modell um die Signal-Tokens erweitern
    tokenizer = add_new_tokens_to_tokenizer(tokenizer)

    # (KRITISCH!) Dieser Schritt ist entscheidend. Er passt die
    # Embedding-Matrix des Modells an, damit es die neuen Tokens
    # (KEY_CRITICAL etc.) lernen kann. Vergisst man dies, crasht das Training.
    model.resize_token_embeddings(len(tokenizer))
    print("✅ Tokenizer und Modell um neue Signal-Tokens erweitert.")

    # === Schritt 6: Tokenize-Funktion definieren und anwenden (Modifiziert) ===
    # Diese Funktion reichert den Text AN und tokenisiert ihn DANN.
    def tokenize_and_enrich_function(examples):
        # 1. Text kombinieren (wie im Original)
        raw_texts = [str(subject) + " " + str(body) for subject, body in
                     zip(examples["subject"], examples["body"])]

        # ==============================================================================
        # === HIER BEGINNT DIE ÄNDERUNG (2 von 2) ===
        # Der Aufruf von preprocess_with_vocab wird um die Gewichte (Trigger) erweitert.
        # ==============================================================================
        enriched_texts = [
            preprocess_with_vocab(
                text,
                neg_vocab, pos_vocab, sla_vocab,

                # --- HIER SIND DIE REGLER ---
                # Erhöhe diese Zahlen, um den "Ausschlag" (Trigger) zu verstärken.
                # 5 bedeutet: Das Wort "KEY_CORE_APP" wird 5x an den Anfang
                # des Textes geschrieben, um es für das Modell unübersehbar zu machen.

                sla_weight=5,  # STARKER TRIGGER für SLA-Tickets
                neg_weight=3,  # MITTLERER TRIGGER für kritische Tickets
                pos_weight=1  # LEICHTER TRIGGER für Anfragen
            )
            for text in raw_texts
        ]
        # ==============================================================================
        # === HIER ENDET DIE ÄNDERUNG (2 von 2) ===
        # ==============================================================================

        # 3. Angereicherten Text tokenisieren
        # 'padding="max_length"' füllt alle Texte auf 128 Tokens auf.
        # 'truncation=True' schneidet Texte ab, die länger als 128 Tokens sind.
        # Optimierung: max_length=128 ist ein Kompromiss.
        #   Erhöhen (z.B. auf 256) erfasst mehr Kontext, verbraucht aber
        #   DEUTLICH mehr GPU-Speicher (VRAM) und verlangsamt das Training.
        return tokenizer(enriched_texts, padding="max_length", truncation=True,
                         max_length=128)  # max_length hinzugefügt

    print("Starte Anreicherung und Tokenisierung des Datasets...")
    # KORREKTUR: .map() wird auf das gesamte DatasetDict ('dataset') angewendet.
    # Es verarbeitet jetzt automatisch 'train' UND 'validation'.
    tokenized_datasets = dataset.map(tokenize_and_enrich_function, batched=True)

    # === Schritt 7: Finale Vorbereitung (wie im Original) ===
    print("Benenne die 'priority'-Spalte in 'labels' um...")
    # Der 'Trainer' erwartet, dass die Label-Spalte 'labels' heißt.
    tokenized_datasets = tokenized_datasets.rename_column("priority", "labels")

    # Spalten entfernen (body ist jetzt 'body' nicht 'description')
    try:
        # Wir entfernen die Roh-Text-Spalten, da sie nicht mehr
        # benötigt werden. Das spart Speicher.
        tokenized_datasets = tokenized_datasets.remove_columns(['subject', 'body', 'queue', 'language'])
    except ValueError:
        print("Hinweis: Einige Spalten zum Entfernen wurden nicht gefunden, fahre fort.")
        pass  # Ignoriere Fehler, falls Spalten schon weg sind

    # === Schritt 8: Trainings-Argumente definieren (Optimiert) ===
    training_args = TrainingArguments(
        output_dir=output_dir,

        # --- Kern-Optimierungen ---
        eval_strategy="epoch",  # Evaluiere nach jeder Epoche
        save_strategy="epoch",  # Speichere nach jeder Epoche

        # Dies ist deine "Anti-Overfitting-Versicherung":
        # Der Trainer lädt am Ende automatisch das Modell (den "Checkpoint")
        # zurück, das den besten 'f1'-Score auf dem Validierungs-Set hatte.
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,  # (Da ein höherer F1-Score besser ist)

        # --- Hyperparameter-Tuning ---
        num_train_epochs=10,  # 10 Epochen sind ein guter Start.
        # Dank 'load_best_model_at_end' ist es nicht schlimm,
        # wenn das Modell in Epoche 9 oder 10 schlechter wird.

        # Batch Size ist ein Kompromiss:
        #   Höher (64, 128): Schnelleres Training, stabiler, ABER hoher VRAM-Bedarf.
        #   Niedriger (8, 16): Langsamer, aber passt auf fast jede GPU.
        #   32 ist ein guter Standardwert.
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,  # Kann bei Evaluierung oft höher sein

        learning_rate=5e-5,  # Standard-Lernrate für BERT-Modelle
        weight_decay=0.01,  # Kleine Regularisierung gegen Overfitting

        # Optimierung: 'warmup_ratio' ist robuster als 'warmup_steps'.
        # 0.1 bedeutet: Die ersten 10% der Trainingsschritte werden
        # genutzt, um die Lernrate langsam auf 5e-5 zu steigern.
        # Das stabilisiert das Training enorm.
        # warmup_steps=500, # Alter Wert
        warmup_ratio=0.1,  # Neuer, robusterer Wert

        # --- Logging & Sonstiges ---
        logging_dir=run_log_dir,
        logging_strategy="epoch",  # Logge Metriken (Loss, F1, Acc) jede Epoche
        overwrite_output_dir=overwrite_output,
        report_to="none",  # Deaktiviert Online-Logger wie W&B

        # --- Performance-Optimierung ---
        # Wenn eine GPU vorhanden ist, nutze "Mixed Precision Training".
        # Dies beschleunigt das Training um 30-50% und spart VRAM,
        # fast ohne Genauigkeitsverlust.
        fp16=is_gpu_available,
    )

    # === Schritt 9: Trainer initialisieren ===
    # Der Trainer ist die Haupt-Engine von Hugging Face.
    # Er verbindet Modell, Argumente, Datensätze und die Metrik-Funktion.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],  # Das 90%-Trainingsset
        eval_dataset=tokenized_datasets["validation"],  # Das 10%-Validierungsset
        compute_metrics=compute_metrics,  # Die Funktion zur Erfolgsmessung
    )

    # === Schritt 10: Training starten ===
    print("Starte das optimierte Training (mit Evaluierung)...")
    # Dieser Befehl startet den gesamten Prozess:
    # Epoche 1 -> Trainieren -> Evaluieren (mit compute_metrics) -> Speichern
    # Epoche 2 -> Trainieren -> Evaluieren -> Speichern
    # ...
    # Epoche 10 -> Trainieren -> Evaluieren -> Speichern
    # Am Ende: Lade das beste Modell (z.B. von Epoche 7)
    trainer.train()

    # === Schritt 11: Modell explizit speichern (Verbessert) ===
    print("Speichere das finale *beste* Modell...")
    # Speichert das Modell, das dank 'load_best_model_at_end=True'
    # jetzt im Speicher ist (z.B. das aus Epoche 7 mit dem höchsten F1-Score).
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)  # Speichere auch den Tokenizer (wichtig!)
    print(f"\n🎉 Training erfolgreich abgeschlossen! Das Hybrid-Modell wurde im Ordner '{output_dir}' gespeichert.")


if __name__ == "__main__":
    main()