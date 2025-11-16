import os
import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional, Dict, List
from tqdm.auto import tqdm
import re
import argparse
import os.path  # Für Dateipfad-Manipulation

# --- TensorFlow/oneDNN-Warnungen unterdrücken ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Globale NLLB-Sprachcode-Map ---
LANG_CODE_MAP = {
    "de": "deu_Latn", "en": "eng_Latn", "fr": "fra_Latn", "es": "spa_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "nl": "nld_Latn", "pl": "pol_Latn",
    "ru": "rus_Cyrl", "ja": "jpn_Jpan", "zh": "zho_Hans", "ar": "ara_Arab",
}
REVERSE_LANG_MAP = {v: k for k, v in LANG_CODE_MAP.items()}

# --- Modell-Repos ---
DEFAULT_NLLB_REPO = "facebook/nllb-200-3.3B"
HELSINKI_REPO_TEMPLATE = "Helsinki-NLP/opus-mt-{src}-{tgt}"


def map_lang_code(code: str) -> str:
    """Wandelt gängige 2-Buchstaben-Codes in NLLB-Codes um."""
    if not isinstance(code, str):
        return None
    code = code.lower().strip()
    return LANG_CODE_MAP.get(code, code)


def parse_arguments() -> argparse.Namespace:
    """Parst Kommandozeilen-Argumente für die nicht-interaktive Nutzung."""
    parser = argparse.ArgumentParser(description="Batch-Übersetzungs-Skript")

    parser.add_argument("--input_file", type=str,
                        default="trainingsdaten/5_prio_stufen/dataset-tickets-german_normalized_50_5_2.csv",
                        help="Pfad zur Quell-CSV.")
    parser.add_argument("--output_file", type=str,
                        default="trainingsdaten/5_prio_stufen/dataset-tickets-german_normalized_50_5_2_TRANSLATED.csv",
                        help="Pfad zur Ziel-CSV.")
    parser.add_argument("--target_language_nllb", type=str, default="eng_Latn",
                        help="Zielsprache (NLLB-Code, z.B. eng_Latn).")
    parser.add_argument("--columns_to_translate", type=str, default="body,subject",
                        help="Zu übersetzende Spalten (kommagetrennt).")
    parser.add_argument("--lang_column", type=str, default="language",
                        help="Name der Spalte mit 2-Buchstaben-Sprachcodes.")
    parser.add_argument("--default_src_lang", type=str, default="de", help="Standard-Quellsprache (2-Buchstaben-Code).")
    parser.add_argument("--model_family", type=str, default="helsinki", choices=["nllb", "helsinki"],
                        help="Modell-Familie (nllb oder helsinki).")
    parser.add_argument("--nllb_model_repo", type=str, default=DEFAULT_NLLB_REPO,
                        help="Name des NLLB-Modell-Repositorys.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch-Größe. (Standard: 32 für Helsinki, 4 für NLLB).")
    parser.add_argument("--checkpoint_interval", type=int, default=50,
                        help="Checkpoint-Intervall (alle X Zeilen speichern).")
    parser.add_argument("--pre_processing", type=str, default="isolate", choices=["none", "clean", "isolate"],
                        help="Sonderzeichen-Behandlung.")

    return parser.parse_args()


def get_user_config() -> Optional[Dict]:
    """Sammelt alle notwendigen Konfigurationen interaktiv vom Benutzer."""
    config = {}

    # --- Standardwerte ---
    default_input = "trainingsdaten/5_prio_stufen/dataset-tickets-german_normalized_50_5_2.csv"
    default_output = "trainingsdaten/5_prio_stufen/dataset-tickets-german_normalized_50_5_2_TRANSLATED.csv"
    default_target_lang = "eng_Latn"
    default_text_cols = "body,subject"
    default_lang_col = "language"
    default_checkpoint_interval = 50
    default_pre_processing = "isolate"
    default_nllb_repo = DEFAULT_NLLB_REPO
    default_model = "helsinki"

    # 1. Eingabedatei
    while True:
        in_file = input(f"Pfad zur Quell-CSV [Standard: {default_input}]: ").strip()
        config["input_file"] = in_file if in_file else default_input
        if os.path.exists(config["input_file"]):
            break
        print(f"Fehler: Datei nicht gefunden unter '{config['input_file']}'")

    # 2. Ausgabedatei
    out_file = input(f"Pfad zur Ziel-CSV [Standard: {default_output}]: ").strip()
    config["output_file"] = out_file if out_file else default_output

    # 3. Zielsprache
    target_lang_nllb = input(f"Zielsprache (NLLB-Code, z.B. eng_Latn) [Standard: {default_target_lang}]: ").strip()
    config["target_language_nllb"] = target_lang_nllb if target_lang_nllb else default_target_lang
    config["target_language_helsinki"] = REVERSE_LANG_MAP.get(config["target_language_nllb"])
    print(
        f"Zielsprache festgelegt auf: {config['target_language_nllb']} (Helsinki-Code: {config['target_language_helsinki']})")

    # 4. Quelltext-Spalten
    cols_input = input(f"Zu übersetzende Spalten (kommagetrennt) [Standard: {default_text_cols}]: ").strip()
    cols_input = cols_input if cols_input else default_text_cols
    config["columns_to_translate"] = [col.strip() for col in cols_input.split(',')]
    print(f"Spalten zur Übersetzung: {config['columns_to_translate']}")

    # 5. Quellsprache-Spalte
    lang_col = input(
        f"Name der Sprachspalte (sollte 2-Buchstaben-Codes enthalten) [Standard: {default_lang_col}]: ").strip()
    config["lang_column"] = lang_col if lang_col else default_lang_col
    print(f"Quellsprache wird aus Spalte '{config['lang_column']}' gelesen.")
    default_src = "de"
    src_lang = input(f"Standard-Quellsprache (2-Buchstaben-Code) [Standard: {default_src}]: ").strip()
    config["default_src_lang"] = src_lang if src_lang else default_src

    # 6. Modell-Auswahl
    model_input = input(f"Welche Modell-Familie (helsinki, nllb)? [Standard: {default_model}]: ").strip().lower()
    config["model_family"] = model_input if model_input in ["nllb", "helsinki"] else default_model
    print(f"Verwende Modell-Familie: {config['model_family']}")

    # 6a. NLLB-Modell-Repo
    if config["model_family"] == "nllb":
        nllb_repo_input = input(f"NLLB-Modell-Repository [Standard: {default_nllb_repo}]: ").strip()
        config["nllb_model_repo"] = nllb_repo_input if nllb_repo_input else default_nllb_repo
    else:
        config["nllb_model_repo"] = default_nllb_repo

    # 7. Pre-Processing
    print("\nSonderzeichen-Behandlung (PRE-Processing):")
    print("  'none'    = Text 1:1 übergeben (riskiert Übersetzungsfehler).")
    print("  'clean'   = Flacht Text ab (ersetzt \\n, \\t etc. durch Leerzeichen).")
    print("  'isolate' = Isoliert Steuerzeichen (\\n, \\t) von Wörtern (empfohlen).")
    char_input = input(f"Auswahl (none, clean, isolate) [Standard: {default_pre_processing}]: ").strip().lower()
    config["pre_processing"] = char_input if char_input in ["none", "clean", "isolate"] else default_pre_processing
    print(f"Sonderzeichen-Modus: {config['pre_processing']}")

    # 8. Batch-Verarbeitung
    if config["model_family"] == "helsinki":
        default_batch_size = 32
        batch_prompt = f"Batch-Größe (Helsinki optimiert) [Standard: {default_batch_size}]: "
    else:  # nllb
        default_batch_size = 4
        batch_prompt = f"Batch-Größe (NLLB Kompromiss) [Standard: {default_batch_size}]: "

    if config["pre_processing"] == "isolate":
        warn_text = "\nHINWEIS: 'isolate' kann VRAM-Nutzung erhöhen. Ggf. kleinere Batch-Größe wählen.\n"
        print(warn_text)
        batch_prompt = f"Batch-Größe ('isolate' aktiv) [Standard: {default_batch_size}]: "

    try:
        batch_size_str = input(batch_prompt).strip()
        config["batch_size"] = int(batch_size_str) if batch_size_str else default_batch_size
    except ValueError:
        print(f"Ungültige Zahl. Setze Batch-Größe auf {default_batch_size}.")
        config["batch_size"] = default_batch_size
    print(f"Batch-Größe festgelegt auf: {config['batch_size']}")

    # 9. Checkpoint-Intervall
    try:
        interval_str = input(
            f"Checkpoint-Intervall (alle X Zeilen speichern) [Standard: {default_checkpoint_interval}]: ").strip()
        config["checkpoint_interval"] = int(interval_str) if interval_str else default_checkpoint_interval
    except ValueError:
        print(f"Ungültige Zahl. Setze Intervall auf {default_checkpoint_interval}.")
        config["checkpoint_interval"] = default_checkpoint_interval
    print(f"Speichere alle {config['checkpoint_interval']} Zeilen einen Checkpoint.")

    return config


def setup_config() -> Optional[Dict]:
    """Entscheidet, ob Argumente geparst oder interaktiv abgefragt werden."""
    if len(sys.argv) > 1:
        print("--- Lade Konfiguration aus Kommandozeilen-Argumenten ---")
        args = parse_arguments()
        config = vars(args)

        config["target_language_helsinki"] = REVERSE_LANG_MAP.get(config["target_language_nllb"])

        if config["batch_size"] is None:
            config["batch_size"] = 32 if config["model_family"] == "helsinki" else 4

        print(f"Eingabedatei: {config['input_file']}")
        print(f"Modell-Familie: {config['model_family']}")
        print(f"Batch-Größe: {config['batch_size']}")
        print(f"Pre-Processing: {config['pre_processing']}")

        if config["pre_processing"] == "isolate" and config["batch_size"] > 16:
            print(
                f"-> HINWEIS: 'isolate' kann VRAM-Nutzung erhöhen. Die gewählte Batch-Größe von {config['batch_size']} ist möglicherweise zu hoch.")

        return config
    else:
        print("--- Interaktiver Konfigurations-Modus ---")
        return get_user_config()


# --- Modell-Ladefunktionen ---

def load_nllb_model_and_tokenizer(model_repo: str, device: str):
    """Lädt das NLLB-Modell und den Tokenizer."""
    print(f"\n--- Lade NLLB-Modell ({model_repo}) ---")
    print("Dies kann je nach Download und Hardware einige Minuten dauern...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_repo)
        model_args = {}
        if device == "cuda":
            print("GPU (CUDA) erkannt. Lade NLLB-Modell in halber Genauigkeit (fp16)...")
            model_args["dtype"] = torch.float16

        model = AutoModelForSeq2SeqLM.from_pretrained(model_repo, **model_args)
        if device == "cuda":
            model.to(device)
        print("NLLB-Modell erfolgreich geladen.")
        return model, tokenizer
    except Exception as e:
        print(f"Fehler beim Laden des NLLB-Modells ({model_repo}): {e}")
        return None, None


def load_helsinki_model_and_tokenizer(src_code: str, tgt_code: str, device: str):
    """Lädt ein Helsinki-Modell dynamisch."""
    model_name = HELSINKI_REPO_TEMPLATE.format(src=src_code, tgt=tgt_code)

    print(f"\n--- Lade Helsinki-Modell ({model_name}) ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_args = {}

        # <<< (V15-Optimierung): FP16 für Helsinki auf CUDA aktiviert >>>
        if device == "cuda":
            print("GPU (CUDA) erkannt. Lade Helsinki-Modell in halber Genauigkeit (fp16)...")
            model_args["dtype"] = torch.float16

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_args)
        if device == "cuda":
            model.to(device)
        print(f"Helsinki-Modell {model_name} erfolgreich geladen.")
        return model, tokenizer
    except Exception as e:
        print(f"Fehler beim Laden des Helsinki-Modells {model_name}: {e}")
        print("Möglicherweise existiert dieses Sprachpaar nicht bei Helsinki-NLP.")
        return None, None


# --- Übersetzungsfunktion ---

def translate_batch(
        texts: List[str],
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        device: str,
        model_family: str,
        src_lang_nllb: Optional[str] = None,
        tgt_lang_id: Optional[int] = None
) -> List[str]:
    """Übersetzt einen Batch von Texten (generisch für NLLB oder Helsinki)."""
    inputs = None
    generated_tokens = None
    try:
        if model_family == "nllb":
            if src_lang_nllb is None:
                print("WARNUNG: src_lang_nllb fehlt für NLLB. Verwende 'eng_Latn' als Fallback.")
                tokenizer.src_lang = "eng_Latn"
            else:
                tokenizer.src_lang = src_lang_nllb

        # <<< GEÄNDERT (V16): Umstellung auf Dynamic Padding (padding="longest") >>>
        # Dies ist die wichtigste VRAM-Optimierung.
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",  # Füllt nur bis zum längsten Element im Batch, nicht bis max_length
            truncation=True,  # Kürzt Sätze, die länger als max_length sind
            max_length=512  # Maximale Input-Länge
        ).to(device)

        # Output-Länge (512) bleibt als VRAM-Sparmaßnahme (war 1024)
        gen_kwargs = {"max_length": 512}

        if model_family == "nllb":
            if tgt_lang_id is None:
                print("FATALER FEHLER: NLLB tgt_lang_id nicht gesetzt.")
                return ["[FEHLER_TARGET_ID_NICHT_GESETZT]"] * len(texts)
            gen_kwargs["forced_bos_token_id"] = tgt_lang_id

        with torch.no_grad():
            generated_tokens = model.generate(**inputs, **gen_kwargs)

        translated_batch = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )
        return translated_batch

    except Exception as e:
        if "CUDA" in str(e) and "out of memory" in str(e).lower():
            print(f"\n!!! FATALER CUDA-FEHLER (Out of Memory): {e} !!!")
            print("!!! Dynamisches Padding war nicht ausreichend. Bitte Batch-Größe weiter reduzieren. !!!\n")
            raise e
        else:
            print(f"Fehler bei der Übersetzung eines Batches: {e}")
        return ["[ÜBERSETZUNGSFEHLER]"] * len(texts)

    finally:
        try:
            del inputs
            del generated_tokens
            if device == "cuda":
                torch.cuda.empty_cache()
        except UnboundLocalError:
            pass
        except Exception:
            pass

        # --- Hilfsfunktionen ---


def save_dataframe(df, path, pbar_instance=None):
    """Speichert den DataFrame sicher ab."""
    if pbar_instance:
        pbar_instance.set_description(f"Speichere Checkpoint in {os.path.basename(path)}...")
    try:
        df.to_csv(path, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"Fehler beim Speichern des Checkpoints: {e}")
    if pbar_instance:
        pbar_instance.set_description("Übersetzung...")


def preprocess_text_clean(text: str) -> str:
    """Ersetzt alle erkannten Steuerzeichen (\n, \t etc.) durch ein Leerzeichen."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'([\\])([ntrsS])', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def preprocess_text_isolate(text: str) -> str:
    """Isoliert erkannte Steuerzeichen (\n, \t etc.) auf beiden Seiten mit Leerzeichen."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'([\\])([ntrsS])', r' \1\2 ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def find_first_non_empty_example(df: pd.DataFrame, columns: List[str]) -> (Optional[str], str):
    """Sucht den ersten nicht-leeren Text in den angegebenen Spalten für ein Beispiel."""
    for col in columns:
        if col in df.columns:
            for item in df[col].dropna():
                if isinstance(item, str) and item.strip():
                    return item, col
    return None, ""


# <<< NEU (V16): Metrik-Funktion (Ihre Anfrage) >>>
def show_token_metrics(df: pd.DataFrame, config: dict):
    """
    Analysiert die Token-Längen der zu übersetzenden Spalten und gibt Metriken aus.
    """
    print("\n--- 7a. Analysiere Token-Metriken (Vorschau) ---")

    # Lade einen Beispiel-Tokenizer, um die Längen zu schätzen
    # Dies ist eine Annäherung; der Tokenizer für jede Sprache kann leicht abweichen.
    try:
        tokenizer_name = ""
        if config["model_family"] == "nllb":
            tokenizer_name = config["nllb_model_repo"]
        else:
            # Verwende die Standard-Quell/Zielsprache als Schätzung
            src = config["default_src_lang"]
            tgt = config["target_language_helsinki"]
            if not tgt:
                print("Warnung: Zielsprache (Helsinki) nicht gefunden, verwende 'de-en' für Metrik.")
                src, tgt = "de", "en"
            tokenizer_name = HELSINKI_REPO_TEMPLATE.format(src=src, tgt=tgt)

        print(f"Lade Beispiel-Tokenizer für Metriken: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Fehler beim Laden des Metrik-Tokenizers: {e}. Überspringe Metriken.")
        return

    all_lengths = pd.Series(dtype='int64')
    longest_text_overall = ""
    longest_len_overall = 0

    for col_name in config["columns_to_translate"]:
        if col_name in df.columns:
            print(f"Analysiere Spalte '{col_name}'...")

            # (tqdm für Pandas hier nicht nötig, da apply schnell ist)
            lengths = df[col_name].astype(str).apply(
                lambda x: len(tokenizer.encode(x, max_length=4096, truncation=True)))

            # Längsten Text in *dieser* Spalte finden
            max_len_col = lengths.max()
            if max_len_col > longest_len_overall:
                longest_len_overall = max_len_col
                # Finde den Text, der dieser Länge entspricht
                longest_text_overall = df.loc[lengths.idxmax()][col_name]

            all_lengths = pd.concat([all_lengths, lengths])

    if all_lengths.empty:
        print("Keine Daten zum Analysieren gefunden.")
        return

    print("\n--- Token-Metriken (nach Pre-Processing) ---")
    print(f"  Durchschnittl. Token-Anzahl: {all_lengths.mean():.0f}")
    print(f"  95%-Perzentil Token-Anzahl: {all_lengths.quantile(0.95):.0f}")
    print(f"  MAXIMALE Token-Anzahl:      {all_lengths.max()}")
    print("-" * 40)
    print(f"Der längste Text (gekürzt auf 500 Zeichen) hat {longest_len_overall} Tokens:")
    print(longest_text_overall[:500] + "..." if len(longest_text_overall) > 500 else longest_text_overall)
    print("=" * 80)


# --- Hauptlogik ---

def main():
    pbar = None
    df = None
    config = None
    try:
        # 1. Konfiguration abfragen
        config = setup_config()
        if config is None:
            print("Konfiguration fehlgeschlagen.")
            return

        # 2. Gerät festlegen
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Verwende Gerät: {device.upper()}")

        # 3. Modell-Vorbereitung (wird nur geladen, wenn gebraucht)
        nllb_model, nllb_tokenizer, nllb_tgt_id = None, None, None

        if config["model_family"] == "nllb":
            nllb_model, nllb_tokenizer = load_nllb_model_and_tokenizer(config["nllb_model_repo"], device)
            if nllb_model is None: return
            try:
                nllb_tgt_id = nllb_tokenizer.convert_tokens_to_ids(config["target_language_nllb"])
                if nllb_tgt_id == nllb_tokenizer.unk_token_id: raise KeyError
            except KeyError:
                print(f"Fehler: Zielsprache '{config['target_language_nllb']}' ist kein gültiger NLLB-Code.")
                return
        else:
            if not config["target_language_helsinki"]:
                print(
                    f"Fehler: Konnte keinen 2-Buchstaben-Code für Zielsprache '{config['target_language_nllb']}' finden.")
                return
            print("Helsinki-Modelle werden dynamisch pro Sprachpaar geladen und entladen.")

        # 4. Pre-Processing als separater Dateischritt
        print("\n--- 4. Prüfe Pre-Processing ---")

        base_name = os.path.splitext(os.path.basename(config['input_file']))[0]
        dir_name = os.path.dirname(config['input_file'])

        preprocessed_file_path = None
        input_for_translation = config['input_file']  # Standard

        if config["pre_processing"] != "none":
            suffix = f"_{config['pre_processing']}.csv"
            preprocessed_file_path = os.path.join(dir_name, f"{base_name}{suffix}")
            input_for_translation = preprocessed_file_path  # Ziel ist die neue Datei

            print(f"Modus '{config['pre_processing']}' aktiv.")

            force_preprocess = False
            if os.path.exists(preprocessed_file_path):
                print(f"-> 👍 Bereits vorverarbeitete Datei gefunden: {os.path.basename(preprocessed_file_path)}")

                if len(sys.argv) <= 1:
                    overwrite_pre = input("   Möchten Sie diese Datei neu erstellen (j/n)? [n]: ").strip().lower()
                    if overwrite_pre == 'j':
                        force_preprocess = True

            if not os.path.exists(preprocessed_file_path) or force_preprocess:
                if force_preprocess:
                    print(f"-> ⏳ Erzwungenes Überschreiben von: {os.path.basename(preprocessed_file_path)}")
                else:
                    print(f"-> ⏳ Erstelle vorverarbeitete Datei: {os.path.basename(preprocessed_file_path)}")

                try:
                    df_pre = pd.read_csv(config['input_file'])

                    preprocess_func = None
                    if config["pre_processing"] == "clean":
                        preprocess_func = preprocess_text_clean
                    elif config["pre_processing"] == "isolate":
                        preprocess_func = preprocess_text_isolate

                    if preprocess_func:
                        print("\n--- Pre-Processing Vorschau ---")
                        example_before, example_col = find_first_non_empty_example(df_pre,
                                                                                   config["columns_to_translate"])
                        if example_before:
                            example_after = preprocess_func(example_before)
                            print(f"Beispiel aus Spalte '{example_col}':")
                            print("VORHER:\n" + "=" * 80 + f"\n{example_before}\n" + "=" * 80)
                            print(
                                f"\nNACHHER ({config['pre_processing']}-Logik):\n" + "=" * 80 + f"\n{example_after}\n" + "=" * 80 + "\n")
                        else:
                            print("Kein Beispieltext in Quellspalten gefunden.")

                        tqdm.pandas(desc=f"Bereinige ({config['pre_processing']})")

                        for col_name in config["columns_to_translate"]:
                            if col_name in df_pre.columns:
                                print(f"  Verarbeite Spalte '{col_name}'...")
                                df_pre[col_name] = df_pre[col_name].fillna("").astype(str)
                                df_pre[col_name] = df_pre[col_name].progress_apply(preprocess_func)
                            else:
                                print(f"  WARNUNG: Spalte '{col_name}' für Pre-Processing nicht gefunden.")

                        print(f"Speichere in {preprocessed_file_path}...")
                        save_dataframe(df_pre, preprocessed_file_path)
                        print("-> ✅ Vorverarbeitung abgeschlossen.")

                    del df_pre

                except Exception as e:
                    print(f"FEHLER beim Pre-Processing: {e}")
                    return
        else:
            print("-> Modus 'none'. Originaldatei wird direkt verwendet.")

        # 5. CSV laden ODER Checkpoint wiederaufnehmen
        print(f"\n--- 5. Lade Daten für Übersetzung ---")
        if os.path.exists(config['output_file']):
            resume_input = 'f'
            if len(sys.argv) <= 1:
                resume_input = input(
                    f"Zieldatei '{config['output_file']}' existiert.\nFortsetzen (f) oder Überschreiben (ü)? [f]: ").lower().strip()

            if resume_input == 'ü':
                print(f"Lade '{os.path.basename(input_for_translation)}' und überschreibe Ziel...")
                df = pd.read_csv(input_for_translation)
            else:
                print(f"Lade bestehenden Fortschritt aus '{os.path.basename(config['output_file'])}'...")
                df = pd.read_csv(config['output_file'])
        else:
            print(f"\nLese CSV-Datei: {os.path.basename(input_for_translation)}")
            df = pd.read_csv(input_for_translation)

        # 6. Spalten validieren
        print("\n--- 6. Validiere Spalten ---")
        for col_name in config["columns_to_translate"]:
            if col_name not in df.columns:
                print(f"Fehler: Textspalte '{col_name}' nicht in der CSV gefunden. Breche ab.")
                return
        if config["lang_column"] not in df.columns:
            print(f"Fehler: Sprachspalte '{config['lang_column']}' nicht gefunden. Breche ab.")
            return
        print("-> ✅ Alle Spalten gefunden.")

        # 7. Daten für Übersetzung vorbereiten (NaN füllen, Sprachcodes)
        print("--- 7. Bereite Übersetzungs-DataFrame vor ---")
        for col_name in config["columns_to_translate"]:
            df[col_name] = df[col_name].fillna("").astype(str)

        df[config['lang_column']] = df[config['lang_column']].astype(str)
        df['temp_src_lang_2code'] = df[config['lang_column']].str.lower().str.strip()
        df['temp_src_lang_2code'] = df['temp_src_lang_2code'].replace("", config["default_src_lang"])
        df['temp_src_lang_2code'] = df['temp_src_lang_2code'].fillna(config["default_src_lang"])
        df['temp_src_lang_nllb'] = df['temp_src_lang_2code'].apply(map_lang_code)
        print("-> ✅ Sprachspalten und NaN-Werte vorbereitet.")

        # <<< NEU (V16): Metrik-Analyse wird hier ausgeführt >>>
        show_token_metrics(df, config)

        # 8. Äußere Schleife für jede zu übersetzende Spalte
        for col_name in config["columns_to_translate"]:
            target_col_name = f"translated_{col_name}"
            print(f"\n--- 8. Verarbeite Spalte: '{col_name}' -> '{target_col_name}' ---")

            if target_col_name not in df.columns:
                print(f"Erstelle neue Zielspalte: '{target_col_name}'")
                df[target_col_name] = pd.NA
                save_dataframe(df, config['output_file'])

            grouped = df.groupby('temp_src_lang_2code')
            total_rows = len(df)
            processed_rows_count = df[target_col_name].notna().sum()

            print(f"\n--- Starte Übersetzung für '{col_name}' ---")
            print(f"{total_rows} Zeilen insgesamt, {processed_rows_count} bereits übersetzt.")

            pbar = tqdm(total=total_rows, initial=processed_rows_count, desc=f"Übersetze {col_name}", unit="Zeile")

            rows_since_checkpoint = 0
            checkpoint_interval = config["checkpoint_interval"]

            for src_2letter_code, group_df in grouped:
                src_nllb_code = group_df['temp_src_lang_nllb'].iloc[0]
                if pd.isna(src_nllb_code) and config["model_family"] == "nllb":
                    print(f"WARNUNG: Kein NLLB-Code für '{src_2letter_code}' gefunden. Überspringe Gruppe.")
                    src_nllb_code = None

                pbar.set_description(f"Gruppe {src_2letter_code} ({col_name})")

                model_to_use, tokenizer_to_use = None, None

                if config["model_family"] == "nllb":
                    if src_nllb_code is None:
                        unprocessed_mask = group_df[target_col_name].isna()
                        num_to_skip = unprocessed_mask.sum()
                        if num_to_skip > 0:
                            df.loc[group_df[
                                unprocessed_mask].index, target_col_name] = f"[SKIPPED - UNKNOWN NLLB CODE {src_2letter_code}]"
                            pbar.update(num_to_skip)
                            rows_since_checkpoint += num_to_skip
                        continue
                    model_to_use, tokenizer_to_use = nllb_model, nllb_tokenizer
                else:
                    model_to_use, tokenizer_to_use = load_helsinki_model_and_tokenizer(
                        src_2letter_code,
                        config["target_language_helsinki"],
                        device
                    )

                if model_to_use is None:
                    unprocessed_mask = group_df[target_col_name].isna()
                    num_to_skip = unprocessed_mask.sum()
                    if num_to_skip > 0:
                        df.loc[group_df[
                            unprocessed_mask].index, target_col_name] = f"[SKIPPED - MODEL FOR {src_2letter_code} NOT FOUND]"
                        pbar.update(num_to_skip)
                        rows_since_checkpoint += num_to_skip
                    continue

                target_lang_mask = (group_df['temp_src_lang_2code'] == config["target_language_helsinki"]) & (
                    group_df[target_col_name].isna())
                num_to_skip_target = target_lang_mask.sum()
                if num_to_skip_target > 0:
                    df.loc[group_df[target_lang_mask].index, target_col_name] = "[SKIPPED - ALREADY TARGET LANGUAGE]"
                    pbar.update(num_to_skip_target)
                    rows_since_checkpoint += num_to_skip_target

                rows_to_process_mask = (group_df[target_col_name].isna()) & \
                                       (group_df['temp_src_lang_2code'] != config["target_language_helsinki"])
                rows_to_process = group_df[rows_to_process_mask]

                if rows_to_process.empty:
                    if rows_since_checkpoint >= checkpoint_interval:
                        save_dataframe(df, config['output_file'], pbar)
                        rows_since_checkpoint = 0
                    if config["model_family"] == "helsinki":
                        del model_to_use, tokenizer_to_use
                        if device == "cuda": torch.cuda.empty_cache()
                    continue

                batch_size = config["batch_size"]
                texts_to_translate = rows_to_process[col_name].tolist()
                original_indices = rows_to_process.index.tolist()

                for i in range(0, len(texts_to_translate), batch_size):
                    batch_texts = texts_to_translate[i: i + batch_size]
                    batch_indices = original_indices[i: i + batch_size]

                    non_empty_texts, non_empty_indices, empty_indices = [], [], []

                    for text, index in zip(batch_texts, batch_indices):
                        if text.strip():
                            non_empty_texts.append(text)
                            non_empty_indices.append(index)
                        else:
                            empty_indices.append(index)

                    rows_processed_in_this_step = 0
                    if empty_indices:
                        df.loc[empty_indices, target_col_name] = "[SKIPPED - EMPTY INPUT]"
                        pbar.update(len(empty_indices))
                        rows_processed_in_this_step += len(empty_indices)

                    if not non_empty_texts:
                        rows_since_checkpoint += rows_processed_in_this_step
                        if rows_since_checkpoint >= checkpoint_interval:
                            save_dataframe(df, config['output_file'], pbar)
                            rows_since_checkpoint = 0
                        continue

                    pbar.set_description(
                        f"Übersetze {col_name} (Sprache: {src_2letter_code}, {len(non_empty_texts)} Texte)")

                    translated_batch = translate_batch(
                        non_empty_texts,
                        model_to_use,
                        tokenizer_to_use,
                        device,
                        config["model_family"],
                        src_lang_nllb=src_nllb_code,
                        tgt_lang_id=nllb_tgt_id
                    )

                    df.loc[non_empty_indices, target_col_name] = translated_batch

                    pbar.update(len(non_empty_texts))
                    rows_processed_in_this_step += len(non_empty_texts)

                    rows_since_checkpoint += rows_processed_in_this_step
                    if rows_since_checkpoint >= checkpoint_interval:
                        save_dataframe(df, config['output_file'], pbar)
                        rows_since_checkpoint = 0

                if config["model_family"] == "helsinki":
                    print(f"Entlade Helsinki-Modell für {src_2letter_code}...")
                    del model_to_use
                    del tokenizer_to_use
                    if device == "cuda":
                        torch.cuda.empty_cache()

            save_dataframe(df, config['output_file'], pbar)
            rows_since_checkpoint = 0

            pbar.close()
            print(f"--- Übersetzung für Spalte '{col_name}' abgeschlossen ---")

        # 9. Aufräumen
        print("\n--- 9. Alle Übersetzungen abgeschlossen ---")
        if 'temp_src_lang_2code' in df.columns:
            df.drop(columns=['temp_src_lang_2code'], inplace=True)
        if 'temp_src_lang_nllb' in df.columns:
            df.drop(columns=['temp_src_lang_nllb'], inplace=True)

        print(f"Speichere finale Ergebnisse in: {config['output_file']}")
        save_dataframe(df, config['output_file'])

        print("Vorgang erfolgreich abgeschlossen.")

    except KeyboardInterrupt:
        print("\n\nVorgang vom Benutzer abgebrochen.")
        if pbar: pbar.close()
        if df is not None and config:
            print("Speichere aktuellen Fortschritt als finalen Checkpoint...")
            save_dataframe(df, config['output_file'])
            print(f"Checkpoint in '{config['output_file']}' gespeichert.")

    except Exception as e:
        print(f"\nEin unerwarteter Fehler ist aufgetreten: {e}")
        import traceback
        traceback.print_exc()
        if pbar: pbar.close()

    finally:
        print("Skript beendet.")


if __name__ == "__main__":
    main()