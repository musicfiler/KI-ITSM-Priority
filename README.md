# KI-ITSM-Priority

Ein Machine-Learning-Projekt zur automatischen Analyse und Priorisierung von ITSM-Tickets basierend auf der initialen Nachricht des Kunden.
Requires Python 3.13 !!

## Projektziel

Das Ziel dieses Projekts ist es, ein auf `distilbert-base-uncased` feinabgestimmtes Modell zu trainieren, das in der Lage ist, deutschsprachige Support-Tickets zu lesen und ihnen eine von fünf Prioritätsstufen (`critical`, `high`, `medium`, `low`, `very_low`) zuzuweisen.

---

## Setup & Installation

Folge diesen Schritten, um das Projekt lokal einzurichten.

### 1. Repository klonen

```bash
git clone <URL-deines-Repositories>
cd KI-ITSM-Priority
```

### 2. Virtuelle Umgebung erstellen

Es wird dringend empfohlen, eine virtuelle Umgebung zu verwenden.

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Python-Pakete installieren

Installiere alle notwendigen Bibliotheken mit der `requirements.txt`-Datei.

```bash
pip install -r requirements.txt
```

### 4. Trainingsdaten beziehen (2 Optionen)

Du benötigst den Datensatz `multilingual-customer-support-tickets`.

#### Option A: Manueller Download (Empfohlen, falls der automatische fehlschlägt)

1.  Gehe zur Kaggle-Datensatzseite: [tobiasbueck/multilingual-customer-support-tickets](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)
2.  Klicke auf den **Download**-Button, um die ZIP-Datei herunterzuladen.
3.  Entpacke das heruntergeladene ZIP-Archiv.
4.  Erstelle im Hauptverzeichnis deines Projekts einen Ordner namens `trainingsdaten`.
5.  Kopiere die Datei `dataset-tickets-german_normalized_50_5_2.csv` aus dem entpackten Archiv in den `trainingsdaten`-Ordner.

#### Option B: Automatischer Download

Für den automatischen Download musst du einmalig deinen Kaggle API-Token einrichten:
1.  Gehe zu deinem Kaggle-Account -> "API" -> "Create New Token", um die `kaggle.json` herunterzuladen.
2.  Platziere die `kaggle.json` in `C:\Users\<Dein-Benutzer>\.kaggle\` (Windows) oder `~/.kaggle/` (macOS/Linux).

### 5. Setup-Skript ausführen

Das Setup-Skript lädt das Basis-Modell von Hugging Face und versucht, die Trainingsdaten von Kaggle herunterzuladen, falls sie nicht manuell hinzugefügt wurden.

```bash
python setup_project.py
```

---

## Verwendung

Nach dem erfolgreichen Setup kannst du die folgenden Skripte ausführen:

1.  **Daten analysieren (Optional)**
    ```bash
    python analyze_csv.py
    ```
2.  **Modell trainieren**
    ```bash
    python train_model.py
    ```
3.  **Mit dem Modell chatten**
    ```bash
    python chat_with_model.py
    ```

---

## Wichtiger Hinweis zum GPU-Training

Der `transformers.Trainer` erkennt eine GPU automatisch. Das Hauptproblem ist oft nicht der Code, sondern eine PyTorch-Installation ohne GPU-Unterstützung (CUDA).

1.  **Bestehende PyTorch-Version deinstallieren:**
    ```bash
    pip uninstall torch torchvision torchaudio
    ```
2.  **GPU-Treiber prüfen:** Öffne die Kommandozeile und gib `nvidia-smi` ein, um deine unterstützte CUDA-Version zu sehen.
3.  **Korrekten Installationsbefehl generieren:** Gehe auf die [offizielle PyTorch-Webseite](https://pytorch.org/get-started/locally/) und wähle die passenden Optionen für dein System (z.B. Stable, Windows, Pip, CUDA).
    *Beispiel für eine NVIDIA 4070 (CUDA 12.1 oder höher):*
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```
