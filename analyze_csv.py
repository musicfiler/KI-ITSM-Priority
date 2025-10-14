# analyze_csv.py

import pandas as pd
import os

def main():
    """
    Analysiert die CSV-Datei, um alle einzigartigen 'priority'-Klassen
    und deren Verteilung zu ermitteln.
    """
    # Pfad zur CSV-Datei
    # Stelle sicher, dass dieser Pfad korrekt ist
    csv_file_path = os.path.join('trainingsdaten', 'dataset-tickets-german_normalized_50_5_2.csv')

    print(f"Lese und analysiere die Datei: {csv_file_path}")

    # --- Schritt 1: CSV-Datei laden ---
    try:
        # Lade die CSV in einen pandas DataFrame
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"\nFEHLER: Die Datei '{csv_file_path}' wurde nicht gefunden.")
        print("Stelle sicher, dass der Pfad korrekt ist und du das Skript vom Hauptverzeichnis deines Projekts aus startest.")
        return
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
        return

    # --- Schritt 2: 'priority'-Spalte analysieren ---
    if 'priority' not in df.columns:
        print("\nFEHLER: Die Spalte 'priority' wurde in der CSV-Datei nicht gefunden.")
        return

    # value_counts() zählt die Vorkommen jedes einzigartigen Werts
    priority_counts = df['priority'].value_counts()

    # Sortiere die Ergebnisse nach dem Label-Namen (dem Index) für eine logische Anzeige
    priority_counts = priority_counts.sort_index()

    total_tickets = len(df)
    num_unique_priorities = len(priority_counts)

    # --- Schritt 3: Ergebnisse ausgeben ---
    print("\n" + "="*40)
    print("      Analyse der Ticket-Prioritäten")
    print("="*40)
    print(f"Gesamtzahl der Tickets: {total_tickets}")
    print(f"Anzahl einzigartiger Prioritätsstufen: {num_unique_priorities}\n")

    print("Verteilung der Prioritäten:")
    for priority, count in priority_counts.items():
        percentage = (count / total_tickets) * 100
        print(f"  - {priority:<20}: {count:>5} Tickets ({percentage:5.2f}%)")

    print("="*40)


if __name__ == "__main__":
    main()