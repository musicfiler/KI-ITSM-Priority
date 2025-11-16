Die Index.html ist ein komplett lauffähiges HTML-Fragment (inkl. CSS + JavaScript) das:
- den Konsolen-Output farblich hervorhebt (Progress-Balken, Prozent, Loss/epoch-Infos),
- automatisch alle Evaluation-Metriken (Objekte wie {'eval_loss': ..., 'epoch': 14.0}) aus dem Text extrahiert,
- für jede Metrik (eval_loss, eval_accuracy, eval_f1, eval_precision, eval_recall, eval_runtime, eval_samples_per_second, eval_steps_per_second, train_loss, epoch) ein eigenes Chart zeichnet,
- ein kombiniertes Chart (mehrere Metriken zusammen, skalierbar mit sekundärer Y-Achse) anbietet,
- per Tabs zwischen den einzelnen Metrik-Charts wechseln lässt,
- Chart.js verwendet (leichtgewichtig, gut für Linien/vergleichende Plots),
- Fortschrittsbalken im Console-Output in gleicher Farbgebung wie im Screenshot stylt.

Einfügen: kopiere die Datei lokal als .html und öffne im Browser. Wenn dein Konsolen-Log bereits als Textdatei vorliegt, kopiere den Text in das große Textfeld (oder lade ihn per Drag & Drop, siehe Kommentar im HTML). Das Script parst automatisch die epoch-Metriken und aktualisiert Charts.

Kurze Hinweise zur Integration und Anpassung
• 	Falls dein Log-Format abweicht (andere Anführungszeichen, JSON statt Python-dict-Strings), passe die Regex in parseMetricsFromText an. Der Parser nutzt eine robuste Heuristik: Er sucht JSON-ähnliche Objekte, normalisiert einfache Single-Quotes und versucht ein JSON.parse.
• 	Wenn du statt Chart.js lieber d3.js oder anychart nutzen willst, kann ich das Script entsprechend umstellen — Chart.js ist für Linien- und Kombinationsplots am schnellsten einzusetzen.
• 	Die Farbgebung orientiert sich an deinem Screenshot: grün für gute (hohe) Prozente/Accuracy, gelb/orange für mittlere Werte, rot für niedrige. Die gleichen Farben finden sich in den Charts und Progress-Balken.
• 	Optional: Du kannst die Konsole editierbar setzen, sodass das Live-Training den Text per WebSocket / Append pushen kann. Aktuell ist das Script für statische Logs ausgelegt (einfügen / drop / paste).
Wenn du möchtest, passe ich:
• 	das Parsing so an, dass es direkt eine Log-Datei (z. B. PDF/TXT) automatisch einliest,
• 	die Charts so, dass sie interaktiv geringe Glättung (moving average) darstellen,
• 	oder ersetze Chart.js durch d3.js für maßgeschneiderte Visuals.
