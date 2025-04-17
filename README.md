# Konservierungsmapping
Repositorium mit Materialien zum Mapping ausgewählter Konzepte des LEIZA Konservierungsthesaurus auf den Getty AAT

## Konservierungs- und Restaurierungsfachthesaurus für archäologische Kulturgüter

<img src="https://github.com/LasseMempel/Konservierungsmapping/blob/main/Thesaurus_Logo.png" width="500">

Der Fachthesaurus kann unter [https://lassemempel.github.io/terminologies/conservationthesaurus.html](https://lassemempel.github.io/terminologies/conservationthesaurus.html) in seiner hierarchischen Struktur durchblättert werden und ist als JSON-LD downloadbar.

... Folien zur Geschichte des Thesaurus folgen ...

## Open Refine

Open Refine kann hier [https://openrefine.org/download.html](https://openrefine.org/download.html) heruntergeladen werden. Das Open Source Tool ermöglicht die Bearbeitung, Bereinigung und Transformation von Daten.

## Reconciliation

Die Verwendung von Reconciliation Services ([https://reconciliation-api.github.io/specs/1.0-draft/](https://reconciliation-api.github.io/specs/1.0-draft/)) ermöglicht den Datenabgleich zwischen verschiedenen Quellen, um diese aufeinander zu matchen.
Dieser Prozess wird für Open Refine hier ([https://openrefine.org/docs/manual/reconciling](https://openrefine.org/docs/manual/reconciling)) detaillierter erklärt.

## Getty AAT

Der Art & Architecture Thesaurus ® (AAT) (([https://www.getty.edu/research/tools/vocabularies/aat/about.html](https://www.getty.edu/research/tools/vocabularies/aat/about.html))) ist eine metadatenreiche Knowledge im Bereich der Kunst, Architektur und der materiellen Kultur, die häufig zur Anbindung weiterer Vokabulare durch Mappings verwendet wird. Durch eine eigene Reconciliation API kann dieser Prozess teilweise automatisiert werden. Das Getty AAT Open Refine Tutorial im Repositorium erklärt das Anlegen eines neuen Open Refine Projektes, die Einbindung des Getty Reconciliation Services, Auswahl geeigneter Vergleichsfelder und Durchführung des Mappings.

Link für Open Refine: https://services.getty.edu/vocab/reconcile/.

## Ein Mapping-Experiment

Uns interessiert inwieweit Menschen beim Mapping von Vokabularen übereinstimmende oder sich unterscheidende Entscheidungen treffen und freuen uns sehr, wenn ihr uns dabei unterstützt. Deshalb möchten wir euch bitten:

1. Die Datei Konservierungsbegriffe.csv in Open refine zu laden
2. Anhand des Tutorials den Getty AAT Reconciliation Service in Open Refine einzubinden.
    - verwendet dazu die Spalte prefLabel@en, da diese die größte Übereinstimmung zu den ebenfalls englischen Labels des GettyAAT bietet
    - Kreuzt als Type AAT search an
    - Und entfernt den Haken bei Auto-Match
3. Übereinstimmende Begriffe auszuwählen. 
4. Unter dem Spalten-Menupünkt "Reconcile" lässt sich nun eine neue Spalte mit den URLs der Treffer erzeugen
5. Kopiert diese Treffer bitte je nach Übereinstimmung in die Spalte exactMatch, closeMatch oder relatedMatch des jeweiligen Begriffs
6. Falls keine passenden Treffervorschläge angezeigt werden, lassen sich vielleicht auch weitere Kandidaten über ([https://www.getty.edu/research/tools/vocabularies/aat/l](https://www.getty.edu/research/tools/vocabularies/aat/)) finden.

Vielen Dank für eure Hilfe!









