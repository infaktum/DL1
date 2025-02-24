# Deep Learning 1 
## Mathematische und programmiertechnische Grundlagen künstlicher neuronaler Netze

Die Notebooks in diesem Paket stammen aus meinen sporadisch abgehaltenen Kursen zu __Künstlichen Neuronalen Netzen__ (KNN), neuerdings eher unter dem Begriff __Deep Learning__ (DL) bekannt.

Bereits Mitte des letzten Jahrhundert wurde mit der Hebbschen Lernregel eine erste Beschreibung vorgenommen, wie ein natürliches neuronales Netzwerk lernen könnte. Zur selben Zeit entstanden bereits theoretische Arbeiten über konstliche neuronale Netzwerke, und mit dem dem __Perzeptron__ (gegen 1960) das erste funktionsfähige neuronale Netzwerk, das bereits in der Lage war, Ziffern zu erkennen.

Waren die damaligen Möglichkeiten zur experimentellen Arbeit an KNN nur wenigen Fachleuten mit Zugriff auf entsprechene selten Ressourcen möglich, so ist heute jeder Interessierte im Besitz ines Laptops mit einem Internet-Zugang in der Lage, wesentlich leistungsfägigere neuronale Netze in Minuten zu entwickeln. Kostenlos stehen ihm dazu großartige Werkzeuge sowie riesige Datenmengen zur freien und sofortigen Verfügung.

### Ziel des Kurses

Der Kurs soll einerseits die mathematischen Grundlagen zur Theorie der künstlichen neuronalen Netzwerke (genauer: __Backpropagation-Netzwerke__) liefern, und andererseits frei zugängliches  programmiertechnisches Werkzeug vorstellen.

##### 1. Die mathematischen Grundlagen

So alt die Idee künstlicher neuronaler Netze ist: Die darunterliegenden mathematischen Prinzipien sind noch wesentlich älter. Das zentrale Prinzip ist die Minimierung einer _Kostenfunktion_ durch den _Gradientenabstieg_. Zum Verständnis dieses Vorgehens sind lediglich einfache Kenntnisse der (multidimensionalen) Anlaysis und Linearen Algebra notwendig, die in diesem Kurs ebenfalls vermittelt werden und zum größten Teil Thema eines Mathematik-Grundkurses der Oberstufe sind. 

Nebenbei werden auch wichtige Begiffe wie __Kostenfunktion__, __Modell__, __Bias__ u.ä. definiert.

##### 2. Die programmiertechnischen Werkzeuge 

Als Programmiersprache hat sich __Python__ in der Künstlichen Intelligenz einen festen Platz gesichert. Dies hat verschiedene Gründe. Bei der Entwicklung von Python hatte Einfachheit und Pragmatismus immer Vorrang vor einem festen Programmier-Paradigma. So bietet etwa auch die Sprache __Java__ mitterweile _funktionale Programmierung_ und _Lambdas_ sowie eine Palette verschiedener Datenstrukturen, doch steht hier immer die Objekt-Orietniterung im Vordergrund und manchmal eben auch im Weg. Python ist hier völlig undogmatisch; Algorithmen und Datenstrukturen sind viel klarer herausgearbeitet, und Funktionen sind wirklich _first class citizens_.

Darüber hinaus bietet die Technologie der __Jupyter-Notebooks__ nicht nur die Möglichkeit, interaktiv schnelle Lösung interaktiv fast schon experimentell zu erarbeiten. Die Notebooks selbst können hinterher auch zur Dokumentation und Demonstration verwendet werden. Die vorhandenen Python-Bibliotheken, allen voran _matplotlib_, ermöglichen es, die unbedingt notwendigen grafischen Darstellungen von Daten und Ergebnissen mit nur weniger Zeilen Code zu erstellen, als etwa zu einem einfachen "Hello, world" in einer OOP-Sprache wie Java notwendig sind.

### Aufbau und Inhalt

Mathematische Erläuterungen und Live-Coding wechseln einander ab. Es existieren keine Folien; die Notebooks enthalten den gesamten Code und die dazugehörenden Erkärungen.

Die zum Einsatz kommende Mathematik ist nicht sonderlich komplex; sie kann trotzdem nicht vollständig in die Tiefe gehend behandelt werden. Im Internet, insbesondere bei der Wikipedia, sind besser Darstellungen zu finden.

Zu den behandelten Themen gehören:

* Lineare Regression
* Das Gradientenabstiegsverfahren
* Das Perzeptron
* Klassifizierungsverfahren wie _k-means_
* Klassifizierung des Iris-Datensatzes durch unterschiedliche Verfahren
* Zeichenerkennung beim MNISt-Datensatz durch ein KNN

### Der Quellcode

Der Python-Code in diesem Kurs wurde bewusst einfach gehalten; Lesbarkeit und verständlichkeit standen im Vordergrund. Weiterhin wurde auf spezifische Python-Bibliotheken wie _scikit-learn_, _Tensorflow_ und _PyTorch_ verzichtet, weil hier das eigentliche Geschehen nicht sichtbar ist. Es kommen lediglich die beiden Python-Bibliotheken __NumPy__ (schnellere numerische Berechnungen) und __matplotlib__ (Visualisierung) zum Einsatz. 

Der Python-Code ist in keiner Weise optimiert, um höhere Geschwindigkeit zu erzielen. Trotzdem ist es mit ihm möglich, die bekannte Aufgabe der Ziffernerkennung des MNIST-Zeichensatzes auf einem einfachen Laptop ohne Spezialprozessor in weniger als einer Minute Rechenzeit zu lösen. Darauf aufbauend können Teilnehmer des Kurses bei Interesse(!) einfach eigene Experimente vornehmen, bevor sie dann vielleicht auf die schlagkräftigeren Werkzeuge umsteigen.

__Hinweis__:Der Quellcode für die neuronalen Netze stammt zum Teil aus dem Buch von _Tariq Rashid_, das ich sehr empfehlen kann.


### Biografie

* Tariq Rashid: _Neuronale Netze selbst programmieren_ (O'Reilly)


### Links

https://archive.ics.uci.edu/

### Dauer der Veranstaltung
7 UE an 1 Tagen

### Voraussetzung
Interesse an den Grundlagen der KI, insbesondere neuronaler Netzwerke.

### Zielgruppe
Beschäftige, die verstehen wollen, wie neuronale Netze funktionieren, oder sogar selbst damit arbeiten wollen.

### Lernziel
Die Teilnehmenden sollen 

* einen tiefen Einblick in die Funktionsweise neuronaler Netzwerke erhalten,
die wichtigsten Begriffe und Ideen verstehen und (im Idealfall, 
falls Programmier-Skills vorhanden sind) danach selbst neuronale Netzwerke programmieren können.
* einen Einblick in die Beliebtheit der Sprache Python in der KI haben.
* den Einsatz des Bias verstehen.

### Hinweis
Es wird viel "live" programmiert (in Python), und Interessierte können auf Wunsch mitmachen. Es sind zum Verstehen des Kurses keine Programmierkenntnisse erforderlich. Die mathematischen Prinzipien sind recht einfach und basieren auf sehr alten Erkenntnissen, die man bereits auf der Schule gelernt hat.

### Inhalt
* Grundlagen, Mathematik und Geschichte der KI
* Die Methode der kleinsten Fehlerquadrate
* Lineare Regression
* Polynome
* Skalarprodukte und Grafikkarten
* Der Bias
* Künstliche Neuronale Netze:
* Das Perzeptron – das erste KNN
* Der Backpropagation-Algorithmus
* Programmierung eines neuronalen Netzwerk, das handgeschriebene Zahlen erkennen kann