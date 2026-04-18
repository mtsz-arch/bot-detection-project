# 🧠 Wykrywanie botów w komentarzach Reddit (Machine Learning)

##

Projekt polega na stworzeniu modelu uczenia maszynowego, który analizuje komentarze użytkowników na platformie Reddit w celu wykrywania potencjalnej aktywności botów.

Model wykorzystuje cechy językowe oraz statystyczne, aby sklasyfikować treści jako:

* generowane przez człowieka
* generowane przez automat (bot)

Projekt jest inspirowany koncepcją tzw. *"Dead Internet Theory"*.

---

## Dane

Zbiór danych pochodzi z Kaggle:
https://www.kaggle.com/datasets/nudratabbas/the-dead-internet-theory-reddit-bot-vs-human

Zawiera komentarze wraz z etykietą (bot / human).

---

## Użyte Technologie

* Python
* pandas
* scikit-learn
* seaborn / matplotlib

---

## Etapy projektu

### 1. Wczytanie i eksploracja danych

* analiza struktury danych (`df.info()`)
* sprawdzenie braków (`df.isnull()`)
* wstępna eksploracja

### 2. Przygotowanie danych

* czyszczenie danych
* skalowanie cech (min-max)
* przygotowanie danych do modelu

### 3. Inżynieria cech

* wykorzystanie cech językowych
* cechy statystyczne komentarzy

### 4. Trenowanie modelu

* zastosowanie algorytmów klasyfikacji
* podział na zbiór treningowy i testowy

### 5. Ewaluacja

* ocena skuteczności modelu
* analiza wyników

---

## Wyniki

Model pozwala na rozróżnianie komentarzy generowanych przez boty i ludzi na podstawie analizy tekstu oraz cech statystycznych.

(Szczegółowe wyniki znajdują się w notebooku)

---

## Autor

Projekt wykonany w ramach zajęć z uczenia maszynowego.
