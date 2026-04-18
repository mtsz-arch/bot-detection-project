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

* analiza struktury danych
 <img width="602" height="298" alt="obraz" src="https://github.com/user-attachments/assets/0fee07f7-4276-4613-8afb-0b3d6f228d83" />
<img width="290" height="383" alt="obraz" src="https://github.com/user-attachments/assets/1c9e97f9-b2aa-4ee7-8389-e48265186445" />
* wstępna eksploracja
<img width="661" height="137" alt="obraz" src="https://github.com/user-attachments/assets/3fb4ce5a-f7a7-4b0f-a987-7e00b86c9a40" />
<img width="661" height="137" alt="obraz" src="https://github.com/user-attachments/assets/ce082948-dfa5-411e-acda-f8ecf54a34b8" />

- Najbardziej rozróżniającą cechą okazała się średnia długość słów (`avg_word_length`) — boty częściej używają dłuższych, bardziej jednorodnych słów.
- Zmienna `contains_links` sugeruje, że boty częściej zawierają linki niż komentarze pisane przez ludzi.
- Pozostałe cechy, takie jak `user_karma`, `account_age_days` czy `sentiment_score`, mają ograniczoną zdolność separacji klas.
- Nie zaobserwowano silnych korelacji liniowych między większością zmiennych, co sugeruje potrzebę bardziej zaawansowanych modeli lub feature engineeringu.

Wniosek: skuteczna klasyfikacja botów wymaga kombinacji wielu cech — pojedyncze zmienne nie są wystarczające do jednoznacznego rozróżnienia.
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
