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
<img width="671" height="527" alt="obraz" src="https://github.com/user-attachments/assets/1e1c70d2-c21e-4186-b6ee-5a18d7afa777" />

<img width="1575" height="1528" alt="wykres" src="https://github.com/user-attachments/assets/777115e5-c750-444d-8a7a-f303a41fa81a" />

- Najbardziej rozróżniającą cechą okazała się średnia długość słów (`avg_word_length`) — boty częściej używają dłuższych, bardziej jednorodnych słów.
- Zmienna `contains_links` sugeruje, że boty częściej zawierają linki niż komentarze pisane przez ludzi.
- Pozostałe cechy, takie jak `user_karma`, `account_age_days` czy `sentiment_score`, mają ograniczoną zdolność separacji klas.
- Nie zaobserwowano silnych korelacji liniowych między większością zmiennych, co sugeruje potrzebę bardziej zaawansowanych modeli lub feature engineeringu.

Wniosek: skuteczna klasyfikacja botów wymaga kombinacji wielu cech — pojedyncze zmienne nie są wystarczające do jednoznacznego rozróżnienia.
### 2. Przygotowanie danych

* czyszczenie danych
  <img width="451" height="457" alt="obraz" src="https://github.com/user-attachments/assets/9ca0d15e-5f14-4ec1-a0f4-2990fb334933" />

* skalowanie cech (min-max)
  <img width="1178" height="716" alt="obraz" src="https://github.com/user-attachments/assets/6fbb9a8f-1569-4c05-bf15-b90af590acc2" />

* przygotowanie danych do modelu
<img width="1448" height="222" alt="obraz" src="https://github.com/user-attachments/assets/43a57ece-68c5-40aa-a551-7903e0e961cd" />

### 3. Inżynieria cech

* wykorzystanie cech językowych
* cechy statystyczne komentarzy
<img width="923" height="626" alt="obraz" src="https://github.com/user-attachments/assets/3f98ec56-a2e7-4911-a8e9-63888ce598ac" />
Zasadniczą różnicą między ludźmi a botami jest czas odpowiedzi, gdzie u botów większość odpowiedzi odbywa się w czasie niższym niż 5 sekund u ludzi czas odpowiedzi jest wielokrotnie dłuższy. Dodatkową dość wyraźną różnicą jest obecność linków w treści komentarza, w naszym zbiorze bardzo niewielka część prawdziwych użytkowników umieszczała w swoim komentarzu link, komentarze umieszczane przez boty zdecydowanie częściej zawierały linki zewnętrzne. Konta należące do botów są również nowsze. Średnia długóść słów jest również dłuższa w komentarzach umieszczonych przez boty.
<img width="486" height="418" alt="obraz" src="https://github.com/user-attachments/assets/f8273483-82a0-4db0-accf-7affb0267868" />
reply_delay_seconds = -0.76 — najsilniejsza korelacja, ujemna

contains_links = 0.58 — boty częściej wstawiają linki

avg_word_length = 0.56 — boty piszą dłuższymi słowami

account_age_days = -0.41 — konta botów są młodsze

user_karma = -0.12 — prawie brak korelacji

sentiment_score = 0.085 — praktycznie zero
zmienne z wysoką dodatnią korelacją są to: is_bot_flag i avg_word_length, is_bot_flag i contains_links, zmienne z wysoką negatywną korelacją są to: is_bot_flag i reply_delay_in_seconds
<img width="492" height="417" alt="obraz" src="https://github.com/user-attachments/assets/9b38a7ec-bff4-4e7f-b3dd-c0872fb919f7" />
Spearman pokazuje zbliżone wyniki

PCA
<img width="675" height="708" alt="obraz" src="https://github.com/user-attachments/assets/7080ed56-aaea-4eab-a657-b73aa79b6bd5" />
PCA potrafi rozdzielić klasy, choć nie idealnie — co jest zgodne z wynikiem 4 komponentów wyjaśniających pewien procent wariancji wynoszącej 58,17%.

T-SNE
<img width="666" height="661" alt="obraz" src="https://github.com/user-attachments/assets/854c2bb9-6259-4afb-a6ab-aeb0fb7418f7" />
T-SNE pokazuje że klasy nie są dobrze separowalne globalnie. Boty i ludzie mieszają się w tych samych gromadach, prawie każdy klaster zawiera oba kolory. Wykres pokazuje podział na 4 grupy w z bardzo małym przeplataniem się danych. Wynika to najprawdopobniej z syntetyczności otrzymanych danych.


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
