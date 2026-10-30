# Sonuçlar

## SMOTE (Sentetik Azınlık Aşırı Örnekleme) çalışması

100 Amazon iPhone 15 yorumundan türetilen sözlük etiketlerinde belirgin bir
sınıf dengesizliği vardır: toplamda **75 olumlu (%75)**, **18 nötr (%18)** ve
**7 olumsuz (%7)** yorum bulunur. Sınıf oranlarını koruyan %80 eğitim / %20 test
bölmesinde eğitim kümesi **60 olumlu, 14 nötr ve 6 olumsuz** (toplam 80 yorum;
10:1 çoğunluk/azınlık oranı), bağımsız test kümesi ise **15 olumlu, 4 nötr ve 1
olumsuz** (toplam 20 yorum) örneğe sahiptir.

Veri sızıntısını (data leakage) önlemek amacıyla **SMOTE (Synthetic Minority
Over-sampling Technique, $k=5$ komşu)** yalnızca eğitim kümesinin TF-IDF özellik
uzayına uygulanmıştır:
- **SMOTE Öncesi Eğitim Kümesi (80 örnek):** 60 olumlu (%75,0), 14 nötr (%17,5), 6 olumsuz (%7,5)
- **SMOTE Sonrası Eğitim Kümesi (180 örnek):** 60 olumlu (%33,3 · +0 sentetik), 60 nötr (%33,3 · +46 sentetik), 60 olumsuz (%33,3 · +54 sentetik) — toplam **+100 sentetik azınlık örneği**
- **Bağımsız Test Kümesi (20 örnek · dokunulmadı):** 15 olumlu (%75,0), 4 nötr (%20,0), 1 olumsuz (%5,0)

| Model | SMOTE Durumu | Eğitim doğruluğu | Test doğruluğu | Kesinlik (macro) | Duyarlılık (macro) | F1 (macro) | ROC-AUC (macro) |
|---|---|---:|---:|---:|---:|---:|---:|
| Naive Bayes | SMOTE yok | 0,76 | 0,75 | 0,25 | 0,33 | 0,29 | 0,78 |
| Naive Bayes | SMOTE var | 1,00 | **0,80** | **0,73** | **0,79** | **0,68** | **0,92** |
| Lojistik Regresyon | SMOTE yok | 0,75 | 0,75 | 0,25 | 0,33 | 0,29 | 0,93 |
| Lojistik Regresyon | SMOTE var | 0,97 | 0,75 | 0,25 | 0,33 | 0,29 | 0,93 |
| Support Vector Machine | SMOTE yok | 0,99 | 0,75 | 0,25 | 0,33 | 0,29 | 0,94 |
| Support Vector Machine | SMOTE var | 1,00 | **0,85** | **0,61** | **0,50** | **0,53** | 0,93 |
| Random Forest | SMOTE yok | 1,00 | 0,85 | 0,61 | 0,50 | 0,53 | 0,77 |
| Random Forest | SMOTE var | 1,00 | 0,85 | 0,61 | 0,50 | 0,53 | **0,87** |

### SMOTE Değerlendirmesi

- **Naive Bayes:** SMOTE öncesinde %75'lik çoğunluk sınıfı önsel olasılığı (prior) nedeniyle test kümesindeki 20 yorumun tamamını olumlu tahmin ederken, SMOTE ile sınıflar eşitlendiğinde azınlık sınıflarını yakalamaya başlamış; macro F1 skoru **%28,6'dan %67,8'e (+39,2 puan)**, macro duyarlılığı **%33,3'ten %78,9'a (+45,6 puan)**, test doğruluğu **%75,0'dan %80,0'a (+5,0 puan)** ve ROC-AUC değeri **%77,7'den %92,2'ye (+14,5 puan)** yükselmiştir.
- **Support Vector Machine (LinearSVC):** Sentetik azınlık örnekleri karar sınırını çoğunluk sınıfından uzaklaştırarak test doğruluğunu **%75,0'dan %85,0'a (+10,0 puan)** ve macro F1 skorunu **%28,6'dan %52,5'e (+24,0 puan)** çıkarmıştır.
- **Random Forest:** Halihazırda `class_weight="balanced_subsample"` kullandığından %85,0 test doğruluğu ve %52,5 macro F1 değerini korumuş, buna ek olarak macro ROC-AUC sıralama başarısını **%76,6'dan %87,4'e (+10,7 puan)** taşımıştır.

## Bag of Words (BoW) ve TF-IDF karşılaştırma çalışması

100 Amazon iPhone 15 yorumu üzerinde `stopword_removed_tokens` aşamasından sonra
unigram ve bigram (`ngram_range=(1, 2)`) sözlüğü çıkarılmıştır. Toplam sözlük
boyutu **3.548 benzersiz terim** (1.333 tekil kelime / unigram ve 2.215 ikili
ifade / bigram) olup, özellik matrisinin seyrekliği (sparsity) **%98,69**
(4.658 sıfırdan farklı hücre) olarak ölçülmüştür.

- **Terim Ağırlıklandırma Farkı:** Bag of Words (`CountVectorizer`) ham geçiş
  sayısını kullandığından `amazon` (37 kez, 27 yorum), `iphone` (31 kez, 22
  yorum), `telefon` (25 kez, 22 yorum), `aldım` (24 kez) ve `yok` (24 kez) en
  baskın terimler olmuştur. TF-IDF (`TfidfVectorizer`, `sublinear_tf=True`) ise
  yaygın kelimeleri ters belge sıklığı (IDF) ile baskılayıp L2 normalizasyonu
  uyguladığı için `teşekkürler` (11 kez geçmesine karşın 0,0237 ortalama TF-IDF
  ile 2. sırada), `iyi` (0,0222 ile 3. sırada), `hızlı` (0,0211 ile 4. sırada)
  ve `kalitesi` (0,0168) gibi duygu/değerlendirme terimlerini öne çıkarmıştır.

Aynı sabit tohumlu, sınıf oranlarını koruyan %80 eğitim / %20 test bölmesinde
dört klasik modelin Bag of Words ve TF-IDF sonuçları:

| Model | Vektörleştirme | Eğitim doğruluğu | Test doğruluğu | Kesinlik (macro) | Duyarlılık (macro) | F1 (macro) | ROC-AUC (macro) |
|---|---|---:|---:|---:|---:|---:|---:|
| Naive Bayes | Bag of Words (BoW) | 0,98 | 0,75 | 0,25 | 0,33 | 0,29 | 0,81 |
| Naive Bayes | TF-IDF | 0,76 | 0,75 | 0,25 | 0,33 | 0,29 | 0,78 |
| Lojistik Regresyon | Bag of Words (BoW) | 0,99 | 0,75 | 0,25 | 0,33 | 0,29 | 0,91 |
| Lojistik Regresyon | TF-IDF | 0,75 | 0,75 | 0,25 | 0,33 | 0,29 | 0,93 |
| Support Vector Machine | Bag of Words (BoW) | 1,00 | 0,85 | 0,61 | 0,50 | 0,53 | 0,89 |
| Support Vector Machine | TF-IDF | 0,99 | 0,75 | 0,25 | 0,33 | 0,29 | 0,94 |
| Random Forest | Bag of Words (BoW) | 1,00 | 0,80 | 0,50 | 0,48 | 0,48 | 0,81 |
| Random Forest | TF-IDF | 1,00 | 0,85 | 0,61 | 0,50 | 0,53 | 0,77 |

## Stemming var/yok metrik karşılaştırması

Projede 100 Amazon iPhone 15 yorumu vardır. Etiketler yıldız puanından değil,
yorum metnindeki olumlu/olumsuz sözlük işaretlerinden çıkarılmıştır:
75 olumlu, 18 nötr, 7 olumsuz. Her koşulda aynı sabit tohumlu, sınıf oranlarını
koruyan %80 eğitim/%20 test bölmesi kullanılmıştır. Her iki koşul da
`stopword_removed_tokens` ile aynı küçük harf, token ve stop-word temizliğini
kullanır; **stemming yok** koşulunda bu metin olduğu gibi, **stemming var**
koşulunda aynı tokenların Türkçe kökleri kullanılır. Böylece denenen değişken
stemming adımıdır.

| Model | İşlem | Test doğruluğu | Kesinlik (macro) | Duyarlılık (macro) | F1 (macro) | ROC-AUC (macro) |
|---|---|---:|---:|---:|---:|---:|
| Naive Bayes | Stemming yok | 0,75 | 0,25 | 0,33 | 0,29 | 0,85 |
| Naive Bayes | Stemming var | 0,75 | 0,25 | 0,33 | 0,29 | 0,91 |
| Lojistik Regresyon | Stemming yok | 0,75 | 0,25 | 0,33 | 0,29 | 0,89 |
| Lojistik Regresyon | Stemming var | 0,75 | 0,25 | 0,33 | 0,29 | 0,91 |
| Support Vector Machine | Stemming yok | 0,75 | 0,25 | 0,33 | 0,29 | 0,84 |
| Support Vector Machine | Stemming var | 0,75 | 0,25 | 0,33 | 0,29 | 0,92 |
| Random Forest | Stemming yok | 0,90 | 0,63 | 0,58 | 0,60 | 0,72 |
| Random Forest | Stemming var | 0,90 | 0,63 | 0,58 | 0,60 | 0,74 |

### Değerlendirme

Bu sabit bölmede stemming dört modelin hiçbirinde test doğruluğunu veya macro
F1 değerini değiştirmemiştir. Naive Bayes, lojistik regresyon ve doğrusal SVM
için macro F1 0,29'da kalmış; Random Forest her iki koşulda 0,60 macro F1 ve
0,90 doğruluk elde etmiştir. Stemming ile macro ROC-AUC Naive Bayes'te 0,85'ten
0,91'e, lojistik regresyonda 0,89'dan 0,91'e, SVM'de 0,84'ten 0,92'ye ve
Random Forest'ta 0,72'den 0,74'e çıkmıştır. Bu veri bölmesinde stemming'in
gözlenen katkısı sınıf sıralama metriğinde görünmüş, seçilen sınıf tahminlerinin
doğruluk ve F1'ine yansımamıştır.

Test kümesinde 20 yorum ve yalnızca bir olumsuz örnek bulunur. Ayrıca hedef
etiketleri otomatik sözlükten geldiğinden bunlar insan doğrulamalı gerçek
duygu metrikleri değildir; küçük ve dengesiz test kümesi sonuçları hassas
kılar. Bulgular, stemming'in genel olarak daha iyi olduğu veya üretim
başarısını artıracağı şeklinde yorumlanmamalıdır. Çıktının tamamı
`classification_stemming_comparison.csv` dosyasındadır ve yerel panelde
görselleştirilir.

## Hedef tabanlı sıfır-atış analizi

Önceki analizde `vicgalle/xlm-roberta-large-xnli-anli` modeli beş iPhone hedefi
üzerinde kullanılmıştır. Üç kelimeden kısa 9 yorum dışarıda, 91 yorum analiz
edilmiştir. Her hedefe ait bağımsız olumlu ve olumsuz puanlardan:

| Hedef | Olumlu | Olumsuz | Karma | Hedef belirtilmedi |
|---|---:|---:|---:|---:|
| Cihaz kalitesi ve tasarım | 59 | 17 | 10 | 5 |
| Performans ve pil | 53 | 15 | 12 | 11 |
| Ekran ve kamera | 49 | 19 | 11 | 12 |
| Satıcı ve teslimat | 54 | 20 | 12 | 5 |
| Fiyat ve değer | 50 | 14 | 27 | 0 |

Bir yorum birden fazla hedefte sayılabildiğinden hedef sütunları toplamı
benzersiz yorum sayısı değildir. NLI puanları kalibre edilmiş olasılık veya
doğruluk değildir; model sonuçları elle etiketlenmiş bir örnek üzerinde
doğrulanmamıştır. Özellikle fiyat/değer başlığında hiç “hedef belirtilmedi”
sonucu çıkmaması, yanlış hedef eşleştirme ihtimaline dikkat edilmesi gerektiğini
gösterir.

Makalede kullanılan yöntem kaynağı: Koruyan, K. (2025). [Restoran Müşteri
Yorumlarının Duygu Analizi: Sıfır-Atış Metin Sınıflandırma Yaklaşımı](https://doi.org/10.38016/jista.1543187).
_Journal of Intelligent Systems: Theory and Applications_, 8(1), 47–62.
