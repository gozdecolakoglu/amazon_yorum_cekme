# Sonuçlar

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
