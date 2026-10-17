# Sonuçlar

## Veri ve ölçüm özeti

Çalışmada 100 Amazon iPhone 15 yorumu incelenmiştir. Sözlük tabanlı etiketleme
75 yorumu olumlu, 18 yorumu nötr ve 7 yorumu olumsuz olarak sınıflandırmıştır.
Veri, sınıf oranları korunarak 80 eğitim ve 20 test yorumuna ayrılmıştır.

| Model | Test doğruluğu | Kesinlik (macro) | Duyarlılık (macro) | F1 (macro) | ROC-AUC (macro) |
|---|---:|---:|---:|---:|---:|
| Naive Bayes | 0,75 | 0,25 | 0,33 | 0,29 | 0,78 |
| Lojistik Regresyon | 0,75 | 0,25 | 0,33 | 0,29 | 0,93 |
| Support Vector Machine | 0,75 | 0,25 | 0,33 | 0,29 | 0,94 |
| Random Forest | 0,85 | 0,61 | 0,50 | 0,53 | 0,77 |
| Türkçe BERT | 0,75 | 0,25 | 0,33 | 0,29 | 0,68 |

## Değerlendirme

Bu çalışmada kaydedilmiş 100 Amazon iPhone 15 yorumu incelenmiştir. Yorumların duygu etiketleri yıldız puanlarından değil, metindeki olumlu ve olumsuz sözcük köklerinden türetilmiştir. Sonuçta 75 yorum olumlu, 18 yorum nötr ve 7 yorum olumsuz etiketlenmiştir. Olumlu sınıfın baskın olması, veri kümesinin dengesiz olduğunu göstermektedir. Veriler sınıf oranları korunarak eğitim ve test kümelerine ayrılmış, beş sınıflandırıcı aynı test kümesinde değerlendirilmiştir. Random Forest, yüzde 85 doğruluk, 0,61 macro kesinlik, 0,50 macro duyarlılık ve 0,53 macro F1 ile tahmin başarısı bakımından diğer modellerin önüne geçmiştir. Buna karşılık en yüksek ROC-AUC değeri 0,94 ile Support Vector Machine modelinde görülmüştür; bu, eşiklerden bağımsız sıralama başarısının iyi olabileceğini, fakat seçilen sınıf tahminlerinin aynı ölçüde başarılı olmadığını düşündürür. Doğruluk tek başına yeterli değildir: çoğunluk sınıfının oranı yüzde 75 olduğundan, yüzde 75 doğruluk veren modeller azınlık sınıflarını yakalamakta başarısız olabilir. Test kümesinde yalnızca bir olumsuz yorum bulunması, bu sınıfa ilişkin ölçümlerin kararlılığını ciddi biçimde sınırlar. Ayrıca sözlük etiketleri insan doğrulamalı gerçekler değil, otomatik ve yaklaşık hedeflerdir. Bu nedenle sonuçlar ürün memnuniyetinin kesin ölçümü veya genel kullanıcı kitlesine genellenebilir kanıt olarak yorumlanmamalıdır. Daha güvenilir değerlendirme için daha geniş, dengeli ve uzmanlarca etiketlenmiş yorumlar toplanmalı; sınıf bazlı metrikler ve çapraz doğrulama birlikte raporlanmalıdır. Bu nedenle bulgular ön değerlendirme olarak ele alınmalıdır.
