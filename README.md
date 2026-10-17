# Amazon iPhone 15 Yorum Analizi

Bu proje kaydedilmiş Amazon yorumlarını çıkarır, Türkçe metin ön işleme
aşamalarını CSV'ye kaydeder ve yorumların içeriğinden duygu sınıflandırması
yapar. Sınıflandırmada yıldız puanı etiket olarak kullanılmaz.

## Gereksinimler

- Python 3.13
- `requirements.txt` içindeki paketler
- Türkçe BERT modeli ilk kullanımda Hugging Face'ten indirilir

PowerShell'de proje klasörüne geçip bağımlılıkları kurun:

```powershell
cd "C:\karışık\amazon_yorum_cekme"
py -3.13 -m pip install -r requirements.txt
```

## 1. Yorumları çıkarma ve işleme

Varsayılan olarak betik
`%USERPROFILE%\Desktop\amazon_yorumlar_files\amazon_yorumları_cekilenler.csv`
dosyasını okur ve proje klasöründe `iphone15_reviews_processed.csv` oluşturur:

```powershell
py -3.13 amazon_iphone15_reviews.py
```

Farklı bir CSV, kaydedilmiş HTML dosyası veya HTML klasörü ve çıktı yolu
belirtilebilir:

```powershell
py -3.13 amazon_iphone15_reviews.py --input "C:\veri\amazon_yorumları_cekilenler.csv" --output ".\iphone15_reviews_processed.csv"
py -3.13 amazon_iphone15_reviews.py --input "C:\veri\amazon_html" --output ".\iphone15_reviews_processed.csv"
```

CSV'de ürün/model sütunu varsa yalnızca temel iPhone 15 yorumları seçilir;
karışık ürün verisinde bu sütun yoksa kaynak CSV önceden filtrelenmiş olmalıdır.
HTML girdisi Amazon yorumlarının içeriklerini barındıran `.html` veya `.htm`
sayfası olmalıdır. İşlenmiş CSV; yorum, puan ve kimlik gibi alanların yanında
küçük harf, token, stop-word temizliği, kök bulma ve lemmatization aşamalarını
da içerir.

## 2. Duygu sınıflandırması

Sınıflandırıcıları eğitmek için:

```powershell
py -3.13 amazon_review_classification.py
```

Etiketler yorum metnindeki olumlu/olumsuz kök öneklerinden çıkarılır. Türkçe
harfler ASCII karşılıklarına çevrilir; örneğin çekimli sözcükler sözlükteki
kök önekleriyle eşleştirilir. “Güzel değil” gibi ifadelerde olumsuzluk
sözcüğü duygu işaretini tersine çevirir. Olumlu veya olumsuz işaret
bulunmayan ya da iki tür işaretin eşit olduğu yorumlar nötr kabul edilir.
Bu etiketler otomatik sözlük etiketleridir; doğruluk puanları insan tarafından
doğrulanmış etiketlere göre değil, sözlük etiketlerine göre hesaplanır.

Veri, sabit tohumla ve sınıf oranları korunarak %80 eğitim / %20 test
kümelerine ayrılır. Naive Bayes, lojistik regresyon, doğrusal SVM ve Random
Forest TF-IDF özellikleriyle eğitilir; Türkçe BERT
(`dbmdz/bert-base-turkish-cased`) ince ayar yapılarak değerlendirilir. Eğitim
ve test doğrulukları `classification_accuracy.csv` dosyasına yazılır.
Bu sonuç dosyasında `model`, `train_accuracy` ve `test_accuracy` sütunları
bulunur; doğruluk değerleri 0–1 aralığında kaydedilir. İşlenmiş yorumlar
`iphone15_reviews_processed.csv` dosyasındadır.

Girdi, çıktı veya BERT modelini değiştirmek için:

```powershell
py -3.13 amazon_review_classification.py --input ".\iphone15_reviews_processed.csv" --output ".\classification_accuracy.csv" --bert-model "dbmdz/bert-base-turkish-cased"
```

## Testler

```powershell
py -3.13 -X utf8 -m unittest -v
```

Daha ayrıntılı parametreler, girdiler ve çıktı açıklamaları için
[KULLANIM.md](KULLANIM.md) dosyasına bakın.
