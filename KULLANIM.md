# Kullanım Kılavuzu

Bu kılavuz, Amazon iPhone 15 yorumlarını çıkarma/işleme ve işlenmiş
yorumlarla duygu sınıflandırması adımlarını açıklar.

Betik varsayılan olarak
`%USERPROFILE%\Desktop\amazon_yorumlar_files\amazon_yorumları_cekilenler.csv`
dosyasından iPhone 15 yorumlarını alır ve `iphone15_reviews_processed.csv`
dosyasına yazar. CSV'de ürün/model sütunu bulunursa yalnızca iPhone 15 satırları
işlenir; yoksa kaynak CSV'nin önceden iPhone 15 yorumlarına filtrelenmiş olması
gerekir. Kaydedilmiş Amazon HTML dosyaları veya bunları içeren klasör de girdi
olarak kullanılabilir; bu durumda yorumlar BeautifulSoup ile ayrıştırılır.

PowerShell terminalinde önce proje klasörüne geç:

```powershell
cd "C:\karışık\amazon_yorum_cekme"
```

Bu bilgisayarda `python` komutu PATH'te bulunmadığından komutlarda Windows
Python başlatıcısı `py -3.13` kullanılıyor.

Kurulum:

```powershell
py -3.13 -m pip install -r requirements.txt
```

Çalıştırma:

```powershell
py -3.13 amazon_iphone15_reviews.py
```

Kaynak CSV'yi/dosyayı/klasörü veya çıktı dosyasını belirtebilirsin:

```powershell
py -3.13 amazon_iphone15_reviews.py --input "C:\veri\amazon_yorumları_cekilenler.csv" --output ".\sonuclar\iphone15.csv"
py -3.13 amazon_iphone15_reviews.py --input "C:\veri\amazon_html" --output ".\sonuclar\iphone15.csv"
```

CSV'de ham yorumun yanı sıra küçük harfe dönüştürme, noktalama temizleme,
NLTK tokenization, stop-word temizleme, TurkishStemmer kökleri, tek karakterli
tokenların elenmesi ve Zemberek lemmatization sonuçları ayrı sütunlarda yer alır.
Türkçe NLTK stop-word korpusu kurulu değilse betik kendi Türkçe stop-word listesi
ile devam eder ve bunu günlükte bildirir.

Varsayılan girdi yolu betikte tanımlı bilgisayar yoludur. Bu yol farklıysa kendi
veri dosyanızı `--input` ile verin. Çıktı yolu varsayılan olarak proje klasöründe
`iphone15_reviews_processed.csv` olur.

HTML kullanılırsa Amazon yorumlarının kendisini içeren kaydedilmiş `.html` veya
`.htm` sayfası bulunmalıdır; sayfa kaynaklarının yanındaki JavaScript/CSS
dosyaları tek başına yorum metnini içermez.

## Yorum duygu sınıflandırması

Duygu etiketleri `rating` puanlarından değil, `review` yorum metnindeki olumlu ve
olumsuz kök başlangıçlarından üretilir. Türkçe harfler ASCII karşılıklarına
çevrilir; "beğendim" gibi çekimli sözcükler kök önekleriyle eşleştirilir.
"Güzel değil" ve "sorun yok" gibi kullanımlarda olumsuzluk sözcüğü önceki duygu
ipucunun kutbunu tersine çevirir. Olumlu ve olumsuz ipucu bulunmayan veya iki
kutbun eşit olduğu yorumlar **nötr** sayılır. Bu otomatik sözlük etiketleri
insan tarafından doğrulanmış gerçek etiketler değil, zayıf etiketlerdir;
modellerin doğruluğu bu sözlükle üretilen etiketleri tahmin etme başarısını gösterir.

Veri, sınıf oranlarını koruyan sabit tohumlu %80 eğitim / %20 test kümelerine
ayrılır. Naive Bayes, lojistik regresyon, doğrusal SVM ve Random Forest
TF-IDF özellikleriyle eğitilir. Türkçe BERT (`dbmdz/bert-base-turkish-cased`)
ise eğitim kümesinde ince ayar yapar. Her modelin eğitim doğruluğu ve test
kümesindeki doğruluk, macro kesinlik (precision), macro duyarlılık (recall),
macro F1 ve macro one-vs-rest ROC-AUC değerleri terminalde gösterilir ve
`classification_accuracy.csv` dosyasına yazılır. CSV sütunları `model`,
`train_accuracy`, `test_accuracy`, `test_precision`, `test_recall`, `test_f1`
ve `test_roc_auc` şeklindedir. ROC-AUC her sınıf için one-vs-rest hesaplanır,
sonuç sınıflar üzerinden macro ortalamadır. Değerler 0–1 arası saklanır
(ör. `0.75`, yani %75). Sınıf adetleri ve yüzdeleri CSV'ye eklenmez.

Mevcut 100 yorumluk veri kümesinde, sabit tohumlu bölmeden sonra eğitim
kümesinde 60 olumlu (%75), 14 nötr (%17,5), 6 olumsuz (%7,5); test kümesinde
15 olumlu (%75), 4 nötr (%20), 1 olumsuz (%5) yorum vardır. Girdi verisi
değiştiğinde bu sayılar da değişir.

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 amazon_review_classification.py
```

Girdi ve sonuç dosyalarını belirlemek veya farklı bir Hugging Face Türkçe BERT
modeli kullanmak için:

```powershell
py -3.13 amazon_review_classification.py --input ".\iphone15_reviews_processed.csv" --output ".\classification_accuracy.csv" --bert-model "dbmdz/bert-base-turkish-cased"
```

İlk BERT çalıştırmasında model dosyalarının Hugging Face'ten indirilmesi gerekir.
Veri kümesinde yalnızca 100 yorum ve dengesiz sınıflar bulunduğundan ölçümler
küçük test kümesinde hesaplanır; sonuçlar bu sınırlılıkla yorumlanmalıdır.

## İşlem sonrası sonuç

Yorum çıkarma/işleme ve sınıflandırma adımlarını tamamladıktan sonra
`classification_accuracy.csv` dosyasındaki güncel metrikleri inceleyin.
Mevcut proje verisi ve metriklerinden çıkarılan 200 kelimelik değerlendirme
[Sonuçlar.md](Sonuçlar.md) dosyasındadır. Girdi verisi veya model sonuçları
değiştiğinde bu değerlendirmedeki veri dağılımını ve metrikleri yeni CSV'ye göre
güncelleyin; raporun otomatik olarak oluşturulmadığını dikkate alın.

## Testleri çalıştırma

Proje klasöründe birim testlerini çalıştırın:

```powershell
py -3.13 -X utf8 -m unittest -v
```
