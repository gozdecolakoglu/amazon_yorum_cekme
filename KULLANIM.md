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

## Bag of Words (BoW) ve TF-IDF çalışması

Aynı dört klasik modeli (`Naive Bayes`, `Lojistik Regresyon`, `Support Vector Machine`,
`Random Forest`) hem Bag of Words (`CountVectorizer`) hem de TF-IDF (`TfidfVectorizer`)
özellikleriyle karşılaştırmak ve en baskın unigram/bigram terimlerin frekans ile
TF-IDF ağırlıklarını çıkarmak için:

```powershell
py -3.13 amazon_review_classification.py --compare-bow-tfidf
```

Komut, aynı %80 eğitim / %20 test bölmesinde dört modelin BoW ve TF-IDF metriklerini
`classification_bow_tfidf_comparison.csv` dosyasına, derlemdeki en yüksek frekanslı
ve en yüksek ortalama TF-IDF ağırlıklı terimleri (sözlük boyutu ve seyreklik
istatistikleriyle birlikte) `bow_tfidf_top_terms.csv` dosyasına kaydeder ve yerel
grafik panelinde (`Bag of Words (BoW) ve TF-IDF çalışması` bölümünde) görselleştirir.

## SMOTE (Sentetik Azınlık Aşırı Örnekleme) çalışması

Eğitim kümesindeki sınıf dengesizliğini (60 olumlu, 14 nötr, 6 olumsuz) Chawla vd.
(2002) SMOTE algoritmasıyla ($k=5$ en yakın komşu) 60 olumlu, 60 nötr ve 60
olumsuz (toplam 180 eğitim örneği, +100 sentetik vektör) olacak biçimde
dengelemek ve dört klasik TF-IDF modelinde (`SMOTE yok` / `SMOTE var`) etkisini
ölçmek için:

```powershell
py -3.13 amazon_review_classification.py --compare-smote
```

Veri sızıntısını önlemek için SMOTE yalnızca eğitim kümesine uygulanır; %20 test
kümesi gerçek dağılımıyla korunur. Sonuçlar `classification_smote_comparison.csv`
dosyasına kaydedilir ve yerel grafik panelinde (`SMOTE (Sentetik Azınlık Aşırı
Örnekleme) çalışması` bölümünde) sınıf dengeleme grafiği, model metrik grafiği,
fark tablosu ve detaylı açıklamalarla gösterilir.

## Eğitim oranı deneyi

Dört klasik TF-IDF modelini %60, %65, %70, %75, %80 ve %85 eğitim oranlarında
karşılaştırmak için:

```powershell
py -3.13 amazon_review_classification.py --compare-training-size
```

Her oran için sınıf dağılımı korunmuş, sabit tohumlu ayrı bir eğitim/test
bölmesi oluşturulur. Eğitim oranı değiştikçe test kümesinin boyutu da değişir;
grafik ve tabloda bu adetler gösterilir. Naive Bayes, lojistik regresyon,
doğrusal SVM ve Random Forest için eğitim/test doğruluğu, macro kesinlik,
duyarlılık, F1 ve ROC-AUC değerleri
`classification_training_size_comparison.csv` dosyasına yazılır. Türkçe BERT
bu deneyde yer almaz.

Sonuçları yerel panele veya Vercel'e aktarmak için ardından dışa aktarma
komutunu çalıştırın:

```powershell
py -3.13 export_dashboard_data.py
```

Metrikler küçük veri kümesinde, her oran için farklı boyuttaki test kümelerinde
ve otomatik sözlük etiketlerine karşı hesaplanır; bu karşılaştırma insan
doğrulamalı gerçek duygu başarımını kanıtlamaz.

## Stemming var/yok metrik karşılaştırması

Stemming'in sınıflandırma sonuçlarına etkisini aynı dört TF-IDF modeli üzerinde
ölç:

```powershell
py -3.13 amazon_review_classification.py --compare-stemming
```

Karşılaştırma `stopword_removed_tokens` (stemming yok) ve `stemmed_tokens`
(stemming var) sütunlarını kullanır. İki girdi de aynı küçük harf, noktalama,
token ve stop-word temizliği aşamalarından geçmiştir; farklı olan yalnızca
stemming adımıdır. Yorumdan sözlükle çıkarılan etiketler, %80 eğitim/%20 test
stratified ayırma ve sabit tohum her iki koşulda aynıdır. Naive Bayes,
lojistik regresyon, doğrusal SVM ve Random Forest için eğitim/test doğruluğu,
macro kesinlik/duyarlılık/F1 ve macro ROC-AUC hesaplanıp
`classification_stemming_comparison.csv` dosyasına kaydedilir.

Bu metrikler mevcut yorum sözlüğünün otomatik etiketlerini tahmin etme
başarısını gösterir; insan etiketli gerçek duygu başarımı değildir. Mevcut
100 yorumda 75 olumlu, 18 nötr, 7 olumsuz etiket vardır; test kümesinde sadece
bir olumsuz yorum bulunması macro metriklerin güvenilirliğini sınırlar.
Karşılaştırma CSV'si mevcutsa yerel panelde `Stemming etkisi: var / yok`
grafiği, ayrıntılı metrikler ve farklar görünür. Yeni veri için karşılaştırma
komutunu yeniden çalıştırıp paneli yenile.

## Makaleye dayalı hedef tabanlı duygu analizi

Koruyan'ın 2025 tarihli çalışması, restoran yorumlarında tek bir genel duygu
etiketi yerine kalite boyutlarını hedef alan çok etiketli sıfır-atış analiz
kullanır. Makalede Türkçe ve farklı dillerdeki yorumlara
`vicgalle/xlm-roberta-large-xnli-anli` modeliyle NLI tabanlı aday sınıfları
uygulanır; yemek kalitesi, servis, fiziksel ortam ve fiyat için olumlu ve
olumsuz sınıflar ayrı puanlanır. Yorumlar üç kelimeden kısaysa analize alınmaz.
Bu projede aynı yaklaşım telefon yorumlarına uyarlanmıştır: cihaz kalitesi ve
tasarım, performans ve pil, ekran ve kamera, satıcı ve teslimat, fiyat ve değer.

İşlenmiş yorum CSV'si üzerinden yeni analizi üret:

```powershell
py -3.13 amazon_zero_shot_sentiment.py
```

Betik her yorum/hedef için hem olumlu hem olumsuz sınıf puanını bağımsız hesaplar;
bu nedenle aynı yorumda birden çok hedef bulunabilir ve bir hedef hem olumlu hem
olumsuz olarak işaretlenebilir. Varsayılan 0,50 eşik, makaledeki eşikle aynıdır:
iki puan da eşiği geçerse hedef “karma”, ikisi de geçmezse hedef belirtilmemiş
kabul edilir. Üç kelimeden kısa yorumlar saklanır fakat hedef puanı üretilmez.
Ham puanlar `iphone15_sentiment_analysis.csv` dosyasına yazılır. Puanlar NLI
modelinin sınıf skorlarıdır; kalibre edilmiş olasılık veya bağımsız performans
ölçümü değildir. Yorumların insan etiketli bir doğrulama kümesi bulunmadığından
bu veri üzerinde accuracy/F1 iddiası yapılmaz.

İlk çalıştırmada modelin yaklaşık 2,24 GB ağırlıkları Hugging Face'ten
indirilir; indirme ve CPU çıkarımı zaman alabilir. Varsayılan batch boyutu 4'tür;
bellek kısıtında küçültülebilir. Girdi, çıktı, model, eşik ve batch boyutu:

```powershell
py -3.13 amazon_zero_shot_sentiment.py --input ".\iphone15_reviews_processed.csv" --output ".\iphone15_sentiment_analysis.csv" --model "vicgalle/xlm-roberta-large-xnli-anli" --threshold 0.5 --batch-size 2
```

Bu yöntem, Koruyan'ın restoran çalışmasının sonuçlarını kopyalamaz: alan,
hedefler ve veri değiştirildiği için bildirilen restoran doğruluk/F1 değerleri
telefon yorumlarına genellenemez. Makale: [Restoran Müşteri Yorumlarının Duygu
Analizi: Sıfır-Atış Metin Sınıflandırma Yaklaşımı, Koruyan (2025),
DOI: 10.38016/jista.1543187](https://dergipark.org.tr/tr/download/article-file/4188972).

## Yerel grafik paneli

İşlenmiş yorum ve model CSV dosyalarındaki verileri tarayıcıda grafiklerle
incelemek için proje klasöründe şu komutu çalıştır:

```powershell
py -3.13 amazon_review_classification.py --compare-bow-tfidf
py -3.13 amazon_review_classification.py --compare-stemming
py -3.13 amazon_review_classification.py --compare-training-size
py -3.13 amazon_zero_shot_sentiment.py
py -3.13 dashboard.py
```

Ardından [http://127.0.0.1:8000](http://127.0.0.1:8000) adresini aç. Panelde
ortalama yıldız puanı, hedef bazında olumlu/olumsuz/karma sonuçlar, puan
dağılımı, aylık yorum sayısı, sözlük tabanlı genel duygu karşılaştırması,
model metrikleri, Bag of Words (BoW) ve TF-IDF çalışması, eğitim oranı ve
stemming karşılaştırmaları ile arama filtresi bulunan yorum listesi yer alır.
Panel verileri yerel CSV'lerden her istekte okur; yorumlar değiştiğinde ilgili
analizleri yeniden çalıştırıp sayfayı yenile. Sunucuyu kapatmak için terminalde
`Ctrl+C` tuşlarına bas.

## Vercel'e dağıtım

Vercel bu projede grafik panelini statik site olarak yayımlar. Yorum çıkarma,
sınıflandırma ve sıfır-atış analizini önce yerel bilgisayarınızda çalıştırın;
Vercel üzerinde Python analiz betikleri çalışmaz. Sonuçları statik JSON'a
aktarmak için proje klasöründe:

```powershell
py -3.13 amazon_review_classification.py --compare-bow-tfidf
py -3.13 amazon_review_classification.py --compare-training-size
py -3.13 export_dashboard_data.py
```

Komut, panelin kullandığı `dashboard-data.json` dosyasını oluşturur. Git tabanlı
dağıtımda bu dosyayı commit edip Vercel'de depoyu içe aktarın. `vercel.json`
Python framework algılamasını kapatıp **Other** seçimini zorlar. Vercel proje
ayarlarında framework hala Python görünüyorsa **Other** seçin; build komutu ve
output directory boş kalabilir. Konfigürasyon ana sayfayı `dashboard.html`
dosyasına, panel veri isteğini de `dashboard-data.json` dosyasına yönlendirir.
Terminalden dağıtmak için Node.js kurulu olmalı; proje klasöründe ilk seferde
`npx vercel`, üretim dağıtımında `npx vercel --prod` komutunu çalıştırın.

Veri veya analizler değiştiğinde analizleri yeniden çalıştırıp dışa aktarma
komutunu tekrarlayın; güncellenen JSON'u yeni dağıtımla yayımlayın. Bu dağıtım
yalnızca sabit bir veri anlık görüntüsüdür, canlı analiz yapmaz. JSON panelde
gösterilen yorum metinlerini ve yorumcu adlarını içerdiğinden, herkese açık
dağıtımdan önce bu verileri yayımlamak istediğinizi doğrulayın.

## İşlem sonrası sonuç

Yorum çıkarma/işleme ve sınıflandırma adımlarını tamamladıktan sonra
`classification_accuracy.csv` dosyasındaki güncel metrikleri inceleyin.
Mevcut proje verisi ve metriklerinden çıkarılan değerlendirme
[Sonuçlar.md](Sonuçlar.md) dosyasındadır. Girdi verisi veya model sonuçları
değiştiğinde bu değerlendirmedeki veri dağılımını ve metrikleri yeni CSV'ye göre
güncelleyin; raporun otomatik olarak oluşturulmadığını dikkate alın.

## Testleri çalıştırma

Proje klasöründe birim testlerini çalıştırın:

```powershell
py -3.13 -X utf8 -m unittest -v
```
