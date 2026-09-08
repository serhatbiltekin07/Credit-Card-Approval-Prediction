# Credit Card Approval Prediction: K-Nearest Neighbors (KNN) & Analitik Veri İşleme

**Yazar:** Biltekin Kurtuluş  
**Eğitim:** Marmara Üniversitesi - Ekonometri  
**Kullanılan Teknolojiler:** Python, Pandas, Scikit-learn, Matplotlib, NumPy

## Proje Özeti
Bu proje, makine öğrenmesi algoritmalarından K-En Yakın Komşu (KNN) kullanılarak müşterilerin kredi kartı onay/red (risk) durumlarının tahmin edilmesini amaçlamaktadır. Projenin ana odağı, ham ve dengesiz bir finansal veri setinin makine öğrenmesi süreçlerine (Data Preprocessing & Feature Engineering) nasıl analitik bir yaklaşımla hazırlanacağıdır.

## Veri Seti ve Ön İşleme (Data Preprocessing) Süreçleri
Veri seti Kaggle üzerinden alınmış olup, proje boyunca aşağıdaki kritik veri ön işleme adımları uygulanmıştır:

* **Kayıp Veri Yönetimi:** Veri setinde %30'a yakın oranda eksik (null) değer barındıran `OCCUPATION_TYPE` sütunu, modele gürültü (noise) katmaması ve boyut lanetinden (curse of dimensionality) kaçınmak amacıyla veri setinden çıkarılmıştır.
* **Özellik Mühendisliği (Feature Engineering):** 
  * Negatif gün cinsinden verilen `DAYS_BIRTH` ve `DAYS_EMPLOYED` verileri mutlak değere alınarak sırasıyla `AGE_YEARS` ve `YEARS_EMPLOYED` formatına dönüştürülmüştür.
  * `DAYS_EMPLOYED` sütununda çalışmayan/emekli kişileri temsil eden `365243` aykırı değeri (outlier) tespit edilmiş ve modelin sapmasını engellemek için `0` olarak düzeltilmiştir.
* **Hedef Değişken (Target) Entegrasyonu:** Müşteri bilgileri (`application_record.csv`) ile borç/gecikme geçmişi (`credit_record.csv`) tabloları `ID` üzerinden birleştirilmiştir. 60 günden fazla gecikmesi olan müşteriler `1 (Riskli)`, diğerleri `0 (Güvenilir)` olarak etiketlenmiştir.
* **Encoding & Feature Selection:** Kredi skoru ile doğrudan ilgisi olmayan iletişim bilgileri (telefon, e-mail) elenmiş; kategorik değişkenler `LabelEncoder` ve One-Hot Encoding (`get_dummies`) kullanılarak çoklu bağlantı (multicollinearity) problemine karşı `drop_first=True` argümanıyla sayısallaştırılmıştır.
* **Ölçeklendirme:** KNN'in mesafe temelli yapısı gereği, maaş ve çocuk sayısı gibi farklı varyanslara sahip değişkenler `StandardScaler` ile standartlaştırılmıştır.

## Model Kurulumu ve Değerlendirme
Veri seti %70 Eğitim ve %30 Test olarak ayrıldıktan sonra `KNeighborsClassifier` ile modellenmiştir.

### Accuracy Paradox (Doğruluk Paradoksu) Analizi
İlk model testlerinde **%98.07** gibi oldukça yüksek bir doğruluk (Accuracy) oranı elde edilmiştir. Ancak Karmaşıklık Matrisi (Confusion Matrix) incelendiğinde bu yüksek oranın veri setindeki dengesizlikten (imbalance) kaynaklandığı tespit edilmiştir:
* Modelin asıl riskli olan "1" sınıfını yakalama (Recall) oranı oldukça düşük kalmıştır. 
* Model, veri setindeki güvenilir müşteri çoğunluğuna uyum sağlayarak (majority class domination) riskli müşterilerin büyük kısmına kredi onay kararı vermiştir (False Negative).

### K Değeri Optimizasyonu (Elbow Method)
Modelin davranışı `K` değerinin 1'den 20'ye kadar değiştirildiği bir döngü ile analiz edilmiş ve görselleştirilmiştir. Analiz sonucunda, K değeri 10'u geçtiğinde modelin tamamen çoğunluk sınıfına uyum sağlayarak %98.3 Baseline Accuracy'de sabitlendiği ve risk yakalama gücünün tamamen kaybolduğu kanıtlanmıştır.

## Sonuç
Bu çalışma; finansal risk analizi gibi dengesiz veri setlerinde sadece "Accuracy" metriğine odaklanmanın iş hedefleri açısından yanıltıcı olacağını açıkça ortaya koymuştur. Verinin temizlenmesi, dönüştürülmesi ve modelin matematiksel zaaflarının (paradokslarının) tespit edilmesi süreci başarıyla tamamlanmıştır.

> **Gelecek Çalışmalar:** Projenin bir sonraki aşamasında, modelin `1 (Riskli)` sınıfını yakalama gücünü artırmak amacıyla veri dengesizliğini çözen **SMOTE** algoritmasının entegre edilmesi planlanmaktadır.
