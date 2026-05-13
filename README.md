# PathVQA-TR

Patoloji görüntüleri üzerinde Türkçe destekli medikal soru-cevap sistemi.

## Mimari

```
Kullanici goruntü + Türkçe soru
        |
        v
[Helsinki-NLP TR->EN]   [ViT-B/16 encoder (freeze)]
        |                        |
        v                        v
   EN soru vektörü        Görsel feature vektörü
        \                       /
         \                     /
          v                   v
          [Concat + MLP Fusion]  <- sadece bu katman eğitilir
                  |
                  v
          yes / no classifier
                  |
                  v
     [Gemini 1.5 Flash API]
     "VQA sonucu + kullanıcı sorusu -> Türkçe açıklama"
                  |
                  v
          Kullanıcıya cevap
```

Encoder'lar freeze tutulur. Sadece MLP fusion eğitilir.
Bu sayede RTX 4060 (8GB) ve Kaggle T4 üzerinde 1-2 saatte eğitim tamamlanır.

## Klasör Yapısı

```
pathvqa-tr/
├── model/
│   ├── train.py          # Kaggle/Colab üzerinde çalıştır
│   ├── dataset.py        # PathVQA yükleme ve filtreleme
│   └── model.py          # ViT + BERT + MLP mimarisi
├── backend/
│   ├── main.py           # FastAPI uygulama
│   ├── inference.py      # Model yükleme ve tahmin
│   └── gemini.py         # Gemini 1.5 Flash entegrasyonu
├── frontend/
│   └── index.html        # Tek dosya web arayüzü
├── notebooks/
│   └── train_kaggle.ipynb
├── requirements.txt
└── README.md
```

## Kurulum

```bash
pip install -r requirements.txt
```

`.env` dosyası oluştur:
```
GEMINI_API_KEY=your_key_here
```

Google AI Studio üzerinden ücretsiz key: https://aistudio.google.com

## Eğitim

Kaggle üzerinde (önerilen, ücretsiz T4 GPU):
- `notebooks/train_kaggle.ipynb` dosyasını Kaggle'a yükle
- GPU T4 x2 seç
- Tümünü çalıştır, model `best_model_last.pth` olarak kaydedilir
- Gerekli pth dosyasına https://www.kaggle.com/code/fatihmehmetldr/vqa-tr-patoloji-g-r-nt-analizi/output?scriptVersionId=318969097 linkinden output kısmından ulaşabilirsiniz. Pth dosyası sistemin çalışması için zorunludur. İndirdikten sonra projenin bulunduğu dizine aktarıp aşağıdaki işlemleri yapınız.

Kendi makinende (RTX 4060):
```bash
python model/train.py
```

## Çalıştırma

```bash
uvicorn backend.main:app --reload
```

Tarayıcıda `frontend/index.html` dosyasını aç veya `http://localhost:8000` adresine git.

## Veri Seti

PathVQA — flaviagiammarino/path-vqa (HuggingFace)
- Toplam: 32,799 soru-cevap çifti, 4,998 patoloji görüntüsü
- Kullanılan subset: yes/no soruları (~21,000 örnek)
- Split: train / val / test hazır gelir

## Katkı ve Özgünlük

- PathVQA üzerine Türkçe dil desteği literatürde ilk (TM-PathVQA yalnızca EN/DE/FR)
- Medikal VQA modelini LLM destekli chatbot'a entegre eden prototip
- End-to-end web arayüzü

## Sorumluluk

Bu sistem eğitim amaçlıdır. Tıbbi teşhis yerine geçmez.

