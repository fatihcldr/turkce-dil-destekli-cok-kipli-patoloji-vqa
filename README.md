<div align="center">

# PathVQA-TR

**Turkish-Language Visual Question Answering for Pathology Images**
**Patoloji Görüntüleri için Türkçe Destekli Görsel Soru Cevaplama Sistemi**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-Model-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](#)
[![HuggingFace](https://img.shields.io/badge/Dataset-PathVQA-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/flaviagiammarino/path-vqa)

[English](#english) · [Türkçe](#türkçe)

</div>

---

## English

### Overview

PathVQA-TR is a multimodal Visual Question Answering (VQA) system that answers **Turkish-language** natural-language questions about pathology images. To the best of the author's knowledge, this is the first system to add Turkish-language support on top of the PathVQA benchmark (prior multilingual work, e.g. TM-PathVQA, covers only EN/DE/FR).

The pipeline translates the incoming Turkish question to English, encodes the image with a frozen Vision Transformer, fuses both representations through a lightweight trainable MLP head to produce a yes/no classification, and finally generates a natural Turkish explanation of the answer using a sequence-to-sequence language model.

### Architecture

```
User image + Turkish question
        |
        v
[Helsinki-NLP TR→EN]      [ViT-B/16 encoder (frozen)]
        |                          |
        v                          v
  EN question embedding      Visual feature vector
        \                          /
         \                        /
          v                      v
          [Concat + MLP fusion head]   <- only this layer is trained
                     |
                     v
             yes / no classifier
                     |
                     v
                mT5-small
   "VQA result + original question → Turkish explanation"
                     |
                     v
              Final answer to user
```

Both encoders (translation model and ViT) are kept frozen; only the fusion MLP head is trained from scratch, followed by an optional fine-tuning stage.

### Repository Structure

```
pathvqa-tr/
├── model/
│   ├── train.py          # Training entry point (run on Kaggle/Colab)
│   ├── dataset.py        # PathVQA loading & filtering
│   └── model.py          # ViT + BERT + MLP architecture
├── backend/
│   ├── main.py           # FastAPI application
│   ├── inference.py      # Model loading & prediction
│   └── nlg.py            # Gemini 1.5 Flash integration for explanations
├── frontend/
│   └── index.html        # Single-file web UI
├── notebooks/
│   └── train_kaggle.ipynb
├── requirements.txt
└── README.md
```

### Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file with the required API keys (e.g. Gemini API key for the explanation module).

### Training

**Kaggle (recommended — free T4 GPU):**
1. Upload `notebooks/train_kaggle.ipynb` to Kaggle.
2. Select GPU **T4 x2** as the accelerator.
3. Run all cells; the trained model is saved as `best_model_last.pth`.
4. Alternatively, download the pretrained checkpoint from the [Kaggle output page](https://www.kaggle.com/code/fatihmehmetldr/vqa-tr-patoloji-g-r-nt-analizi/output?scriptVersionId=318969097). This `.pth` file is required for the system to run — place it in the project root before continuing.

**Local (tested on RTX 4060):**
```bash
python model/train.py
```

### Running the App

```bash
uvicorn backend.main:app --reload
```

Then open `frontend/index.html` in your browser, or visit `http://localhost:8000`.

### Dataset

[PathVQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa) (HuggingFace)
- 32,799 question–answer pairs across 4,998 pathology images
- Subset used here: yes/no questions (~21,000 samples)
- Train / validation / test splits provided by the original dataset

### Contribution & Novelty

- First Turkish-language extension of the PathVQA benchmark (prior multilingual work covers only EN/DE/FR).
- Integrates a medical VQA classifier with an LLM-based explanation layer (chatbot-style output).
- Ships as a complete, end-to-end web application rather than a notebook-only prototype.

### Disclaimer

This project is intended for **educational and research purposes only** and must not be used as a substitute for professional medical diagnosis.

---

## Türkçe

### Genel Bakış

PathVQA-TR, patoloji görüntüleri üzerinde **Türkçe** doğal dil sorularını yanıtlayabilen çok kipli (multimodal) bir Görsel Soru Cevaplama (VQA) sistemidir. Yazarın bilgisi dahilinde, PathVQA veri seti üzerine Türkçe dil desteği ekleyen ilk çalışmadır (literatürdeki benzer çok dilli çalışmalar — ör. TM-PathVQA — yalnızca İngilizce/Almanca/Fransızca desteklemektedir).

Sistem, gelen Türkçe soruyu İngilizceye çevirir, görüntüyü dondurulmuş (frozen) bir Vision Transformer ile kodlar, iki temsili hafif bir eğitilebilir MLP katmanında birleştirerek evet/hayır sınıflandırması yapar ve son olarak bir sequence-to-sequence dil modeliyle sonucu doğal bir Türkçe açıklamaya dönüştürür.

### Mimari

```
Kullanıcı görüntüsü + Türkçe soru
        |
        v
[Helsinki-NLP TR→EN]      [ViT-B/16 encoder (dondurulmuş)]
        |                          |
        v                          v
  EN soru vektörü            Görsel öznitelik vektörü
        \                          /
         \                        /
          v                      v
          [Concat + MLP Fusion]   <- yalnızca bu katman eğitilir
                     |
                     v
             yes / no sınıflandırıcı
                     |
                     v
                mT5-small
   "VQA sonucu + kullanıcı sorusu → Türkçe açıklama"
                     |
                     v
              Kullanıcıya nihai cevap
```

Her iki encoder da (çeviri modeli ve ViT) dondurulmuş halde tutulur; yalnızca fusion MLP katmanı sıfırdan eğitilir, ardından isteğe bağlı bir fine-tuning aşaması uygulanır.

### Klasör Yapısı

```
pathvqa-tr/
├── model/
│   ├── train.py          # Eğitim başlangıç noktası (Kaggle/Colab üzerinde çalıştırın)
│   ├── dataset.py        # PathVQA yükleme ve filtreleme
│   └── model.py          # ViT + BERT + MLP mimarisi
├── backend/
│   ├── main.py           # FastAPI uygulaması
│   ├── inference.py      # Model yükleme ve tahmin
│   └── nlg.py            # Açıklama modülü için Gemini 1.5 Flash entegrasyonu
├── frontend/
│   └── index.html        # Tek dosyalık web arayüzü
├── notebooks/
│   └── train_kaggle.ipynb
├── requirements.txt
└── README.md
```

### Kurulum

```bash
pip install -r requirements.txt
```

Gerekli API anahtarlarını (ör. açıklama modülü için Gemini API anahtarı) içeren bir `.env` dosyası oluşturun.

### Eğitim

**Kaggle üzerinde (önerilir — ücretsiz T4 GPU):**
1. `notebooks/train_kaggle.ipynb` dosyasını Kaggle'a yükleyin.
2. Hızlandırıcı olarak **T4 x2** GPU seçin.
3. Tüm hücreleri çalıştırın; eğitilen model `best_model_last.pth` olarak kaydedilir.
4. Alternatif olarak, önceden eğitilmiş checkpoint dosyasını [Kaggle çıktı sayfasından](https://www.kaggle.com/code/fatihmehmetldr/vqa-tr-patoloji-g-r-nt-analizi/output?scriptVersionId=318969097) indirebilirsiniz. Sistemin çalışması için bu `.pth` dosyası zorunludur — devam etmeden önce proje kök dizinine yerleştirin.

**Yerel makinede (RTX 4060 üzerinde test edilmiştir):**
```bash
python model/train.py
```

### Çalıştırma

```bash
uvicorn backend.main:app --reload
```

Ardından tarayıcıda `frontend/index.html` dosyasını açın veya `http://localhost:8000` adresine gidin.

### Veri Seti

[PathVQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa) (HuggingFace)
- 4.998 patoloji görüntüsü üzerinde toplam 32.799 soru–cevap çifti
- Bu projede kullanılan alt küme: yes/no soruları (~21.000 örnek)
- Eğitim / doğrulama / test bölmeleri veri setiyle birlikte gelir

### Katkı ve Özgünlük

- PathVQA veri seti üzerine Türkçe dil desteği ekleyen literatürdeki ilk çalışma (mevcut çok dilli çalışmalar yalnızca EN/DE/FR desteklemektedir).
- Medikal VQA sınıflandırıcısını LLM tabanlı bir açıklama katmanıyla (chatbot benzeri çıktı) birleştirir.
- Yalnızca not defteri (notebook) prototipi değil, uçtan uca çalışan tam bir web uygulaması olarak sunulur.

### Sorumluluk Reddi

Bu proje yalnızca **eğitim ve araştırma amaçlıdır**; profesyonel tıbbi teşhisin yerine geçmez.
