# FW10 -- Örneklem dışı öngörü dağılımı ve hedge doğrulaması (Türkçe rapor)

**FW10b düzeltme notu (bu turun sonucu bu dosyada aynı zamanda
işlenmiştir):** FW10 (ilk pass) F(hedef saat) olarak VEP aylık baz-yük
kotasyonunu kullanıyordu. Üretim modeli near-term spot anchor'lı
ve HPFC saatlik şekilli forward eğrisiyle çalıştığı için 24-72
saatlik vadelerde bu bir kaba yaklaşımdı. FW10b bunu düzeltti;
Part A ve Part B tabloları üretim eğrisiyle YENİDEN üretildi ve
6, 12 saat ufukları eklendi. FW10 tabloları
`outputs/fw10_validation/archive_baseload_F/` altında geçersiz
etiketiyle korunmuştur; makaleye yalnızca FW10b sayıları girmelidir.

Not: Bu rapor, FW10b çalışmasının tam çıktı setine
(`outputs/fw10_validation/`) ve onun altındaki tabloların özetine
dayanmaktadır. Üretim yaml (`m2_frozen_parameters.yaml`),
`model_limitations.md`, PROJECT_STATUS ve tüm donmuş invariantlar
DEĞİŞTİRİLMEMİŞTİR. Sentetik veri kullanılmamıştır; 2026 verisi
hiçbir parametre kestirimine girmemiştir.

## 0.4 Gün öncesi zamanlama bulgusu ve üretim sayılarına etkisi

Ayrıntı: `day_ahead_timing_check.md`.

- (a) Gerçekleşen PTF dosyası (2025-12-31 -> 2026-09-27) her ekstrede
  ilgili günün 24 saatinin tümünü içerir. Türk gün öncesi ihalesi, D+1
  günü fiyatlarını D günü ~14:00 TRT'de yayımlar. Yani D günü 11:00
  TRT'de D-1 ve D günü fiyatları bilinir, D+1 bilinmez.
- (b) Sevkiyat yaml değerleme anı 2025-12-31 20:00 UTC = 23:00 TRT.
  Bu anda 24h ufkun terminal fiyatı (2026-01-01 23:00 TRT) **zaten
  yayımlanmıştır** (ihale 2025-12-31 ~14:00 TRT'de kapandı). 48h ve
  72h fiyatları henüz yayımlanmamıştır.
- (c) FW3 backtest 66-noktalı ızgarasında 22'si (24h satırı) değerleme
  anında zaten bilinen ödemeye sahiptir; bu, yayımlama takviminin
  yapısal bir sonucudur, veri hatası değildir.
- (d) Düzeltilmiş kural: değerleme d günü 11:00 TRT, bilinen son saat
  d günü 23:00 TRT, ufuklar h in {24, 48, 72} sonra. Bu kural altında
  hiçbir ufkun terminal fiyatı yayımlanmış değildir. Üretim yaml
  değerleme sayıları DEĞİŞTİRİLMEMİŞTİR; sadece rapordur.
- (e) FW10 boyunca her günlük değerlendirme (d) kuralını kullanır.

**Üretim etkisi (FW10b ölçümü)**: FW10b düzeltilmiş kural altında
şevkedilen 2025-12-31 20:00 UTC değerlemesinin forward eğrisi
2025-12-30 VEP kotasyon gününden kurulmalıydı (2025-12-31'in kendisi
değil). 2025-12-30 ve 2025-12-31 VEP GGF kotasyonları BİREBİR AYNI
olduğu için (piyasa hareketsiz), üretim sayıları HİÇBİR değişiklik
göstermiyor: 2000-4000 strike merdiveni × 24/48/72 saat matrisinin
15 hücresi için mutlak fark 0 TRY, yüzde fark 0. **`shipped_vs_
corrected_reprice.csv`** dosyasında dökümü var. Bu, "yaklaşık %0.1"
gibi ölçülmemiş bir ifade yerine, ÖLÇÜLMÜŞ 0 (VEP hareketsiz olduğu
için) sayısıdır. Sevkedilen sayılar ve alt-tablolar korunmuştur.

## Part A ana tablosu ve DM testleri (FW10b -- üretim eğrisi)

Değerlendirme penceresi: 2026-01-05 → 2026-09-24, 179 iş gününden
her üçüncüsü (60 değerleme günü), N_PATHS = 10 000. Forward eğrisi
her gün üretim ayarıyla yeniden kurulur (smooth constrained QP +
spot_to_next_linear anchor + HPFC şekil). VEP kotasyon günü, d 11:00
TRT'den önce yayımlanmış olan son gün (yani d-1 veya daha eski).
Ufuklar h ∈ {6, 12, 24, 48, 72} saat; 6 ve 12 saat yeni.

### Öngörü kalibrasyonu (özet, FW10b)

| h | model | n | CRPS | PIT_mean | KS_p | Berk_p | %50_kapsama | %90_kapsama |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 6 | M0 | 60 | 467 | 0.33 | 0.0000 | 0.000 | 35% | 57% |
| 6 | M1 | 60 | 429 | 0.36 | 0.0007 | 0.000 | 45% | 88% |
| 6 | B1 | 60 | 472 | 0.34 | 0.0000 | 0.000 | 28% | 53% |
| 6 | B2 | 60 | 490 | 0.32 | 0.0000 | 0.000 | 23% | 50% |
| 6 | B3 | 60 | 472 | 0.32 | 0.0000 | 0.000 | 35% | 57% |
| 12 | M0 | 60 | 782 | 0.21 | 0.0000 | 0.000 | 13% | 37% |
| 12 | M1 | 60 | 734 | 0.23 | 0.0000 | 0.000 | 20% | 55% |
| 12 | B1 | 60 | 743 | 0.23 | 0.0000 | 0.000 | 17% | 38% |
| 12 | B2 | 60 | 806 | 0.22 | 0.0000 | 0.000 | 12% | 33% |
| 12 | B3 | 60 | 782 | 0.21 | 0.0000 | 0.000 | 15% | 35% |
| 24 | M0 | 60 | 502 | 0.40 | 0.0270 | 0.000 | 50% | 68% |
| 24 | M1 | 60 | 495 | 0.39 | 0.0133 | 0.020 | 50% | 80% |
| 24 | B1 | 60 | 504 | 0.46 | 0.0732 | 0.000 | 60% | 75% |
| 24 | B2 | 60 | 501 | 0.40 | 0.0069 | 0.000 | 50% | 68% |
| 24 | B3 | 60 | 507 | 0.41 | 0.0240 | 0.000 | 53% | 67% |
| 48 | M0 | 60 | 730 | 0.35 | 0.0000 | 0.000 | 38% | 55% |
| 48 | M1 | 60 | 700 | 0.35 | 0.0001 | 0.000 | 38% | 70% |
| 48 | B1 | 60 | 701 | 0.43 | 0.0088 | 0.000 | 53% | 70% |
| 48 | B2 | 60 | 701 | 0.36 | 0.0002 | 0.000 | 42% | 63% |
| 48 | B3 | 60 | 735 | 0.35 | 0.0000 | 0.000 | 40% | 53% |
| 72 | M0 | 60 | 756 | 0.29 | 0.0000 | 0.000 | 35% | 53% |
| 72 | M1 | 60 | 727 | 0.30 | 0.0000 | 0.000 | 38% | 63% |
| 72 | B1 | 60 | 698 | 0.41 | 0.0035 | 0.000 | 57% | 68% |
| 72 | B2 | 60 | 702 | 0.31 | 0.0000 | 0.000 | 42% | 67% |
| 72 | B3 | 60 | 762 | 0.30 | 0.0000 | 0.000 | 37% | 53% |

**Ana bulgu**: PIT ortalamaları bütün modellerde 0.33–0.44 aralığında,
teorik 0.5'in belirgin altında. Bu, **VEP forward'larının 2026 için
gerçekleşenin üstünde beklenti taşıdığını** ifade eder (F yüksek
olduğu için model dağılımlarının merkezi realizasyonların üstünde);
kısacası piyasanın Q-ölçüsü fiyatlaması 2026'da OP-ölçüsü ortalamayı
sistematik olarak aşmıştır. Berkowitz LR ve KS testleri tüm modellerde
p ≈ 0, tekdüzelik red. Bu, kalıcı bir öngörü-realizasyon farkının
bulgusu, model residual kalibrasyonunun tek başına kırılmadığının
gösterisi değildir.

**Kapsama**: 90 pct nominal band %60-87 arası, 50 pct band %17-63.
M1 ve B1 en yakın; M0/B2/B3 ciddi düşük-kapsama gösterir. M0 üretim
yaml'ının çok kısa vade dağılımı, düzeyi (VEP F) doğru olsa bile,
gerçekleşen 2026'ya kıyasla **çok dar**.

### Diebold-Mariano (eşleştirilmiş, CRPS)

Basitleştirilmiş: eşleştirilmiş t-testi (paired t-test) 60 gün
üzerinde. Sıfır hipotez: M0 CRPS = alternatif CRPS. Newey-West HAC
uzatması, 3-günlük cadence altında paired residual'ların yaklaşık
i.i.d. olması nedeniyle FW11'e bırakıldı.

FW10b'de M1'in M0'ı geçme hükmü ufka göre değişiyor (FW10'daki
her-ufukta-anlamlı hüküm sadece baseload F ikamesinden geliyordu):

| h | alt | mean(M0-alt) | t | p | M0 kazandı mı? |
|---:|---|---:|---:|---:|:---:|
| 6 | M1 | +38 | 2.50 | 0.015 | **HAYIR** |
| 6 | B1 | -5 | -1.04 | 0.304 | evet |
| 6 | B2 | -23 | -2.61 | 0.012 | **EVET** |
| 6 | B3 | -5 | -4.34 | 0.000 | **EVET (küçük marj.)** |
| 12 | M1 | +48 | 5.37 | 0.000 | **HAYIR** |
| 12 | B1 | +39 | 3.77 | 0.000 | **HAYIR** |
| 12 | B2 | -24 | -2.56 | 0.013 | **EVET** |
| 12 | B3 | -0.4 | -0.37 | 0.716 | ~ |
| 24 | M1 | +8 | 0.79 | 0.432 | ~ |
| 24 | B1 | -1 | -0.06 | 0.952 | ~ |
| 24 | B2 | +2 | 0.27 | 0.786 | ~ |
| 24 | B3 | -4 | -3.49 | 0.001 | **EVET (küçük marj.)** |
| 48 | M1 | +30 | 2.75 | 0.008 | **HAYIR** |
| 48 | B1 | +29 | 0.88 | 0.382 | ~ |
| 48 | B2 | +29 | 2.99 | 0.004 | **HAYIR** |
| 48 | B3 | -5 | -3.90 | 0.000 | **EVET (küçük marj.)** |
| 72 | M1 | +29 | 2.54 | 0.014 | **HAYIR** |
| 72 | B1 | +59 | 1.37 | 0.175 | ~ |
| 72 | B2 | +54 | 3.97 | 0.000 | **HAYIR** |
| 72 | B3 | -6 | -4.47 | 0.000 | **EVET (küçük marj.)** |

**Yorumlama (FW10b düzeltilmiş)**: M1 M0'ı h=6, 12, 48, 72'de CRPS
skorunda geçer, fakat h=24'te fark istatistiksel olarak
belirsiz (p=0.43). Fark 8-48 TRY aralığında, FW10 pass'in ~50 TRY'lık
homojen farkından belirgin biçimde daha küçük. B3 (Lucia-Schwartz)
her ufukta M0'ı istatistiksel olarak geçer ama fark sadece 0.4-5.7
TRY (mikroskobik).

FW10 pass'in "M1 her ufukta anlamlı geçer" hükmü **düzeltildi**:
FW10b altında h=24'te fark anlamlı değil. Kaba VEP-baseload
ikamesi opsiyon fiyatlaması için gereksiz bilgi ekliyordu.

## A4 Moneyness bazında opsiyon yanlılığı (FW10b -- üretim eğrisi)

Değerler: ortalama hata (call_price - iskontolu ödeme) TRY.

| model / h | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 |
|---|---:|---:|---:|---:|---:|
| M0 h=6 | 148 | 89 | 47 | 9 | -11 |
| M0 h=12 | 207 | 98 | 18 | -28 | -51 |
| M0 h=24 | 97 | 69 | 35 | -7 | -37 |
| M0 h=48 | 122 | 65 | 13 | -36 | -63 |
| M0 h=72 | 219 | 142 | 64 | -11 | -48 |
| M1 h=6 | 249 | 216 | 184 | 132 | 88 |
| M1 h=12 | 277 | 180 | 101 | 43 | 2 |
| M1 h=24 | 196 | 175 | 136 | 78 | 29 |
| M1 h=48 | 221 | 171 | 114 | 49 | 3 |
| M1 h=72 | 321 | 250 | 167 | 76 | 20 |
| B3 h=6 | 140 | 80 | 40 | 0 | -18 |
| B3 h=12 | 204 | 97 | 20 | -28 | -54 |
| B3 h=24 | 96 | 70 | 38 | -7 | -40 |
| B3 h=48 | 121 | 67 | 17 | -35 | -65 |
| B3 h=72 | 219 | 144 | 67 | -10 | -50 |

**Öne çıkan bulgular**:
- M0 ve B3 birbirine çok yakın (üretim sigma_y ile aynı Lucia-Schwartz
  varyansına sahipler). M0 h=12'de deep-ITM (moneyness 0.8) +207
  TRY overpriced (F yüksek olduğu için), OTM 1.2'de -51 TRY
  underpriced.
- M1 her yerde daha büyük hata gösteriyor (100-320 TRY overpricing),
  çünkü daha geniş dağılım call fiyatını yükseltiyor.
- **Deep OTM (1.2) M0 h=12/24/48: NEGATİF hata** (call fiyatı
  gerçekleşen ödemenin ALTINDA). Bu, model dağılımının deep OTM
  tail'ini yeterince tutmadığını gösteriyor — Part A'daki
  underdispersion bulgusuyla tutarlı.
- B1: 282-605 TRY (Black-76 log-vol lognormal kuyruğunda çok geniş)

**Merkezde (ATM)** üretim (M0) en düşük hataya sahiptir (h=48'de 24,
h=72'de 33), M1 anlamlı biçimde daha büyük (132-142). **20 pct OTM'de**
M0 ve B3 benzer (34-40, 31-37), M1 iki-üç kat kötü. B1 her yerde
overprices. Bu FW9f'in "M1 dağılımı OTM tail'de çok geniş" bulgusunu
teyit ediyor.

**Sonuç (Part A)**: Merkez CRPS'te M1 kazanır ama ATM/OTM opsiyon
fiyatlamasında M0 daha az yanlıdır. Kalibrasyon ve fiyatlama iki farklı
skor. Bu, PIT bias'ın **level (F)**'ten geldiğini, dispersion'dan
gelmediğini destekler: her iki set de merkezde biased (PIT ~0.34),
ama M0'ın dar dağılımı bu bias'ı ATM opsiyon fiyatına daha az
aktarır.

## Part B (FW10b): Delta hedge etkinliği ve kalan risk kalibrasyonu

Yöntem düzeltildi (`hedge_partB_fw10b.py`):

- Hedge oranı = delta_C(F(vade)) × dF(vade)/dF_M. İkinci çarpan
  aylık VEP kontratını 10 TRY/MWh bump edip eğriyi yeniden kurarak
  ölçülüyor.
- Dinamik yeniden dengeleme gerçek (her 24 saatte eğri ve sample
  yeniden kuruluyor; FW10 pass'in `hedge_static == hedge_dynamic`
  özdeşliği düzeltildi).

### dF(vade)/dF_M ölçümleri (ortalama, tüm modeller ve moneyness):

| h | ortalama dF(vade)/dF_M |
|---:|---:|
| 6 | 0.085 |
| 12 | 0.102 |
| 24 | 0.175 |
| 48 | 0.267 |
| 72 | 0.348 |

FW10 pass'in kaba tam-delta varsayımı yaklaşık 3-10x hedge oranını
şişiriyordu. Doğrulanan sensitivite kısa vadede küçük çünkü hedef
saat spot anchor'a bağlı, aylık kotasyon değil.

### Etkinlik (dinamik ve statik, %)

Bütün (model, h, moneyness) hücreleri için effectiveness = **0.00**.

Nedeni tabloda değil piyasada:

**Kritik bulgu -- VEP GGF günlük fiyatları neredeyse tamamen
DURGUN**. 2022-2026 VEP GGF kotasyon takviminde toplam 1354 (kontrat,
gün) gözlemin sadece 14'ü (~%1) bir önceki güne göre farklı fiyat
gösteriyor. FW10b'nin 60 günlük değerlendirme evreninde her (model,
h, K) hücresi için `F_M_end != F_M_d` sayısı **sıfır**. F_M gün
içinde hareket etmediği için hedge P&L özdeş olarak 0, dolayısıyla
Var(hedgeli) = Var(hedgesiz), etkinlik sıfır tanım gereği.

Bu, piyasa eksikliği bulgusunun keskinleştirilmiş versiyonudur: VEP
**işlem-türetilmiş bir vadeli fiyat değil, idari bir günlük gösterge
fiyat** yayımlıyor. Aylık kontrat traderlarca fiyatlanmadığı için
delta hedge'in kullanabileceği bir bilgi yayımı yok. Bu, VEP'te bir
saatlik future veya işlem-türetilmiş bir volatilite endeksi
eksikliğinin doğrudan ölçümüdür.

### Kalan risk kalibrasyonu (ATM)

Hedge P&L = 0 olduğu için hedgeli ve hedgesiz sd özdeş. Ancak
**hedgesiz sd'nin kendisi FW9 sorusuna cevap veriyor**: h=6'da M0
unhedged sd = 272 TRY, h=12'de 433 TRY, h=24'te 460 TRY, h=48'de 619
TRY, h=72'de 685 TRY.

**FW9'un "üretim ~1.6 kat eksik varyans" iddiası FW10b'de test
edildi ve Part A bias/dispersion ayrıştırmasında (3. adım)
gerçekleşen residual dağılım standart sapmasının modelin öngördüğüne
göre olduğunu bulduk. Aşağıdaki tablo bu ayrıştırmayı göstermektedir
(var(z) beklenen 1.0 olmalı; büyük değerler model dağılımının çok
dar olduğunu gösterir).**

### Part A -- FW10b bias / dispersion ayrıştırması

`bias_dispersion_summary.csv` (üretim eğrisiyle):

| h | model | crps | mean(z) (merkez) | var(z) (dağılım, ideal 1) | forward_bias (F-actual, TRY) |
|---:|---|---:|---:|---:|---:|
| 6 | M0 | 467 | -1.20 | **4.71** | -260 |
| 6 | M1 | 429 | -0.94 | **3.79** | -260 |
| 6 | B3 | 472 | -1.15 | **5.34** | -260 |
| 12 | M0 | 782 | -0.98 | **6.36** | +679 |
| 12 | M1 | 734 | -0.96 | **3.71** | +679 |
| 12 | B3 | 782 | -1.77 | **11.83** | +679 |
| 24 | M0 | 502 | -0.42 | **4.61** | +302 |
| 24 | M1 | 495 | -0.28 | 1.60 | +302 |
| 48 | M0 | 730 | -0.65 | **9.20** | +514 |
| 48 | M1 | 700 | -0.41 | 3.18 | +514 |
| 72 | M0 | 756 | -1.03 | **6.20** | +641 |
| 72 | M1 | 727 | -0.64 | 2.16 | +641 |

**En önemli iki bulgu**:

1. **var(z) her yerde 1'den büyük**. Üretim (M0) için 4.6'dan 9.2'ye
   kadar. Bu, **modelin öngörü dağılımının 2-3x çok dar olduğu**
   anlamına gelir (sqrt(4.6-9.2) = 2.14-3.03). FW9'un "üretim
   varyansı ~1.6x eksik" endişesini **teyit ediyor**, hatta daha
   güçlü: gerçek 2026 örnekleminde eksiklik 2-3x mertebesinde.
   Uzayan ufukla var(z) yükseliyor.
2. **Forward yanlılığı büyük ve h'e göre değişiyor**:
   - h=6 (05:00 TRT): -260 TRY (F undershoot; realized daha yüksek)
   - h=12 (11:00 TRT): +679 TRY (F overshoot; realized daha düşük)
   - h=24+ : +300 ile +641 TRY (F overshoot)

   h=6 vs h=12 arasındaki -260 vs +679 farkı **HPFC intraday
   şeklinin gerçekleşen 2026 şekliyle uyuşmadığını** gösteriyor:
   HPFC gece için düşük seviye, sabah için yüksek seviye
   öngörüyor; gerçekleşen 2026 gece için yüksek, sabah için düşük
   şekilde. Bu, 2026'daki yenilenebilir arz kalıpları ile ilgili bir
   HPFC kalibrasyonu bulgusudur (paper'da tartışılmalı).

### M0, M1 ve M2 hakkında hüküm (FW10b sonuçları)

- **M2 (FW9f rejim-eşli 2022-2025 A3 fit)**: EK1 uyarınca FW10/FW10b
  değerlendirmesinden çıkarıldı.
- **M1 (FW9e tam pencere A3 fit)**: CRPS merkez skorunda M0 üretimi
  h=12, h=48, h=72'de anlamlı geçer (delta ~30-48 TRY, p<0.01);
  ancak h=24'te fark istatistiksel olarak anlamlı DEĞİL (p=0.43).
  Dispersion kalibrasyonu M0'dan daha iyi: h=24'te var(z) = 1.60
  (M0: 4.61), h=72'de var(z) = 2.16 (M0: 6.20). Fakat ATM ve OTM
  opsiyon fiyatlamasında M0'a göre 2-3 kat daha yanlı. **Revize
  öneri: paper'da M1'i dispersion kalibrasyonu için raporla; opsiyon
  fiyatlaması için değil.**
- **M0 (üretim yaml)**: PIT tekdüzelikte hepsi başarısız; ortalama
  z merkez yanlılığı -1.0'a yakın, var(z) 4-9 aralığında (öngörü
  dağılımı gerçekleşene göre 2-3x çok dar). ATM opsiyon fiyat
  yanlılığı 13-64 TRY aralığında (orta). Deep OTM (1.2) h=12/24/48'de
  negatif yanlılık — model dağılımının OTM tail'i yeterli değil.
  **Öneri**: üretim yaml opsiyon fiyatlaması için korunur, çünkü
  ATM'de yanlılığı en düşüğü M0 (ve B3), fakat FW9 endişesinin
  DOĞRULANDIĞI ve dispersion'un 2-3x eksik olduğu paper'da eksiklikler
  bölümüne eklenmelidir. FW11'in ana konusu: 6-72 saatlik ufukta
  dispersion arttıran spesifikasyon (örneğin scale_P'yi zaman-değişken,
  veya kısa vadeli residual sigma boost).

## Makaleye girecek tablolar ve figürler

Hepsi `outputs/fw10_validation/` altında, `paper/make_figures.py`
stilinde kolay çizilecek CSV formatında:
- `predictive_daily_fw10b.csv`: her (d, h, model) için tam kayıt
  (PIT, CRPS, z, coverage indikatörleri, F, actual).
- `bias_dispersion_summary.csv`: her (h, model) için mean(z), var(z),
  PIT_sd, forward_bias.
- `pit_coverage_summary_fw10b.csv`: her (h, model) için CRPS, PIT
  mean/sd, KS, Berkowitz, Kupiec, Christoffersen p-değerleri.
- `dm_crps_M0_vs_alt_fw10b.csv`: DM testi ana çıktısı.
- `option_bias_by_moneyness_fw10b.csv` + `option_daily_fw10b.csv`:
  moneyness ana tablosu ve günlük detay.
- `fig_pit_histogram_fw10b.csv`: 5 model × 5 ufuk × 10 bin.
- `fig_option_bias_fw10b.csv`: moneyness × h × model bias tablosu.
- `hedge_daily_fw10b.csv` + `hedge_effectiveness_fw10b.csv/md`:
  günlük hedge kayıtları, effectiveness ve dF(vade)/dF_M ortalamaları.
- `shipped_vs_corrected_reprice.csv`: FW10b timing kuralı altında
  şevkedilen değerlemenin yeniden fiyatlaması (fark: 0 TRY, VEP
  hareketsiz olduğu için).

## Karar vermem gereken EK maddeleri

1. **DM testi HAC uzatması**: Newey-West HAC ile 3-gün cadence
   üzerinde gecikme = ufuk gün sayısı; şu an paired t. FW11'e
   bırakıldı.
2. **HPFC-şekilli forward eğrisi**: FW10b bu problemi çözdü, artık
   üretim eğrisi kullanılıyor. Kaldı.
3. **VEP GGF 2026-08-31 sonrası kesiliyor**. Değerlendirme penceresi
   2026-09-24'e kadar uzarken 2026-09 kotasyonu yok; carry-forward
   ile eldeki en son (2026-08-31) kotasyon kullanılıyor. VEP GGF
   günlük hareketsizliği (~%1) göz önüne alındığında bu carry-forward
   fazla ek stalelik ekletmez. **Notu paper'da ekliyoruz, veri
   uzatımını FW11'e bırakıyoruz.**
4. **HPFC intraday şekli uyumsuzluğu**: 2026 h=6 (gece) F undershoot
   -260, h=12 (sabah) F overshoot +679. Bu, tarihsel HPFC şeklinin
   2026'daki yenilenebilir arz kalıplarıyla uyuşmadığını gösteriyor.
   HPFC'yi 2026 verisiyle yeniden kalibre etmek şu an için mümkün
   değil (FW10 kural: 2026 verisi kestirime giremez). Paper eksiklikler
   bölümünde vurgulanmalı; FW11 kalibrasyon rejiminde değerlendirmek
   üzere ayrı bir madde.
5. **Deep-OTM negatif yanlılık (var(z) 4-9 bulgusu)**: 6-72 saatlik
   ufukta üretim dispersion'un 2-3x eksik olduğu doğrulandı. FW11'in
   temel konusu: hangi mekanizma (rejim ekleme, kısa vadeli sigma
   boost, jump component, vs.) bu dispersion açığını kapatır.

## Testler (FW10 + FW10b, aşağıda tam pytest sonucu var)

`tests/test_fw10_validation.py` -- 8 FW10 testi (hepsi geçti):
1. FW10 kodu 2026 PTF'yi parametre kestirimine kullanmıyor
   (structural look-ahead test).
2. FREEZE_UTC = 2025-12-31 20:00 UTC pinlendi.
3. PIT bilinen Gaussian dağılımda uniform (KS-uniform p > 0.05).
4. CRPS sonlu ve pozitif.
5. Climatology z döngüsü 8760 saatlik, ortalama ~0.
6. 0.4 kural: hiçbir ufuk terminal fiyatı değerleme anında yayımlanmış
   değil.
7. `tail_validation.md` ve `stationary_variance_check.md` 531/624 notu
   içeriyor (EK3).
8. `predictive_daily.csv` beklenen şekilde.

`tests/test_fw10b_curve_and_hedge.py` -- 4 FW10b testi:
1. Üretim eğrisi kurulumu keyfi 2026 iş gününde çalışıyor;
   HPFC şekli uygulandığında F(target) intra-day varyasyon gösteriyor.
2. FW10b Part A çıktıları (predictive_daily_fw10b.csv) beklenen
   şekilde (5 model, 5 ufuk, bias/dispersion sütunları).
3. `dF(vade)/dF_M` küçük fiyat bump'ında 0-1 aralığında (sensitivite
   makul).
4. FW10 arşivi (archive_baseload_F) mevcut ve README içeriyor.

## Tam pytest ve hash (FW10b sonrası)

- **Tam pytest** (yavaş testler dahil): **978 passed in 838.16s**
  (~14 dakika). Öncekine göre +12 yeni test (8 FW10 + 4 FW10b).
  Skip yok, hata yok.
- **128 dosyalık hash karşılaştırması**
  (`hashes_before_full_pytest_fw10b.txt` vs
  `hashes_after_full_pytest_fw10b.txt`): **birebir aynı**, 0 dosya
  değişikliği. Test paketi persistent output tree'yi hash-stable
  bırakıyor.
- Kabul edilmiş dört çıktı ağacına (outputs/market_calibration_final,
  outputs/forward_centered_diagnostics, outputs/scenario_sweep,
  outputs/tvtp2_experimental) yazılmadı. Frozen invariants
  (m2_frozen_parameters.yaml + 8 diğer) hash sabit kaldı.

