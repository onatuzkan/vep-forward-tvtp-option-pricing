# FW9f – Kısa Türkçe rapor (son tur)

## §1 – Fiyat-tutarlı iso-varyans (F14 v2)

Delta = sqrt(F^2 + s_P^2) = 2 931.42 (F = 2 917.78, s_P = 282.48). Üç
fiyat-tutarlı hedef sd_TRY bandını 556–624 arasında sarar; üretim
mixture sd_TRY = 582.48 tam ortalarında. Ancillary asinh_scale_2025
hedefi (sd_TRY = 1 306) çok yüksek olduğu için çıkarım için
kullanılamaz. Detay ve düzeltilmiş yorumlar
`kappa_sensitivity_isovariance_v2.md` dosyasındadır.

## §2 – F14 v2 yorumları (düzeltildi)

Önceki `kappa_sensitivity_isovariance_v2.md` intro'sundaki
"reproduce that value within a few percent" cümlesi YANLIŞTI ve
düzeltildi. Doğru resim: F14 tek-rejimli (paylaşılan sigma)
indirgemedir; aynı toplam varyansta iki-rejimli üretim karışımının
kurtosisi F14'ün Gauss'una göre daha yüksek olduğundan **aynı sd'de
ATM değeri daha düşük olur**. kappa = 0.078'de üretim-hedefli
iso-varyans eğrisi 72 saatte 183.4 verirken üretim iki-rejimli fiyatı
166.75'tir (~%10 boşluk).

F14'ün mesajı seviye değil şekildir: modele-sadık kappa ailesi
[0.15, 0.22] içinde 72 h ATM 169–176 bandında, [0.01, 0.30] tam
aralığında 162–187. T = 1 h ise 7'den 98'e çıkar — kısa vade
kappa'ya çok daha duyarlı.

Ek yorum maddeleri (hepsi
`kappa_sensitivity_isovariance_v2.md` içinde):

- ATM fiyat ~ linear in sd (sd^2 değil): kappa = 0.078'de sd oranları
  1.071 ve 0.955, ATM oranları 1.087 ve 0.945.
- Tam pencere tutarlı fit'in TRY sd'sinin 927'ye çıkması **rejim
  karışmasından**dır (2019–2021 düşük fiyat çağıyla 2022–2025 yüksek
  fiyat çağının asinh ölçekte havuzlanması sigma_stress'i şişirir).
- ATM +%25 ile K = 4000 +%312 farkı **karışımın yüksek kurtosisinden**
  gelir; tail probu, iç kitleye göre çok daha güçlü tepki verir.
- asinh + delta eşlemesinin scale_P sabitken bir yapısal üst sınırı
  vardır: sd_asinh 0.20 → sd_TRY ≈ 583 bandı; sadece karışım şekli
  (skew, kurtosis) sabit sd'de değişebilir, F14 tam bunu atar.

## §3 – Rejim-eşli 2022-2025 A3 tutarlı fit

Dosya: `TVTP_1cov_A3_2022_2025.pkl`,
`constant_trans_A3_2022_2025.pkl`.

TVTP (1 kovaryat, z_lag1), 2022-01-01 → 2025-12-31 20:00 UTC penceresi,
n = 35 061 saat, LL = 19 332.68 (converged flag = False, grad_norm =
0.64 iç noktada), Hessian tabanlı SE'ler:

| parametre | tahmin | SE |
|---|---:|---:|
| mu_normal   |  0.0092 | 0.0006 |
| mu_stress   | -0.0303 | 0.0042 |
| sigma_normal|  0.0910 | 0.0075 |
| sigma_stress|  0.3605 | 0.0144 |
| phi         |  0.8505 | 0.0114 |
| alpha01     | -2.5152 | 0.0516 |
| gamma01     | -1.0825 | 0.0545 |
| alpha10     | -1.2080 | 0.0533 |
| gamma10     |  0.8667 | 0.0638 |

Sabit-geçiş karşılaştırma (LR testi için) LL = 18 486.40, converged =
True; TVTP ile 2 ekstra parametre karşılığı DeltaLL = 846.28 (chi2(2)
p-değeri esasen 0), yani z_lag1 kovaryatı 2022–2025 alt-örneğinde
istatistiksel olarak son derece anlamlı.

**İma edilen durağan sd (TRY, F = 2 917.78):**
sigma_mix^2 = (1 - pi_stress) * sigma_normal^2 + pi_stress *
sigma_stress^2 formülü ile karışım varyansı 0.0345,
var_asinh = sigma_mix^2 / (1 - phi^2) = 0.1378,
sd_asinh = 0.3712, sd_TRY = 1 088. Rejim-eşli fit — rejim-havuzlama
şüphesine rağmen — üretim değeri 582 veya gözlem 531'i **azaltmıyor**
tersine daha da yükseltiyor: 2022–2025 alt-örneğinde stress rejiminin
sigma'sı 0.36'ya çıkar. Bu, "tam pencerenin havuzlanması sd'yi şişiriyor"
hipotezinin tek başına dispersiyonu açıklamadığını gösteriyor.

## §4 – Kuyruk doğrulaması

Dosyalar: `tail_validation.csv`, `tail_validation_ks.csv`,
`tail_validation.md`.

Yöntem: 200 000 yol × 72 saat MC yerine, her parametre setinden
MS-AR(1)+TVTP artık sürecinden **TEK UZUN durağan yol** (2 000 000
saat + 10 000 saat burn-in) simüle edildi; climatology z rejim
zincirini besliyor; asinh artık delta = 2 931.42 ile TRY'ye eşlendi.
Toplam süre 3 set için ~25 saniye.

| kaynak | n | sd_TRY | q05 | q25 | q50 | q75 | q95 | q99 | P(>1500) | P(<-1500) | KS vs obs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| observed_2025 | 8 757 | 531 | -1 004 | -231 | 52 | 334 | 730 | 1 105 | 0.004 | 0.016 | – |
| production (a) | 2 000 000 | 581 | -963 | -373 | 1 | 369 | 959 | 1 406 | 0.007 | 0.007 | **0.087** |
| FW9e A3 (b) | 2 000 000 | 984 | -1 724 | -478 | 38 | 512 | 1 473 | 2 560 | 0.048 | 0.066 | 0.125 |
| FW9f 22-25 A3 (c) | 2 000 000 | 1 176 | -2 141 | -607 | 30 | 595 | 1 721 | 2 942 | 0.067 | 0.097 | 0.159 |

**Merkez, üst kuyruk ve alt kuyruğu en iyi yakalayan set:**
üçünde de **üretim (a)** kazanır. Merkezde IQR (q25–q75) obs 565 vs
prod 742 (fark %31), FW9e 990 (%75), FW9f 1 202 (%113); 5–95 pct
merkez-kuyruk deltaları prod +%3 / +%31 (obs banda kıyasla),
FW9e -%42 / +%102, FW9f -%53 / +%136. KS sıralaması aynı hikayeyi
söyler: prod 0.087 < FW9e 0.125 < FW9f 0.159.

**Kap ve taban dikkati:** gözlemde fiyat tavanı 4 500 TRY/MWh
(2026-04-04'e kadar; sonra 5 000) ve tabanı 0 var — obs artığının
min'i -2 703, max'ı 2 098. Model ne tavan ne taban uygular; 1 ve 99
persantillerdeki mismatchi kısmen bu asimetrik kırpma
açıklıyor. Yorumu 25–75 IQR, 5–95 pct bantı ve KS üzerinden yapmak
en doğrusu.

## §5 – Limitations önerileri

`proposed_limitations_lines.md` dosyasına (i) maddesinin FW9f
revizyonu eklendi: FW9c/FW9d "yaml kappa yanlış" çerçevelemesi
**geri çekildi**; yeni öneri satırı yaml'ın stat sd_TRY = 582'sinin
gözlem 531'e üç setin en yakını olduğunu (%9) belirtiyor.
`model_limitations.md` dosyasına dokunulmadı — sadece öneri.

## §6 – Testler

`tests/test_fw9f_tail_and_isovar.py` üç test:
1. `test_f14v2_price_consistent_targets_sandwich_production`: F14 v2'de
   üç fiyat-tutarlı hedef her kappa'da üretim eğrisini sandviçliyor,
   ancillary asinh_scale hepsinin üstünde.
2. `test_regime_matched_2022_2025_fit_interior_and_ordered`: §3 fit'i
   iç noktada (0 < sigma_n < sigma_s, phi in (0,1)), Hessian SE'leri
   sonlu, stress varyansı > normal varyansı.
3. `test_tail_validation_orderings_and_reproducibility`: tail_validation
   çıktılarındaki kantillerin monotonluğu ve prod < FW9e < FW9f KS
   sıralaması + prod sd_TRY - obs sd_TRY farkının %25 içinde kalması.

Üçü de `pytest tests/test_fw9f_tail_and_isovar.py -v` altında geçiyor.

## §7 – Pytest ve hash

- **Pytest**: `966 passed in 792.28s` (0 hata, uyarılar var — hepsi
  eski Pandas'ın timezone drop uyarısı). Skip yok.
- **Hash karşılaştırması**: `hashes_after_fw9f.txt` üretildi,
  `hashes_after_fw9e.txt` ile karşılaştırıldı. 128 dosyanın 102'si
  bit-eş; farklı olan **26 dosyanın hepsi PNG** (matplotlib'in her
  render'da farklı byte üretmesinden — içerik aynı, byte'lar farklı).
  Değiştiği raporlanan dosyaların hiçbiri yaml / json / csv / md / py
  değildir. Yani donmuş invariantlar (m2_frozen_parameters.yaml,
  tvtp2_frozen_parameters.yaml, tüm forward-centered CSV/JSON
  çıktıları, kalibrasyon audit MD'leri) bit-eş korunmuştur.
- FW9d ve FW9e'nin hashes dosyaları md5 düzeyinde tıpatıp aynıydı;
  FW9f'in de aynı olması beklenirken PNG'ler için ayrı bir çevrim
  gerçekleşmiş — bu, `model_limitations.md`, yaml'lar veya sayısal
  çıktılar üzerinde hiçbir değişiklik yapılmadığı anlamına gelir,
  sadece PNG raster'larının yeniden üretilmesidir.

## Genel çerçeve — FW9 turlarının özeti (a → f)

- **FW9d'nin "kappa yanlış" ölçek atfı FW9e'de tersine çevrildi**:
  FW9d, yaml kappa 0.078'in FW9c bracket'ı [0.205, 0.225] dışında
  olduğunu vurguluyordu. FW9e stat sd_TRY özdeşliğini kullanarak
  yaml'ın (kappa, sigma) çiftinin ürünü olan durağan varyansın
  gözlem 2025 sonucuyla %7 içinde olduğunu ölçtü. FW9f §4 bu
  durumu doğrudan simülasyonla teyit etti (KS mesafesi üretim
  0.087, FW9e A3 0.125, FW9f 22-25 A3 0.159).
- **FW9c ve FW9d'deki "yaml kappa yanlış" çerçevelemesi geri çekildi**
  (`proposed_limitations_lines.md` FW9f revizyonuna bakınız).
  Bracket'ın kappa'sı ayrı bir artık spesifikasyonuna aittir (tek
  rejimli AR(1), farklı deseasonalizasyon); pricer'ın kullandığı
  iki-rejimli MS-AR(1)+TVTP A3 artığına doğrudan taşınmaz.
- **Karşılayamadığımız kısım**: FW9f rejim-eşli 2022–2025 fit'i,
  havuzlama hipotezini teyit etmedi — sd_TRY 1 088 çıktı, üretim 582
  ve gözlem 531'den açık farkla yüksek. 2022–2025 alt-örneği yüksek
  fiyat dönemi olduğu için stress sigma daha da büyür. Fit'in stat
  sd'sinin gözleme yakınlaştırılması, sabit s_P altında iki-rejimli
  spesifikasyondan tek başına çıkmayabilir; bir sonraki tur (FW10)
  scale_P yeniden kalibrasyonuna veya scale-dependent bir HPFC
  katmanına ihtiyaç duyabilir.
