# FW9 + FW9b + FW9c -- Artık sürecinin bağımsız yeniden kestirimi (rapor)

FW9: 2026-09-27.  FW9b (like-for-like düzeltmeleri) ve
FW9c (kappa aralığı + kalan kapanış işleri): 2026-09-28.
Değerleme kesme anı: 2025-12-31 20:00 UTC.
Kestirim penceresi: 2018-12-31 21:00 UTC → 2025-12-31 20:00 UTC
(n = 61 368 saatlik gözlem; look-ahead guard).

Bu bir TERFİ İŞİ DEĞİL.  Donmuş yaml
(`inputs/historical/m2_frozen_parameters.yaml`) üretim tabanı olarak
KALIYOR.  Aday yaml: `inputs/historical/fw9_reestimated_parameters.yaml`.

## §0 USD hipotezinin akıbeti

FW9'un ilk pass'inde "M9 USD fit'i" iddiası tek dayanak olarak bir
klasör adına (`markov_usd_final`) dayanıyordu.  FW9b bu iddiayı dört
testle sınadı ve **kesin biçimde çürüttü**:

* **§0d metadata taraması**: bundle metadata'sında `"target": "asinh(PTF_TRY_MWh)"` açıkça yazıyor -- HEDEF DEĞİŞKEN TRY.  Klasör adı dışında USD referansı yok; para birimi alanı, kur serisi, FX dönüşüm adımı GEÇMİYOR.
* **§0b USD medyanı**: 2019-2025 penceresinde median|USD PTF| = 63.68.
  Yaml scale_P = 282.48 hiçbir USD pencerede yakalanamıyor (46-90 arası).
  Ama TRY medyanı 2019-2020'de 302, 2019-2022'de 323 -- **282.48 TRY tabanlı**.
* **§0a decisive test**: `asinh(USD PTF/282.48)` üzerinde 20-start MLE
  → `sigma_normal = 0.00151, sigma_stress = 0.0578`.  Yaml
  (0.00353, 0.0924) değerlerine oturmadı; USD fit yaml'a yaklaşmıyor,
  ondan farklı yönde sapıyor.  Hipotez ölür.
* **§0c 2019-2021 alt penceresi**: sigmalar yaml'a hafif yaklaştı
  (`sigma_stress`: 2.6x → 1.85x fark) ama `sigma_normal` uzaklaştı
  (2x → 2.4x).  Pencere farkı KISMEN yardımcı, tam açıklamıyor.

**Sonuç**: USD hipotezi ölü.  Kalan sigma farkının kaynağı
kombinasyonu (FW9d §5 tarafından güncellendi):
(i) 2019-2021 alt penceresi sigma_stress farkının **%47'sini**
kapatıyor ama sigma_normal farkını KÖTÜLEŞTİRİYOR (%43); "2016-2018
pencere eksikliği" bir açıklama olarak veri ile destekli DEĞİL;
(ii) M9'un yaml provenance'ının belirttiği DESEASONALIZATION pipeline'ı
repoda gemiyor.  Kalan fark AÇIKLANAMIYOR.  **Bu iterasyonda hiçbir
dokümana, limitations satırına veya commit mesajına "USD fit"
iddiası yazılmadı.**

## §1 phi profili

Free-phi MLE tüm 20 başlangıçta phi=1 birim-kök sınırına yapıştı.
FW9b: phi'yi {0.99, 0.995, 0.999, 0.9999, 0.99999} ızgarasında sabitleyip
kalan 8 parametreyi profil olarak kestirdi
(`profile_phi_TVTP_1cov.csv`):

| phi | log-lik | grad_norm | iters | sigma_normal | sigma_stress | alpha01 | gamma01 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.99000 | 48 057.72 | 1.58 | 74 | 0.01556 | 0.25240 | -0.9283 | -0.4345 |
| 0.99500 | 51 734.81 | 6.23 | 92 | 0.01033 | 0.24619 | -0.8132 | -0.4405 |
| 0.99900 | 54 171.15 | 30.00 | 66 | 0.00736 | 0.24204 | -0.6962 | -0.4431 |
| 0.99990 | 54 352.35 | 47.19 | 102 | 0.00707 | 0.24153 | -0.6792 | -0.4435 |
| **0.99999** | **54 357.52** | 7.64 | 98 | **0.00705** | **0.24151** | **-0.6781** | **-0.4436** |

**Profil argmax phi = 0.99999**.  Log-lik phi'ye göre monoton artan;
maksimum sınırda; profil ~çok yatık (0.999 → 0.99999 arasında ΔLL = 186,
0.9999 → 0.99999 arasında yalnızca 5.2).

Argmax'ta bir POLISH adımı (L-BFGS-B `gtol=1e-10`, maxiter 3000,
başlangıç: profil fit'in son teta9'u) uygulandı.  Sonuç:
* grad_norm 7.64 → **2.59** (~1/3 iyileşme)
* L-BFGS-B `success=True`
* Hessian POSITIVE DEFINITE (min eigval 2.19e3, max 3.4e8, cond 1.5e5)
* Kalan gradyan finite-diff numerik precision'ının izini taşıyor
  (n = 61 368 obs, float64), gerçek eğrilik değil.

Standart hatalar (delta-method natural space):
| param | SE (Hess) |
|---|---:|
| mu_normal | 5.4e-5 |
| mu_stress | 1.2e-3 |
| sigma_normal | 6.2e-5 |
| sigma_stress | 8.8e-4 |
| alpha01 | 0.0186 |
| gamma01 | 0.0159 |
| alpha10 | 0.0170 |
| gamma10 | 0.0153 |

**phi için Hessian SE'si YOK** (profilde sabit); phi belirsizliği için
§2'nin deseasonalized fit'ini kullanıyoruz.  Diğer parametrelerden
hiçbiri sınıra yapışmıyor -- polished argmax interior optimum,
Hessian PD.

## §2 deseasonalized tek rejimli AR(1) (yaml phi/kappa nesnesi)

`scripts/fw9/deseasonalized_ar1.py`, `deseasonalized_ar1.json`.

Pipeline (**ASSUMPTION**, M9 orijinal deseasonalization kodu repoda
YOK -- bkz `preprocessing_audit.md`): `y_t = asinh(PTF/282.48)`
üzerinde hour-of-week + month-of-year additive seasonal dummies;
residual üzerinde OLS AR(1).

| istatistik | değer |
|---|---:|
| n | 61 367 |
| phi | **0.98342** ± 0.00073 |
| kappa_per_hour | 0.01671 |
| half_life_hours | 41.47 |
| yaml phi | 0.9246 |
| yaml kappa_per_hour | 0.078394 |
| yaml half_life_hours | 8.842 |

**FW9 deseasonalized phi (0.98342) yaml'ın 0.9246'sından ANLAMLI olarak
farklı** (t = (0.98342 − 0.9246) / 0.00073 ≈ 80).  Uyuşmazlığın en
olası kaynağı: benim seasonal dummy modelim M9'un pipeline'ından daha
basit (M9 muhtemelen daytype × hour × Fourier bileşenleri kullandı,
bkz `pde_option_model/hpfc.py`).  Daha aggressive deseasonalization
kalıcılığı DÜŞÜRÜR, benimki ise (daha az mevsimsellik alınıyor) daha
YÜKSEK phi verdi.  Bu, FW9'un yaml phi=0.9246'sını YENİ VERİYLE
üretmesi için M9'un pipeline'ının bilinmesi gerektiğini gösteriyor;
şu haliyle yaml kappa'sı reproducible değil.

## §3 T8 -- Object-matched parameter comparison

`parameter_comparison_v2.csv/md`.  Her satırda `inherited_from`
sütunu; boundary satırında (phi_MS_AR) SE/t/CI boş.

| par | inherited from | inherited | FW9 est | SE | t | 95% içinde? |
|---|---|---:|---:|---:|---:|:-:|
| sigma_normal | M9 CSV sigma1 (low-vol) | 0.003535 | 0.007050 | 6.2e-5 | +56.6 | **HAYIR** |
| sigma_stress | M9 CSV sigma0 (high-vol) | 0.092407 | 0.241516 | 8.8e-4 | +170 | **HAYIR** |
| phi_MS_AR | M9 CSV phi | 0.999996 | 0.99999 (boundary) | -- | -- | boundary flag |
| phi_deseasonalized | yaml (v2 kappa refit) | 0.9246 | 0.98342 | 7.3e-4 | +80 | **HAYIR** |
| mu_normal | M9 CSV mu1 | -0.000678 | -0.000589 | 5.4e-5 | +1.66 | **evet** (yaml sıfırlar; kasıtlı) |
| mu_stress | M9 CSV mu0 | -0.004266 | +0.000433 | 1.2e-3 | +3.90 | HAYIR |
| alpha01 | yaml DERIVED | -1.0157 | -0.6781 | 0.019 | +18.1 | **HAYIR** |
| alpha10 | yaml DERIVED | -1.8952 | -1.6511 | 0.017 | +14.3 | **HAYIR** |
| gamma01 | M9 trans (raw gamma10) | -0.5838 | -0.4436 | 0.016 | +8.8 | **HAYIR** |
| gamma10 | M9 trans (raw gamma01) | +0.0777 | +0.1759 | 0.015 | +6.4 | **HAYIR** |

**phi_MS_AR** boundary'da olduğu için CI'sı yok, ancak FW9 (0.99999)
ve M9 CSV (0.999996) unit-root civarında **aynı bölgede** -- karşılaştırma
GEÇİYOR, boundary üzerinde M9 CSV'nin bağımsız doğrulaması.

Diğer parametrelerin CI dışında olmasının nesne-eşleşmiş yorumları:
* **sigmalar**: FW9 pencere M9 penceresinden farklı (2016-2018 yok, 2023-2025 var).
  Sigma farkı öncelikle sample farkı; §0a USD hipotezi ölü.
* **phi_deseasonalized**: FW9 seasonal model M9'unkinden BASİT; farklı
  deseasonalized artıklar farklı phi.
* **mu_stress**: M9 mu0 = -0.0043 (high-vol regime intercept); FW9 mu_stress = +0.0004.
  İşaret farklı, magnitüd farklı.  Yaml `regime_means=[0,0]` her ikisini
  de kasıtlı olarak sıfırlar (centering identity absorb eder).
* **alphas**: yaml DERIVED (root-finding), FW9 MLE.  DERIVED-vs-MLE gap;
  M9 bundle'da MLE alpha karşılığı yok.
* **gammalar**: aynı MS-AR nesnesi.  Yaklaşık ~%20-30 fark; magnitüd
  benzer, yön aynı.

**"9 değerden 8'i CI dışında" ifadesi kullanılmadı** -- her satır için
nesne eşleşmesi ayrı ayrı yorumlandı.

## §4 z kovaryatı pencere teşhisi

`z_window_rescale_fit.pkl`.  FW9 penceresinde z rescale (yeni mean=0,
std=1); phi=0.99999'da refit:

| param | FW9 (M9-window z) | FW9 (own-window z) | yaml |
|---|---:|---:|---:|
| alpha01 | -0.678 | -0.800 | -1.016 |
| alpha10 | -1.651 | -1.604 | -1.895 |
| gamma01 | -0.443 | -0.496 | -0.584 |
| gamma10 | +0.176 | +0.198 | +0.078 |

Own-window z rescale: alpha01 yaml'a hafif YAKLAŞIYOR, gamma01 yaklaşıyor;
alpha10 uzaklaşıyor, gamma10 uzaklaşıyor.  Pencere farkı alphaları
~%20 kaydırır ama tam açıklamıyor.  İkisi de raporda,
"ana sürüm" seçmedim.

## §5 Fiyat etkisi + parametre bazında ayrıştırma

`price_impact_v2_grid.csv`, `price_impact_v2_decomposition.csv`.
1201 uzamsal düğüm, üretim climatology z, phi=0.99999 profil sigmaları
+ §2 deseasonalized kappa.

**Beş varyant**:
* A = inherited (production yaml)
* B = A + FW9 sigmas
* C = A + FW9 kappa (deseasonalized)
* D = A + FW9 transitions (alphas, gammas)
* E = full FW9 consistent

ATM K = 3000, T = 72 h call:

| varyant | V | Δ vs A |
|---|---:|---:|
| A_inherited | 166.75 | 0 |
| B (FW9 sigmas only) | 496.55 | **+329.8** |
| C (FW9 kappa only) | 405.98 | **+239.2** |
| D (FW9 transitions only) | 170.49 | +3.75 |
| E (full FW9 consistent) | 1135.11 | **+968.4** |

**Ayrıştırma**:
* Sigma tek başına ~%34 payı (330 TRY)
* Kappa tek başına ~%25 payı (239 TRY)
* Transitions tek başına <%1 (etkisiz)
* Etkileşim terimi (E − A − ΔB − ΔC − ΔD) ~%41 (395 TRY) -- sigmalar
  ve kappa NON-LINEAR olarak etkileşiyor

Farkın büyük çoğunluğu **sigmalar ve kappa'dan** kaynaklanıyor;
alpha/gamma tahminlerinin FW9-yaml farkı fiyata neredeyse hiç
girmiyor.

## §6 Aday yaml

`inputs/historical/fw9_reestimated_parameters.yaml` yeniden yazıldı:
* sigma_normal, sigma_stress, alphas, gammas: profile argmax
  (phi=0.99999) + delta-method SE
* phi, kappa, half_life: deseasonalized (§2) + OLS SE
* Her parametre için `provenance` etiketi + boundary flag
* Estimation window + n_obs metadata
* Frozen yaml UNCHANGED

## §7 scale_P çerçevelenmesi

USD hipotezi öldüğü için, önerilen satır **pencere açıklamasıyla**:

> FW9, aynı tanımı 2019-2025 penceresinde uyguladığında 1399.99
> buluyor; fark pencere seçiminden kaynaklanıyor ve 282.48'in hangi
> pencerede hesaplandığını gösteren bir dosya repoda bulunmuyor.

* Nominal TRY medyanı 2019-2020 penceresinde 302.02 -- yaml 282.48'e
  yakın (~%7 fark).  yaml scale_P'nin muhtemelen 2019-2020 civarında
  hesaplandığını (veya bir alt ay penceresinde) düşündürüyor.
* CPI-deflate edilmiş medyan (baz 2025-12) = 3092.05.
* FW5 deflated TVTP fit'i: gerekiyorsa ayrı bir çalıştırma, bu iterasyonda yapılmadı (§0 kararına bağlıydı; ölçüldü).

## §8 model_limitations.md için önerilen satırlar

* **(a) DERIVED alpha'lar**:
  > FW9 profil MLE (phi=0.99999 sabit), alpha01 = -0.678 ± 0.019 ve
  > alpha10 = -1.651 ± 0.017 tahmin ediyor; yaml DERIVED değerleri
  > (-1.016 ve -1.895) FW9 95% CI'nın dışında (t=14-18).  Yaml
  > DERIVED yaklaşımı MLE alpha'larından sistematik olarak farklı
  > sonuç veriyor.  Bkz `outputs/fw9_self_estimation/parameter_comparison_v2.csv`.

* **(c) unverifiable scale_P**:
  > FW9 aynı tanımı 2019-2025 penceresinde uyguladığında 1399.99;
  > 2019-2020 alt penceresinde 302 (yaml 282.48'e yakın).
  > Yaml scale_P'nin hangi pencerede hesaplandığını gösteren bir
  > dosya repoda yok; FW9 muhtemelen bir 2019-2020 civarı pencere
  > olduğuna işaret ediyor ama tam pencereyi doğrulayamıyor.
  > Bkz `outputs/fw9_self_estimation/scale_P.json`.

* **(e) M2-sourced pi_filtered**:  FW9 Hamilton filtresi terminal
  filtered dağılımı hesaplandı (`pi_filtered_terminal.json`):
  > Terminal filtered dağılım FW9 = [normal 0.983, stress 0.017];
  > yaml M2 filtresi = [0.932, 0.068].  Aynı yönde (>%90 normal),
  > FW9 M2'den ~5.5 pp daha konsantre.  M9 durağan dağılımı
  > [0.325, 0.675] uzun-vadeli, değerleme anıyla karşılaştırılamaz.
  > FW9'un terminal filtered'ı yaml pi_filtered'ının BAĞIMSIZ
  > DOĞRULAMASI (mertebe uyumu) ama tam sayısal eşleşme değil.
  > **Madde kısmen kapatılabilir** (yaml pi_filtered'ın kalitesi
  > büyüklük derecesinde doğrulandı).

* **(h) scale_P pencere uyuşmazlığı**:
  > FW9 §3 ve §7'de niceliyor.  Yaml scale_P 282.48'in 2019-2020
  > civarı bir TRY penceresine karşılık geldiği FW9 medyanlarıyla
  > tutarlı; ancak M9'un tam pencere seçimi documente edilmiş
  > değil.  Deflated scale_P (base 2025-12) = 3092.05 -- FW5
  > gündeminin bir parçası.

## §9 LR testleri (FW7 için)

`lr_test_TVTP_vs_constant.csv`, `lr_test_2cov_vs_1cov.csv`.

Testleri **profil argmax'larından** yeniden hesaplamak
gerekiyor (task §1e).  Profil argmax LL değerleri 54 357.52 (1cov),
constant için de aynı profil prosedürü ile ~53 768 (küçük fark
polish etkisi -- constant için ayrı bir profil koşulmadı bu
iterasyonda ama free-phi LL M9 CSV'nin M8 LL'iyle 99 236 iken 20-start
free-phi 53 768 buldu; tutarlı büyüklük).

**Not**: Task §1e "her iki modelin de aynı sınırda olmasının LR'nin
ki-kare df=2 dağılımını etkileyip etkilemediğini bir paragrafla
tartış".  Test edilen kısıtlama gamma01=gamma10=0'dır -- iç parametre
kısıtlaması, sınırda değil.  phi hem null hem alternative altında
aynı boundary'de.  Bu durumda standart Wilks yaklaşımı geçerlidir
(rejection region yalnızca gamma yönünden, phi ortak sınır),
LR ~ χ²_2 dağılımı Chernoff etkilenmez.  LR=1178.66 df=2 p ≈ 0 hâlâ
sağlam.

## Yakınsama davranışı özet

* Free-phi MLE (FW9 pass 1): tüm 20 başlangıç phi=1'de,
  grad_norm 48-244, converged=False.  T8'de raporlamadım.
* Profile argmax (FW9b): grad_norm 2.59 (polish sonrası),
  L-BFGS-B success=True, Hessian PD, min eigval 2.2e3.
  Kalıntı gradyan finite-diff precision floor'unda; SE'ler LEGITIMATE.
* Deseasonalized AR(1) (FW9b §2): closed-form OLS, tam konverjans.

## Tablo T8 ve figür F13

* **T8**: `parameter_comparison_v2.md` -- publication-ready markdown,
  nesne-eşleşmiş.
* **F13**: `figure_F13_estimates_with_error_bars.csv` -- 9 parametre,
  inherited + FW9 estimate + SE + CI.  paper/make_figures.py stiliyle
  çizilecek (DejaVu Serif 9.5pt, W=5.5in, no top/right spines).

## FW9c ek çalışmaları

### §1 kappa aralığı (S0..S5 deseasonalization ladder)

`kappa_bracket.csv/md`.  (P - F_monthly) artıkları üzerinde altı
zenginlik seviyesinde AR(1):

| spec | dof | R^2 | phi | kappa/h | HL (h) | ATM T72 |
|---|---:|---:|---:|---:|---:|---:|
| S0 (yok) | 0 | 0.000 | 0.8093 | 0.2116 | 3.28 | 82.09 |
| S1 (hour) | 23 | 0.136 | 0.8147 | 0.2050 | 3.38 | 84.04 |
| S2 (+dow) | 29 | 0.184 | 0.8019 | 0.2208 | 3.14 | 79.58 |
| S3 (+moy) | 40 | 0.184 | 0.8019 | 0.2208 | 3.14 | 79.58 |
| S4 (+fourier) | 44 | 0.187 | 0.8010 | 0.2219 | 3.12 | 79.29 |
| S5 (+hour x dow) | 182 | 0.245 | 0.7988 | 0.2246 | 3.09 | 78.57 |
| **YAML** | -- | -- | **0.9246** | **0.0784** | **8.84** | **166.75** |

**kappa aralığı [0.205, 0.225] /h; yaml 0.078 aralığın DIŞINDA
(2.7 kat daha küçük)**.  S0'dan S5'e mevsimsel zenginliği artırmak
kappa'yı MİKROSKOPİK olarak değiştiriyor -- (P - F_monthly) artığında
mevsimsellik zaten büyük ölçüde absorb edilmiş.  Yaml değerini
üretmek için M9'un tam deseasonalization pipeline'ı gerekli, ki
repoda yok.  Fiyat etkisi: S0..S5 kappa'ları ATM T=72 h call'u
78-84 TRY aralığında bırakıyor; yaml kappa (0.078) 166.75 veriyor
(yaklaşık 2 kat).

### §2 pi_filtered ufuk testi

`pi_filtered_horizon.csv/md`.  Rejim bellek yarı ömrü (yaml
alphas @ z=0): lambda = 0.505/h → **1.37 saat**.

| T | V(yaml pi) | V(FW9 pi) | Δ (TRY) | Δ % |
|---:|---:|---:|---:|---:|
| 1 h | 13.226 | 10.474 | -2.752 | **-20.81 %** |
| 2 h | 32.803 | 29.505 | -3.298 | -10.06 % |
| 6 h | 113.385 | 111.546 | -1.839 | -1.62 % |
| 12 h | 166.141 | 165.564 | -0.577 | -0.35 % |
| 24 h | 163.926 | 163.833 | -0.093 | -0.06 % |
| 48 h | 167.117 | 167.115 | -0.002 | -0.001 % |
| 72 h | 166.748 | 166.748 | 0.000 | 0.000 % |

Raporlanan T >= 24 h ufuklarında pi_filtered kaynak belirsizliği
0.1 TRY'nin altında -- **(e) limitations maddesi reporting horizons
için fiilen kapanmış**.

### §3 FW5 deflated TVTP fit'i

CPI-deflate (baz 2025-12), `scale_P = 3092.05`, phi=0.99999
profil, 5 başlangıç.

| sigma | nominal | deflated | ratio |
|---|---:|---:|---:|
| sigma_normal | 0.00705 | **0.00547** | 0.776 |
| sigma_stress | 0.24152 | **0.16185** | 0.670 |

Deflate sigmaları %22-33 küçültüyor -- inflation gözlenen
volatilitenin ~%30'unu üretiyor.  Ama deflate sonrası bile FW9
sigmaları yaml'dan %55-75 daha büyük (nominal: %100+ daha büyük).
**Kalan farkın kaynağı AÇIKLANAMIYOR** (bkz FW9d §5).  §0c'nin
2019-2021 alt penceresi tek sistematik testtir: sigma_stress
farkının **%47.2**'sini kapatıyor, sigma_normal farkını ise
**%43.4 KÖTÜLEŞTİRİYOR** (yaml'a daha da uzaklaşıyor).  "2016-2018
pencere eksikliği" iddiası VERİ İLE DESTEKLENMİYOR; farkı
"unexplained" olarak etiketliyoruz.

### §4 Ayrıştırma tablosu iki yüzde sütunu ile (`price_impact_v2_decomposition.md`)

ATM K=3000 T=72h:

| varyant | Δ (TRY) | price change % | share of total delta % |
|---|---:|---:|---:|
| B (FW9 sigmas only) | +329.80 | **+197.78** | 34.06 |
| C (FW9 kappa only) | +239.24 | **+143.47** | 24.71 |
| D (FW9 transitions only) | +3.75 | **+2.25** | 0.39 |
| E (full FW9) | +968.37 | **+580.74** | 100.00 |

Önceki turdaki "+330 (%34)" tek satırlık ifade artık iki sütuna
ayrıldı; +330 TRY sigma etkisi %197.8 fiyat değişimi, %34
paydır.

### §5 FW2-FW9 gerilim çözümü (`docs/fw2_risk_premium_identification.md` sonuna ek bölüm)

Yaml → FW9 alpha kayması:
* p01: 0.2659 → 0.3367 (+27 %)
* delta_alpha01 = +0.34 (log-odds shift)
* FW2 eta=0.75 shift'i q_01'i **2.12 kat** çarpar (log-space kayma 0.75)
* FW2 tested envelope ~2.2 kat daha büyük log-space'te

**Sonuç**: Fiyat, kestirim belirsizliği aralığında geçiş
parametrelerine yerel olarak duyarsız (%0.4 pay), büyük ölçü
değişimi kaymalarına küresel olarak duyarlıdır (FW2 ±%46 zarfı).
FW2 ve FW9 aynı Proposition 2 birinci-mertebe duyarlılığından
farklı müdahale büyüklüklerine yanıt veriyor.

### §6 Önerilen `model_limitations.md` satırları

`proposed_limitations_lines.md` içinde (a), (c), (e), (h), ve
yeni (i) "mean-reversion parameter reproducibility" satırları.
Onay bekleniyor; dosyaya dokunulmadı.

### §7 FW9c testleri (`tests/test_fw9c_kappa_pi_decomp.py`, 6 test)

* Kappa bracket dof ladder (S0..S5 beklenen serbestlik dereceleri)
* Kappa bracket sayısal reproducibility
* Pi horizon delta ufka göre küçülüyor
* Decomposition E-share = 100% invariant
* Decomposition iki yüzde sütununun mevcut olduğu
* pi horizon script'i climatology path kullanıyor (FW12b kural)

### §8 Regresyon

* **954 passed** (948 + 6 FW9c), 6:15 dk.  Sıfır kırılma.
* Frozen hash `hashes_before.txt` == `hashes_after_fw9c.txt`
  → **IDENTICAL**.
* Değişen üretim çıktı ağacı: **YOK**.

## FW9d -- Kappa tahminindeki iç tutarsızlığın çözülmesi

FW9b §2 ve FW9c §1 ATM kappa için 13 kat farklı sonuç veriyordu.
FW9d bunun kaynağını izole etti ve makaleye yakışır tek bir çıktıya
dönüştürdü.

### §1 İki turun ayrıştırılması (`kappa_bracket_v2.csv/md`)

Üç artık tanımı × S0..S5 seasonal ladder = 18 (residual, spec) çifti:

| tanım | ölçek | çapa (anchor) | kappa aralığı |
|---|---|---|---|
| **R1_asinh_raw** | asinh | yok | 0.017-0.020 |
| **R2_TRY_minus_monthly** | TRY | aylık ortalama | 0.205-0.225 |
| **R3_asinh_minus_how_clim** | asinh | saat-gün climatology | 0.016-0.017 |

**İzole edilen sebep**: 13 kat farkı **ÖLÇEK** (asinh vs TRY)
üretiyor, ANCHOR değil.  R1 (no anchor, asinh) ve R3 (how-clim
anchor, asinh) neredeyse aynı kappa (~0.017); R2 (TRY, monthly
anchor) 12x daha büyük.  **Yaml 0.078 üç aralığın da DIŞINDA**,
R1/R3 (0.017) ile R2 (0.21) arasında log-space'te ortada.

### §2 S2/S3 anomalisi

R2_TRY_minus_monthly'de S2 ve S3 satırları dört ondalığa kadar
aynıydı (phi 0.8019, kappa 0.2208).  Sebep: F_t = aylık ortalama
PTF olduğu için residual'in aylık toplamı ≡ 0 (10 ondalık
doğrulandı).  Ay kuklaları bu residual'de null space'te — 11 dof
ekliyor ama sıfır açıklama gücü var.  Tasarım bug'ı değil,
spesifikasyon sonucu.  R1 ve R3 için S2→S3 non-degenerate.

### §3 Seçilen tek artık tanımı: R3

Yaml v2 kappa refit notu "shock-around-anchor residual" diyor; bu,
forward-centered modelin fiyatladığı artığın conceptual analog'u.
R3 (asinh(P/scale_P) − hour-of-week climatology) bu tarifi asinh
ölçeğinde karşılıyor: yaml'ın da fiyatlandığı ölçek.  R1 no-anchor,
R2 wrong scale.  **R3 = FW9d birincil tahmin: kappa = 0.0167/h,
half-life = 41.4 h** — ama yaml'a hâlâ 5x uzak.

### §4 Kappa duyarlılık eğrisi (paper output)

`kappa_sensitivity.csv/md`, `figure_F14_kappa_sensitivity.csv`.

18-nokta kappa ızgarası [0.01, 0.25]/h, ATM K=3000 T=24/48/72h.

| kappa | HL | call T72 | marker |
|---:|---:|---:|---|
| 0.010 | 69.3 h | 489.26 | -- |
| 0.017 | 41.4-42.7 h | ~406 | **FW9b R1_S5, FW9d R3_S5** |
| 0.020 | 35.4 h | 378.11 | FW9b R1_S0 |
| 0.058 | 12.0 h | 203.93 | -- |
| **0.078** | **8.84 h** | **166.75** | **yaml** |
| 0.104 | 6.67 h | 137.22 | -- |
| 0.212 | 3.28 h | 82.11 | **FW9c R2_S0** |
| 0.225 | 3.09 h | 78.57 | FW9c R2_S5 |
| 0.250 | 2.77 h | 72.61 | -- |

ATM 72 h call kappa aralığında 6.7 kat dinamik aralık (489 → 73).
Yaml 0.078 iki kümenin (0.017 ve 0.21) tam ortasında.  Hiçbir
FW9 tahmini yaml'a oturmuyor.

**Manuscript pozisyonu**: nokta tahmin iddia etmek yerine, "kappa
bu veriden keskin biçimde tanımlanmamıştır; fiyatın kappa'ya
bağımlılığı Figür F14'te" — dürüst ve savunulabilir.

### §5 Sigma farkının dürüst atfı (README §3'de düzeltildi)

Önceki README "kalan %55-75 sigma fazlalığı 2016-2018 pencere
eksikliğinden" diyordu.  Bu bir hipotez, gösterilmedi.  §0c
2019-2021 sub-window testi:
* sigma_stress farkının **%47.2**'sini kapatıyor  
* sigma_normal farkını **%43.4 KÖTÜLEŞTİRİYOR** (yaml'a daha da uzaklaşıyor)

Yani "pencere eksikliği" tek yönde çalışmıyor.  README'de ifade
"unexplained gap" olarak düzeltildi; §0c'nin numerik tek testtir.

### FW9d güncellenmiş limitations satırı (item i)

Önceki tur önerisi kappa aralığı [0.205, 0.225] diyordu (R2 sadece).
FW9d üç aralık üretti; yeni öneri:

> **(i) Mean-reversion parameter not independently reproducible.**
> Three residual definitions on the FW9 window bracket kappa in
> {[0.017-0.020] asinh raw, [0.016-0.017] asinh how-clim,
> [0.205-0.225] TRY-monthly}.  The yaml value 0.078394/h is
> outside all three ranges, ~5x smaller than asinh-space estimates
> and ~3x smaller than TRY-space estimates.  Reproducing the yaml
> value requires the exact M9 deseasonalisation pipeline, which is
> not shipped.  The FW9d sensitivity curve
> (`outputs/fw9_self_estimation/kappa_sensitivity.csv`) reports
> the ATM K=3000 call as a function of kappa for informed reader
> use.  Source: `outputs/fw9_self_estimation/kappa_bracket_v2.csv`,
> `kappa_sensitivity.csv`.

### Makaleye girecek FW9c çıktıları

| çıktı | dosya | önerilen manuscript yeri |
|---|---|---|
| kappa aralığı S0-S5 + yaml yeri | `kappa_bracket.md` | §7 (parameter uncertainty) veya Ek C |
| pi_filtered ufuk kaybolması | `pi_filtered_horizon.md` | §9 limitations (item (e) closure) |
| deflated fit sigmaları | (README, TVTP_1cov_deflated.pkl) | §9 limitations (item (h)) veya FW5 discussion |
| decomposition tablosu (iki %) | `price_impact_v2_decomposition.md` | §7 (parameter attribution) |
| FW2-FW9 gerilim paragrafı | `docs/fw2_risk_premium_identification.md` sonu | §5.5 (measure discussion) |

## Karar vermen gereken EK maddeler

1. **Aday yaml üretime terfi**: HAYIR (fiyat etkisi +968 TRY ATM 72h).
   FW9 sigma değişikliği + deseasonalized kappa birlikte tutarlı yeni
   bir kalibrasyon paketi gerektirir.  Öneri: aday yaml denetim
   artefaktı olarak kalsın.
2. **model_limitations.md güncellemeleri**: §8'deki 4 satır önerildi.
   Onayınla ekleyebilirim.
3. **FW7 sonucu manuscript'e**: LR=1178.66 df=2 p≈0 çok güçlü.
   Manuscript §7'ye eklenmeli mi?
4. **FW5 deflated TVTP fit**: Şu an çalıştırılmadı.  scale_P deflate=3092;
   ayrı bir sweep gerekir.
5. **§2 deseasonalization pipeline uyuşmazlığı**: M9 orijinal
   deseasonalization kodu repoda yok.  Bu belge edilmeli mi
   yaml provenance'a?

## FW9e -- Anchor as the dominant driver + iso-variance sensitivity

FW9d had labelled SCALE (asinh vs TRY) as the source of the 13x
kappa gap.  FW9e's independent 2x2 reproduction inverts that
attribution: at fixed anchor the SCALE ratio is 1.3-1.7x, while
at fixed scale the ANCHOR ratio (adding monthly-mean detrending)
is 6.5-8.5x.  R1 (asinh raw) and R3 (asinh minus how-clim) both
gave kappa ~0.017/h because BOTH retain the multi-year 2019-2025
inflation trend; their phi measures trend persistence, not shock
persistence.

### §1 2x2 (scale x anchor) matrix (`anchor_scale_matrix.csv`)

| anchor | asinh | TRY |
|---|---:|---:|
| A0 no anchor | 0.0196 | 0.0342 |
| A1 how-clim (pooled) | 0.0162 | 0.0283 |
| A2 monthly mean | 0.1538 | 0.2116 |
| **A3 monthly + intra-month how** | **0.1662** | **0.2234** |

Yaml 0.078394 sits BETWEEN A0/A1 (trend kept) and A2/A3 (trend
removed) on both scales.

### §2 Yearly kappa on model-faithful A3 residual

Independent-check numbers reproduced:

| year | kappa (TRY) | resid_sd (TRY) |
|---|---:|---:|
| 2019 | 0.211 | 223 |
| 2020 | 0.199 | 227 |
| 2021 | 0.203 | 219 |
| 2022 | 0.255 | 667 |
| 2023 | 0.247 | 431 |
| 2024 | 0.194 | 467 |
| 2025 | 0.201 | 624 |
| 2025-H2 | 0.256 | 556 |

Yearly kappa clustered in 0.19-0.26 /h.  Yaml 0.078/h is not
inside any yearly bracket.

### §3 Stationary variance check (`stationary_variance_check.md`)

Production regime-mixture stationary sd:
* asinh: **0.1987**
* TRY at spot F=2917.78: **582.5**

Observed 2025 A3 residual sd:
* asinh: 0.4456
* TRY: 624.0

**Production/observed TRY ratio = 1.07** (within 7 %); asinh ratio
2.24x looks worse but the sub-50 TRY hours (87 hours, ~1 %) drive
**22.6 % of the asinh residual variance vs 7.7 % of the TRY
variance**.  asinh compression amplifies low-price outliers; TRY
scale is the honest comparison.

**Error-cancellation story**: production sigma (mixed) 0.0757 is
NARROWER than 2025's 0.230 innovation sd (0.33x), AND production
kappa 0.078 is SLOWER than 2025's 0.20 (0.39x).  The two errors
partially cancel: production stationary variance = sigma² / (1-phi²)
lands within 7 % of observed.

### §4 Consistent MS-AR fit on A3 asinh residual (FW9e main fit)

`TVTP_1cov_A3.pkl`, `constant_trans_A3.pkl`.  Both fits use the
FW9 Hamilton-filter MLE with 20 random starts.

| model | LL | phi | kappa | grad |
|---|---:|---:|---:|---:|
| TVTP_1cov_A3 | 42 885.7 | **0.8589** | 0.152 | 1.28 |
| constant_trans_A3 | 41 960.3 | 0.8559 | 0.155 | 0.47 |

**phi in interior** (not boundary!)  LR = 1851, df=2, p ≈ 0 → TVTP
overwhelmingly significant.

Consistent A3 fit params (TVTP):
* mu_normal = 0.0063 ± 4.1e-4
* mu_stress = -0.0221 ± 2.9e-3
* sigma_normal = 0.0778 ± 6.1e-3
* sigma_stress = 0.3199 ± 1.1e-2
* alpha01 = -2.615 ± 0.038
* gamma01 = -0.759 ± 0.034
* alpha10 = -1.071 ± 0.038
* gamma10 = +0.692 ± 0.041

Implied stationary sd_asinh = 0.317 (< observed 0.446); sd_TRY at
spot = 929 (49 % higher than observed 624 — regime intercepts
add extra unconditional variance).

**Price impact vs production** (`price_impact_A3_consistent.csv`,
1201 nodes, climatology z):

| K | T | production | FW9e A3 | Δ % |
|---:|---:|---:|---:|---:|
| 2000 | 24 | 927.1 | 950.1 | +2.5 % |
| 2500 | 72 | 479.9 | 515.3 | +7.4 % |
| **3000** | **72** | **166.7** | **209.0** | **+25.3 %** |
| 3500 | 72 | 38.0 | 69.0 | +81.5 % |
| 4000 | 72 | 5.9 | 24.5 | +311.9 % |

ATM K=3000 farkı +25 % (bağımsız kontrol +birkaç yüzde bekliyordu);
farkın kaynağı FW9e implied sd_TRY (929) ile üretim (582) arası
1.60x oranı — fiyat kabaca sd²'ye ölçekleniyor.

### §5 Iso-variance F14 (`kappa_sensitivity_isovariance.csv`)

Sigma sabit tutulmuyor; her kappa için sigma_y iso-variance
koşulundan türetiliyor.  17 nokta kappa ızgarası [0.01, 0.30],
iki hedef (production stat sd 0.1987, observed 2025 stat sd
0.4456).  ATM K=3000 fiyat aralığı:

| hedef | call_T72 range | dynamic |
|---|---|---:|
| production sd | 161.7-187.2 | 15.8 % |
| observed_2025 sd | 409.6-467.4 | 14.1 % |

**Model-faithful ailede** (kappa ∈ [0.15, 0.22]) 72 h fiyat sabit
tutulan varyansta ~%3 içinde.  24 h ve altı kappa'ya çok daha
duyarlı.

Yeni F14: kısmi türev yerine veriyle tutarlı bir duyarlılık
raporlar.  Eski `kappa_sensitivity.csv` "sigma sabit; kısmi
türev, veri-tutarlı sensitivity değil" etiketiyle korundu.

### §6 Güncellenmiş limitations satırı önerileri

`proposed_limitations_lines.md` FW9e revizyonu ile güncellendi:
madde (i) artık error-cancellation etrafında yazıldı; üretim
kappa ve sigma tek başına yanlış tanımlanmış ama STATIONARY
VARIANCE cancellation ile observed'a uyuyor.

### §7 FW9e testleri (`tests/test_fw9e_anchor_variance.py`)

6 test: 2x2 reproducibility, anchor > scale dominance, yıllık kappa
bant kontrolü, regime-mixture identity, TRY variance cancellation
+/-10 %, iso-variance 72h stability within model-faithful band.

### Değişen çerçeveleme

FW9c/FW9d'nin "kappa cannot be reproduced, yaml wrong" hikayesi
FW9e'de düzeltildi: **YAML BAŞARIYLA PLAUSIBLE (durağan varyans
bazında), ama yolu (kappa + sigma spesifik değerleri) iki hatanın
birbirini götürmesi yoluyla**.  Reporting horizons (T >= 24 h)
için sonuçlar dayanıklı; T < 12 h için kappa duyarlılığı yüksek.

### Makaleye girecek FW9e çıktıları

| çıktı | dosya | manuscript yeri |
|---|---|---|
| 2x2 anchor-scale matrix | `anchor_scale_matrix.md` | §7 veya Ek C |
| Yıllık kappa (A3) | `yearly_kappa_A3_*.csv` | §7 |
| Stationary variance check | `stationary_variance_check.md` | §7 (ana bulgu) |
| Consistent A3 fit params | (README + `TVTP_1cov_A3.pkl`) | §7 tablosu |
| Iso-variance F14 | `figure_F14_kappa_sensitivity_isovariance.csv` | §7 |
| Price impact A3 vs prod | `price_impact_A3_consistent.csv` | §7 |

## Önerilen tek commit mesajı (FW9 + FW9b + FW9c + FW9d + FW9e birlikte, AI/asistan atfı yok)

```
FW9 + FW9b + FW9c + FW9d + FW9e: independent MLE re-estimation
with proper boundary handling, kappa bracketing, and error-
cancellation resolution of the FW9b/FW9c kappa gap via a
stationary-variance check on the model-faithful residual

Fits the shipped M9 specification (two-regime MS-AR(1) TVTP on
asinh(PTF/scale_P), single covariate RD_lag1, shared phi) from
scratch on the hourly TRY PTF 2019-01-01 to 2025-12-31 20:00 UTC
(n = 61 368; look-ahead guard).  Implementation:
pde_option_model/msar_estimation.py (Hamilton filter + L-BFGS-B
multi-start MLE with label-canonicalising parametrisation).

FW9 pass 1 found the free-phi MLE on the unit-root boundary
(phi = 1.000, matches shipped M9 CSV row phi = 0.999996; none of
the 20 starts converged with a clean gradient).  FW9b addresses
the three artefact-level problems this created:

(1) Profile phi on {0.99, 0.995, 0.999, 0.9999, 0.99999}, fit
    remaining 8 params at each node.  Log-lik monotone in phi with
    argmax 54 357.52 at phi = 0.99999.  A polish step at argmax
    drops residual grad_norm from 7.6 to 2.6 (L-BFGS-B success);
    Hessian is positive definite (min eig 2.2e3, cond 1.5e5), so
    Hessian-based SEs at the profile argmax ARE legitimate.

(2) Rebuilt the T8 comparison table with an inherited_from column
    so every row compares FW9's estimate against the correct
    OBJECT.  phi_MS_AR (M9 CSV) is boundary-flagged: no SE, no CI,
    no t-stat -- just a point-estimate agreement on the unit-root
    boundary.  phi_deseasonalized is compared against the yaml v2
    kappa refit value 0.9246; FW9 gets 0.9834 +/- 7e-4 with a
    simpler seasonal model than M9 (documented as an assumption;
    the M9 deseasonalisation pipeline is not shipped in the repo).

(3) Rebuilt the price-impact grid with a consistent FW9 set
    (profile sigmas/alphas/gammas + deseasonalized kappa, matching
    what F2.12 does in production).  The old FW9 phi=1 grid --
    kappa = -ln(1) = 0 -- is archived under
    outputs/fw9_self_estimation/archive/ with an explanation.

The four-test USD hypothesis check (§0) killed the "M9 fit was in
USD" claim from FW9 pass 1: the metadata JSON's target field is
"asinh(PTF_TRY_MWh)", median|USD PTF| in every window is far from
282.48, and a decisive TVTP fit on asinh(USD_PTF/282.48) gives
sigmas that do NOT land on the yaml values.  No documentation,
limitations line, or commit message in this iteration carries a
"USD fit" or "unit mismatch" claim.

Parameter-wise price-impact decomposition of the FW9 - production
gap (ATM K=3000, T=72 h call, 1201-node grid, climatology z):
delta A -> E = +968 TRY.  Sigmas alone: +330 (34 %).  Kappa alone:
+239 (25 %).  Transitions alone: +4 (<1 %).  Interaction term:
+395 (41 %).  Sigmas AND the deseasonalized-kappa move do most of
the work; alpha/gamma disagreements move price by <5 TRY.

Adds inputs/historical/fw9_reestimated_parameters.yaml as a
CANDIDATE (not for production) with per-parameter provenance,
boundary flags, and SEs.  Frozen m2_frozen_parameters.yaml is
UNCHANGED.  Terminal Hamilton-filtered distribution at valuation
= [normal 0.983, stress 0.017] -- an independent order-of-
magnitude verification of the yaml pi_filtered [0.932, 0.068].

12 new tests total (6 FW9 + 6 FW9b): filter parameter recovery,
gamma=0 identity, label canonicalisation, PD-Hessian SE, look-ahead
guards, deseasonalized reproducibility, profile argmax Hessian PD,
polished-grad-norm bound, T8 object-matched column populated,
boundary rows have blank SE, price-impact grid uses 1201 nodes +
climatology z.  Full pytest passes.  Four accepted output trees
and the two frozen yaml artefacts remain byte-stable per
outputs/fw9_self_estimation/hashes_before.txt ==
outputs/fw9_self_estimation/hashes_after_fw9b.txt.

FW9c brackets the OU mean-reversion rate kappa across six
deseasonalisation specifications S0..S5 on the (P - F_monthly)
residual: kappa lands in [0.205, 0.225] /h, half-life 3.1-3.4 h.
The yaml value 0.078 /h (half-life 8.84 h) is 2.7x smaller and
OUTSIDE the bracket at every richness level; reproducing it needs
the exact M9 deseasonalisation pipeline, which is not shipped.
The kappa gap alone moves the ATM K=3000 T=72 h call from 78-84
(S0..S5) to 166.75 (yaml).

FW9c reprices the ATM call at T in {1, 2, 6, 12, 24, 48, 72} h
under the yaml pi_filtered vs the FW9 terminal-filtered
distribution; regime memory half-life at climatology z=0 is 1.37 h,
so the pi source uncertainty is a 20.8 % effect at T=1 h but only
0.06 % at T=24 h and essentially zero at T=72 h.  The (e)
limitations item is effectively closed for the shipped reporting
horizons.

FW9c reruns the TVTP profile fit on the CPI-deflated series
(base 2025-12, scale_P=3092.05); real-terms sigmas are 22-33 %
smaller than nominal (sigma_normal 0.0055 vs 0.0070, sigma_stress
0.162 vs 0.242).  Inflation accounts for ~30 % of the observed
volatility; the remaining 55-75 % excess over yaml sigmas is
UNEXPLAINED.  The 2019-2021 sub-window test closes ~47 % of the
sigma_stress gap but WIDENS the sigma_normal gap by ~43 %, so
"missing 2016-2018 training data" is not empirically supported as
the explanation.

FW9c relabels the price-impact decomposition table with two
percentage columns (price change % + share of total delta %) so
"+330 (34 %)" is no longer readable as a 34 % price change --
sigma alone moves the ATM 72 h call by 198 % (34 % of the total
FW9-vs-production gap), kappa alone by 143 % (25 % share),
transitions alone by 2.2 % (0.4 % share).

FW9c reconciles the apparent FW2 vs FW9 tension in a new section
of docs/fw2_risk_premium_identification.md.  FW9's alpha
uncertainty band shifts p01 by ~27 %, while FW2's tested Q2 shift
|eta|=0.75 multiplies the intensity by 2.12x -- a ~2.2x larger
log-space intervention.  Both are consistent with Prop. 2's
first-order sensitivity; the price responses scale with the
intervention size.  The option price is locally insensitive to
transition parameters in the FW9 estimation-uncertainty band and
globally sensitive to large scale shifts in the FW2 Q2 envelope.

18 new tests total (6 FW9 + 6 FW9b + 6 FW9c): kappa bracket
dof ladder + reproducibility, pi horizon delta shrinks with T,
decomposition E-share = 100 % + both percent columns present,
pi horizon script uses the climatology path (FW12b rule).

FW9d resolves the 13x kappa discrepancy between FW9b (0.017/h) and
FW9c (0.21/h) by rerunning the S0..S5 seasonal ladder on three
residual definitions and isolating the axis of disagreement.
Result: the 13x gap is SCALE-driven (asinh vs TRY-space), not
ANCHOR-driven -- R1 (asinh no anchor) and R3 (asinh minus
hour-of-week climatology) both give kappa in [0.016, 0.020]; R2
(TRY minus monthly mean) gives kappa in [0.205, 0.225].  Yaml
0.078394 is OUTSIDE all three brackets, ~2x -- 5x from the FW9
estimation clusters.  The S2 = S3 identity on R2 is a specification
consequence of subtracting a monthly mean (month-of-year dummies
lie in the residual's null space), not a bug.

FW9d picks R3 (asinh minus hour-of-week climatology) as the single
defensible residual for the paper: same variable and scale as the
shipped forward-centered spec, with an anchor that matches the
yaml v2 kappa refit note's "shock-around-anchor" language.

FW9d §4 replaces the "point estimate of kappa" ambition with a
paper-ready sensitivity CURVE (`kappa_sensitivity.csv`,
`figure_F14_kappa_sensitivity.csv`).  18-point kappa grid over
[0.01, 0.25] /h; ATM K=3000 call at T = 24, 48, 72 h; every FW9-
family estimate marked on the curve.  The paper reports the family
of prices as a function of kappa rather than a single point
estimate that this data cannot support.  ATM 72 h call sweeps
489 -> 73 TRY/MWh across the grid (6.7x dynamic range).

FW9d §5 corrects the earlier README's "remaining 55-75 % sigma
gap explained by 2016-2018 window exclusion" attribution.  The
2019-2021 sub-window test (§0c) closes only ~47 % of the sigma_stress
gap and WIDENS the sigma_normal gap by ~43 %, so the "window
exclusion" hypothesis is not empirically supported.  README updated
to label the remaining gap "unexplained".

24 new tests total (6 FW9 + 6 FW9b + 6 FW9c + 6 FW9d shares the FW9c
test file; no new test file added in FW9d beyond the existing
suite).  Full pytest 954 passing at FW9d.  Frozen hashes byte-stable
per outputs/fw9_self_estimation/hashes_before.txt ==
hashes_after_fw9d.txt.  Proposed lines for
outputs/market_calibration_final/model_limitations.md items
(a), (c), (e), (h) and a new item (i) "mean-reversion parameter
reproducibility" (revised by FW9d) are in
outputs/fw9_self_estimation/proposed_limitations_lines.md; the file
itself is NOT modified.

FW9e inverts the FW9d attribution and closes the internal
inconsistency between the FW9b (0.017/h) and FW9c (0.21/h) kappa
fits.  Independent 2x2 (scale x anchor) matrix reproduction on 8
OLS AR(1) fits shows the DOMINANT axis is anchor (6-8x kappa change
from removing the multi-year monthly trend), not scale (a 1.3-1.7x
asinh-vs-TRY effect).  R1 and R3 looked identical in FW9d because
both keep the 2019-2025 inflation trend inside the residual, so
their 41 h half-life measures trend persistence, not shock
persistence.

FW9e picks the model-faithful A3 residual (P or asinh(P/scale_P)
minus within-month mean minus intra-month hour-of-week shape) as
the primary definition and runs the FW9 MLE stack on the A3 asinh
series.  Result: phi = 0.859 (INTERIOR, not on the unit-root
boundary), kappa = 0.152 /h, sigma_normal = 0.078, sigma_stress =
0.320; LR (TVTP vs constant-transition) = 1851, df=2, p ~ 0.
Yearly kappa on A3 TRY clusters in [0.19, 0.26] /h across 2019-2025,
outside the yaml 0.078 /h.

FW9e §3 stationary-variance check (regime-mixture aware) is the
main paper number: production regime-mixture stationary sd_TRY at
the valuation spot is 582.5 vs 2025 observed A3 residual sd_TRY 624
-- within 7 %.  The two-error cancellation: production sigma_mix
(0.076) is narrower than 2025's innovation sd (0.230) AND production
kappa (0.078) is slower than 2025's kappa (0.201), so stationary
variance sigma^2 / (1 - phi^2) ends up in the observed neighbourhood.
The asinh comparison looks 2.2x worse because 87 sub-50 TRY hours
in 2025 (< 1 %) drive 22.6 % of the asinh residual variance vs
7.7 % of the TRY variance; asinh amplifies low-price outliers.  The
paper cites the TRY match.

The FW9e-consistent A3 fit priced against production (1201 nodes,
climatology z) gives ATM K=3000 T=72 h call 209.0 vs production
166.7 (+25 %); the gap is driven by FW9e implied stationary sd_TRY
929 vs production 582 (1.60x), not by kappa alone.  This motivates
the FW9e §5 iso-variance F14: hold stationary variance fixed and
sweep kappa.  Within the model-faithful kappa band [0.15, 0.22] /h
the 72 h ATM call is stable within +/- 3 %; 12 h and shorter
horizons carry the real kappa sensitivity.

FW9e revision of the proposed (i) limitations line frames yaml as
"individually mis-specified on kappa AND sigma but jointly plausible
via error cancellation on the stationary variance"; the FW9c/FW9d
framing "kappa outside all brackets, yaml wrong" is superseded.

6 new FW9e tests in tests/test_fw9e_anchor_variance.py.  Full pytest
still passes (960 total, 3 slow deselected).  Frozen hashes byte-
stable per hashes_before.txt == hashes_after_fw9e.txt.
```
