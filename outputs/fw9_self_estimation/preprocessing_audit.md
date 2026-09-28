# FW9 §0 -- Önişleme denetimi (audit)

## Kestirim verisi envanteri

| dosya | satır | tarih aralığı | eksik saat | NaN |
|---|---:|---|---:|---:|
| `inputs/historical/ptf_raw/ptf_2019.csv` | 8760 | 2019-01-01 → 2019-12-31 | 0 | 0 |
| `inputs/historical/ptf_raw/ptf_2020.csv` | 8784 | 2020-01-01 → 2020-12-31 | 0 | 0 |
| `inputs/historical/ptf_raw/ptf_2021.csv` | 8760 | 2021-01-01 → 2021-12-31 | 0 | 0 |
| `inputs/historical/ptf_raw/ptf_2022.csv` | 8760 | 2022-01-01 → 2022-12-31 | 0 | 0 |
| `inputs/historical/ptf_raw/ptf_2023.csv` | 8760 | 2023-01-01 → 2023-12-31 | 0 | 0 |
| `inputs/historical/ptf_raw/ptf_2024.csv` | 8784 | 2024-01-01 → 2024-12-31 | 0 | 0 |
| `inputs/historical/ptf_raw/ptf_2025.csv` | 8760 | 2025-01-01 → 2025-12-31 23:00 TRT | 0 | 0 |
| `inputs/historical/rd_lag1_standardized.csv` | 87665 | 2016-01-01 01:00 UTC → 2025-12-31 20:00 UTC | 1 (DST-related) | 0 |
| `inputs/historical/rd_standardized.csv` | 87665 | 2016-01-01 01:00 UTC → 2025-12-31 20:00 UTC | 1 | 0 |

Doğrulama: `rd_lag1_standardized[t] == rd_standardized[t-1]` her t için sabit
(1 saat gecikmesi doğru).  `rd_standardized`'ın örneklem istatistikleri
mean=0.15, std=1.08 — bu STANDARDİZE edilmiş serinin train penceresinin
DIŞINDA olduğunu gösteriyor (train penceresinde tanımlı olarak
mean=0, std=1 olmalıydı).  M9 metadata'sına göre train penceresi
2016-01 → 2022-12-31 (bkz. `run_summary.n_train = 78905`, `n_total = 87665`).

## Üretim önişleme zinciri (repodan tespit)

Kaynak: `inputs/historical/archive/calibration_bundle/metadata/model_parameters_and_ou_mapping.json`.

* **Hedef değişken**: `target: "asinh(PTF_TRY_MWh)"` (JSON'un 3. satırı).
* **Dönüşüm**: `y_t = asinh(P_t / scale_P)` -- `scale_P = 282.48 TRY/MWh`,
  provenance'ı `m2_frozen_parameters.yaml` içinde "ASSUMED: training median
  absolute price" olarak etiketli.  Repoda bu medyanın hangi pencerede
  hesaplandığını gösteren bir hesap dosyası YOK; sadece 282.48 sayısı taşınıyor.
* **AR(1) yapı**: JSON'a göre iki rejim, rejime özgü sabit terim `mu_i` ve
  sigma `sigma_i`, PAYLAŞILAN `phi`.  Zaman adımı 1 saat
  (`time_step_hours: 1.0`).
* **TVTP kovaryatı**: `RD_lag1` (standartlaştırılmış artık talebin 1 saat
  gecikmesi).  Yaml içindeki `covariate: RD_WS_standardized_lag1h` bu
  seriyi işaret ediyor.
* **Deseasonalization**: JSON'da y_t üzerine mevsimsellikten arındırma
  UYGULANMIŞ bir referans YOK.  Tam tersine `AR_phi=0.9247` sayısı
  `metadata\deseasonalized_stationarity_summary.csv`'ten geliyor ve
  "deseasonalized" olarak etiketli -- bu FARKLI bir istatistik, sabit
  geçişli tek rejim AR(1)'inde ölçülmüş.  M9 iki rejimli MS-AR(1) fit'i
  `parameter_estimates.csv`'de `phi=0.999996` -- bu neredeyse tam
  birim-kök, mevsimselliği absorb ediyor.
* **v2 kappa refit**: Üretim yaml'ı bugün `phi=0.9246, kappa=0.0784/h`
  taşıyor (F2.12).  Bu, M9 CSV'sinin `phi=0.999996` değerini deseasonalized
  değere göre "reconcile" ediyor -- SIGMALAR değişmeden, sadece
  mean-reversion hızı.

**FW9'un kestirim spesifikasyonu**: `y_t = asinh(P_t / scale_P)`,
`P_t` ham hourly PTF (2019-01-01 00:00 TRT → 2025-12-31 23:00 TRT,
UTC'de 2018-12-31 21:00 → 2025-12-31 20:00), **deseasonalization
YOK**, iki rejimli MS-AR(1) with TVTP (kovaryat = `RD_lag1`), paylaşılan
`phi`.  Bu, M9 CSV satırının fit ettiği spesifikasyonun aynısı.

## Ne repodan yeniden üretilemiyor

* **`scale_P`'nin tam olarak hangi pencerede hesaplanmış olduğu**.  Yaml
  "training median absolute price" diyor ama pencereyi vermiyor.  FW9'da
  `scale_P` REYENİ olarak look-ahead-guarded pencerede (2019-01-01 →
  2025-12-31 20:00 UTC) hesaplanıp devralınan 282.48 ile karşılaştırılacak.
* **M9'un tam train penceresi**.  Metadata `n_train=78905` gösteriyor ama
  başlangıç/bitiş yok.  n_total=87665 ve seri 2016-01'de başlıyor,
  n_total-n_train=8760 = 1 yıl → M9 muhtemelen 2016-01 → 2024-12-31 pencerede
  train edildi (kalan 8760 saat val split).  FW9 tam PTF penceresinde
  (2019-01-01 → 2025-12-31, look-ahead cutoff'ta) fit edecek; M9'un pencere
  seçiminden farklı bir pencere olduğu için doğrudan karşılaştırma değil
  bağımsız verilere göre yeniden kestirim.
* **`mu_i` regime intercepts**: M9 CSV `mu0/mu1` taşıyor ama üretim yaml'ında
  `regime_means: [0, 0]` -- kasıtlı olarak sıfırlanmış (forward-centered
  modelin merkezleme kimliği bunu absorb ediyor).  FW9 kestirimi mu_i'yi
  FREE PARAM olarak tutacak; karşılaştırmada mu_i M9 değerleriyle
  eşleştirilecek.

## Notlar

* Look-ahead cutoff: 2025-12-31 20:00 UTC (= 2025-12-31 23:00 TRT).
  Kestirim penceresinin son saati bu.
* PTF verisi 2019'da başlıyor (2016-2018 yok); M9'un 2016'dan başladığı
  metadata'dan biliniyor.  Bu bir look-ahead değil, örneklem boyutu farkı --
  M9 daha uzun bir seri gördü.
* Sentetik yol sadece §1'de filtre doğrulaması için kullanılacak; hiçbir
  SONUÇ türetilmeyecek.
