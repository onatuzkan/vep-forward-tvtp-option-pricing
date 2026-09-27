# FW12 §0.2 -- Kalem-kalem: FW2 taraması vs üretim ayarları

## Rakamsal bulgu (ATM K=3000, T=72 h call)

| Ayar | FW2 script | Üretim (`run_pde.py price`) | Δ |
|---|---:|---:|---:|
| Fiyat | 179.6485 | 166.7477 | **-12.9008 TRY/MWh** |
| |Δ|/Üretim | | | 7.74 % |

## Ayrıntılı ayar karşılaştırması

| kalem | FW2 | üretim | fark? |
|---|---|---|---|
| `n_space_nodes` | **601** | 1201 | **evet** |
| `n_time_steps` | None → `max(96, 2*τ)` = 144 (τ=72h) | aynı | hayır |
| `n_std` (boundary halfwidth × σ) | 6.0 | 6.0 | hayır |
| `min_halfwidth_TRY_MWh` | 500.0 | 500.0 | hayır |
| `max_halfwidth_TRY_MWh` | 60000.0 | 60000.0 | hayır |
| Boundary kind | default (`bc=None` -> gamma_zero-x) | aynı | hayır |
| Regime coupling (2×2 CTMC) | production yaml pi_filtered | aynı | hayır |
| Curve mode | `smooth_constrained` | `smooth_constrained` | hayır |
| Discount r_annual | 0.40 | 0.40 | hayır |
| **`z_lagged` covariate path** | **None → zeros (silent fallback)** | **climatology** from `rd_standardized.csv` | **EVET** |
| Q1 drift shift a_i | (0, 0) | (0, 0) | hayır |
| Q2 eta_ij | None | None | hayır |

## Farkın kaynağı — deneysel ayrıştırma

Aynı 1201 üretim düğüm sayısında iki senaryonun fiyatı:

| kovaryant yolu | 601 düğüm | 1201 düğüm | 2401 düğüm |
|---|---:|---:|---:|
| **climatology (üretim)** | 166.8415 | **166.7477** | 166.7070 |
| **z = 0 (FW2 default)** | 179.6485 | **179.5586** | (— sweep yapmadım) |

Sonuç: 7.74% farkın **çok büyük çoğunluğu** (~12.8 TRY/MWh) `z_lagged` yolundan
geliyor, sadece ~0.09 TRY/MWh (0.05 %) 601 → 1201 düğüm iyileştirmesinden.

## Kök neden

`ForwardCenteredModel.resolve_covariates`, tek-kovaryant modda
`z_lagged` girdisi verilmezse SESSİZ olarak `zeros(t.size)`'a
düşüyor.  Bu tamamen geçerli bir kod yolu (single-covariate default
zeroth-order fallback), ancak ÜRETİM `run_pde.py price` bu yolu
KULLANMIYOR — `_build_tvtp_scenario` üzerinden climatology z(t)
yolunu inşa edip `z_lagged_fn` olarak geçiyor.  FW2 scriptinde
climatology yolu inşa edilmemişti; dolayısıyla FW2'nin baseline 179.65
sayısı, üretimin fiyatladığı modelden FARKLI bir modeli (z ≡ 0)
fiyatladı.

Bu, düğüm yakınsaması iddiası ile ilgili DEĞİL; iki farklı model
çözümünü aynı isim altında toplamış olmakla ilgili.  FW12 bu iş
paketinde her sweep noktasını üretim climatology yolunu kullanacak
şekilde yeniden inşa ediyor (bkz `scripts/fw12/_shared.py`
`climatology_z_lagged_fn`).

## Anlamı — Fıkra 1

FW2'nin sensitivity_summary.csv'de raporladığı YÜZDE etkileri (Q1 +3.9 %,
Q2 ±27 %, joint ±35 %) hâlâ **kendi tabanına göre doğru** (yüzde bir
başka fiyatın diğer bir fiyata bölümü), ancak TABAN 166.75 değil 179.65
idi.  Etkinin ABSOLÜT büyüklüğü de kendi bağlamında geçerli.
Ama makaleye giren "üretim ATM K=3000 72h call = 166.75'e karşı
sensitivity envelope" cümlesi için FW12 §6'da bu taramayı üretim
climatology yolu + 1201 (veya yakınsak) düğümde yeniden koşacağım.

## Anlamı — Fıkra 2

Üretim değerinin yakınsak olduğu iddiası FW12 §1-§3'te ölçülüyor.
Yukarıdaki climatology sütununa dayanarak ön kestirim:
1201 düğüm ile 2401 düğüm farkı 0.04 TRY/MWh (~0.02 % 166.75'in) →
üretim değeri kaba olarak 0.05 % dahilinde yakınsak.  Kesin sayı §1'de.
