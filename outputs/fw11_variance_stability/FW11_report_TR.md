# FW11 -- Durağan varyans doğrulamasının zaman içinde kararlılığı

**Türkçe rapor.** Bütün hesap ve kod İngilizce; yalnızca bu belge
Türkçe. Kurallar: yeni parametre kestirimi yok (sadece OLS AR(1) ve
betimsel istatistik), look-ahead yok (her tarih için yalnızca o
tarihe kadarki veri kullanıldı), sentetik veri yok, üretim yaml ve
`model_limitations.md` ve `PROJECT_STATUS_AND_FUTURE_WORK.md`
değiştirilmedi, kabul edilmiş çıktı ağaçlarına yazılmadı.

## Amaç

FW9'un doğrulaması şuydu: üretim parametrelerinin ima ettiği artık
sapması sd_prod(2917.78) = 582.5 TRY/MWh, 2025'in gerçekleşen artık
sapmasıyla (saatlik şekil tanımına göre 531 veya 624) örtüşür.
FW10b ise 2026'da bu sapmayı 2-3 kat eksik verdiğini buldu.

Bu iş paketi iki soruyu cevaplar:

1. 2025'teki örtüşme tek bir tarihin tesadüfü mü, yoksa 7 tarihte
   kararlı bir örüntü mü?
2. 2026'daki sapma fiyat seviyesinin (L) düşmesinden dolayı
   sd_prod(L)'nin küçülmesinden mi, yoksa gerçek bir normalize
   oynaklık artışından mı geliyor?

## Yöntem

Sekiz değerleme tarihi: FW4'ün yedi tarihi (2022-12-31, 2023-06-30,
2023-12-31, 2024-06-30, 2024-12-31, 2025-06-30, 2025-12-31) artı
2026-09-27. Her tarihte son 12 ay kayan pencere. Ek olarak yalnız
2026-01-01 ile 2026-09-27 arasını kapsayan bir 9. satır.

Her satırda **model-faithful A3 TRY artığı**:
`residual = P - takvim_ayı_ortalaması - saat_of_hafta_şekli`.
İki şekil varyantı:

- **pooled**: HOW şekli o tarihe kadarki BÜTÜN geçmişten (2019'dan
  başlar) tahmin edilir.
- **12m_only**: HOW şekli yalnızca son 12 ayın penceresinden.

Ay ortalaması her iki varyantta da hücrenin kendi takvim ayının
ortalamasıdır (pencere-invariant).

Diğer metrikler:
- Level L: son 12 ayın ortalama PTF'si (TRY/MWh).
- Üretim sd formülü: `sd_prod(L) = 0.1987 * sqrt(L^2 + 282.48^2)`.
  Buradaki `0.1987`, `stationary_variance_check.md`'de raporlanmış
  ve tekrar hesaplanmayan üretim durağan asinh sapmasıdır;
  `282.48` üretim yaml'ının `scale_P`'sidir.
- Oran = gözlenen sd / sd_prod.
- Normalize sapma = gözlenen sd / L. Üretim için bu oran L büyükken
  yaklaşık 0.1987.
- OLS AR(1) kappa (rezidualın kendi AR(1) katsayısı, MLE değil).
- 50 TRY/MWh altındaki saatlerin payı ve o saatler çıkarıldığında sd.

## Doğrulama (2025-12-31 satırı)

FW9 stationary_variance_check.md 2025 için gözlenen sd_TRY = 624 ve
kappa ~0.20 raporlamıştı; FW10 ile eklenen `tail_validation.md` ise
2025-only şekliyle 531 raporlamıştı.

FW11 2025-12-31 satırı:

| kaynak | değer |
|---|---:|
| L (12 ay ortalaması, 2025) | 2 619.74 |
| sd_pooled (havuzlanmış şekil) | **623.96** |
| sd_12m_only (yalnız 2025 şekil) | **531.08** |
| sd_prod at L=2619.74 | 523.65 |
| kappa_per_hour (pooled) | **0.2013** |

Üç metrik de FW9/FW10 rakamlarını virgülden sonra iki basamağa kadar
yeniden üretiyor. Doğrulama başarılı.

## Ana bulgular

Tam tablo `variance_stability.md` içinde. Kompakt özet (pooled şekil):

| tarih | L | sd | sd_prod | oran | norm_sd/L | kappa | %<50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022-12-31 | 2 511 | 691 | 502 | **1.38** | 0.275 | 0.255 | 0.7% |
| 2023-06-30 | 2 796 | 719 | 559 | 1.29 | 0.257 | 0.263 | 0.5% |
| 2023-12-31 | 2 189 | 453 | 439 | 1.03 | 0.207 | 0.242 | 0.4% |
| 2024-06-30 | 2 053 | 473 | 412 | 1.15 | 0.230 | 0.175 | 0.8% |
| 2024-12-31 | 2 235 | 487 | 448 | 1.09 | 0.218 | 0.188 | 0.7% |
| 2025-06-30 | 2 424 | 560 | 485 | 1.15 | 0.231 | 0.193 | 0.8% |
| 2025-12-31 | 2 620 | 624 | 524 | 1.19 | 0.238 | 0.201 | 1.0% |
| 2026-09-27 | 2 189 | 817 | 439 | **1.86** | **0.373** | 0.198 | **5.0%** |
| 2026-01-01 → 09-27 | 1 963 | 893 | 394 | **2.26** | **0.455** | 0.186 | **6.7%** |

### Soru 1: Örtüşme kararlı mı?

2023-12 ile 2025-12 arasındaki BEŞ tarihte (2023-06 dahil edilirse
altısında) oran 1.03-1.29 aralığında; ortalama 1.12, standart sapma
0.08. Bu, **yaklaşık %15 bandı içinde** kararlı bir örtüşmedir --
yani 2025'teki 1.19 tesadüf değil, dört yıl boyunca istikrarlı bir
üretim-gözlem uyumu. 2022-12 satırı 1.38 ile bandın ucunda çıkıyor,
2022'nin (Rus-Ukrayna sonrası enerji krizi yılı) OTM'de anomali
olduğunu doğruluyor; normalize sd/L o tarihte 0.275 (ortalama 2023-25
oranına 0.22, üretimin 0.199 hedefine karşı) -- bu, gözlenen
ortalama L yükselmişken oynaklığın da orantısal artması demek.

12m-only şekiliyle (aynı beş tarihte) oran 0.93-1.13 aralığında,
ortalama 0.98. **12m-only şekilde 2023-12'den 2025-12'ye kadar oran
1.00'e daha yakın**. Bu FW9'un ana bulgusunu güçlendiriyor: üretim
sd_prod formülü, gözlenen dispersion'un CENTRE değerini iki farklı
şekil tanımlaması altında bir ucundan (531/524 = 1.01) yakalıyor.

### Soru 2: 2026 sapması nereden geliyor?

Seviye (L) etkisi vs normalize oynaklık:

- Historical 2022-2025 pencerelerde L ortalaması 2 318 TRY/MWh
  (min 2 053, max 2 796). Normalize sd/L ortalama 0.235.
- **2026-09-27 penceresinde L = 2 189** -- historik ortalamayla
  karşılaştırılabilir seviyede. Yani sd_prod(L)'nin küçülmesi
  fiyat seviyesi düşüşünden değil, L'nin 2022-2025 bandında
  kalmasından geliyor.
- **Normalize sd/L 2026-09-27'de 0.373**, tarihsel ortalamanın
  neredeyse iki katı (0.235 vs 0.373 → 1.59x).
- **2026-only pencerede L = 1 963** ve normalize sd/L = **0.455**,
  tarihsel ortalamanın 1.94x'i.

**Sonuç: 2026'daki sapma büyük ölçüde seviyeden değil, gerçek bir
normalize oynaklık artışından geliyor.** L 2 189 iken sd_prod(L) =
439; gözlenen sd 817 -- fark 378 TRY (ratio 1.86 fazlalık) L'nin
düşmesinden değil sd/L'nin 0.20'den 0.37'ye çıkmasından. Üretim
`sd_asinh = 0.1987` sabitinin 2026 için doğrusu ~0.37 olurdu; bu
2x'lik bir asinh-scale genişleme demek.

### Düşük fiyatlı saatlerin payı

Sub-50 TRY hours share:
- 2022-2025: 0.4-1.0% (birer istisna)
- 2026-09-27: **5.0%**, 2026-only: **6.7%**

Bu saatlerin 2026'da neredeyse tamamı güneş oversupply saatleri (öğle
civarı sıfıra yakın PTF). Ancak bu saatler çıkarıldığında oran fazla
değişmiyor:

- 2026-09-27 pencere: oran 1.86 → 1.87 (sub-50 çıkarılınca daha da
  artıyor, çünkü sıfır civarı fiyatlar residual'ı ORTALAYA çekiyor)
- 2026-only: 2.26 → 2.29 (aynı yönde)

Yani düşük fiyatlı saatler **oynaklık artışının nedeni değil, arta
kalan yüksek fiyatlı saatlerin çok daha volatil olduğunu maskeleyen
bir sinyal**. 2026'da yenilenebilir arz aynı gün içinde fiyatı hem
çok yükseltiyor hem çok düşürüyor; residual dispersion bu keskin
gün-içi zıtlıktan geliyor.

### 2022 kriz yılı yorumu

2022-12-31 penceresi (2022 boyunca) oran 1.38 -- 2023-25 bandının
üstünde. Sebep: 2022'nin Q1'inde toptan enerji fiyatları küresel
gaz krizi ile aylık %100'e varan sıçramalar yaptı; A3 residual (ay
ortalaması ve HOW şekli çıkarıldıktan sonra) bile bu şok'lardan
ETKİLENDİ. Yıl-içi çok yüksek gün-içi dispersion normalize sd/L'yi
0.275'e çıkardı (2023-25 ort 0.22). Bu, kriz yıllarında üretim
modelinin yaklaşık %30-40 eksik varyans verdiğinin bir dokümentasyonu
-- 2026'da benzer bir mekanizma (fiyat çöküşü ve keskin yenilenebilir
oversupply salınımları) çalışıyor olabilir.

### FW9'un yapısal sınırıyla bağlantı

FW9 kappa_sensitivity_isovariance_v2.md'de not ettiğimiz **yapısal
sınır**: sabit `scale_P = 282.48` altında asinh + delta eşlemesi
sd'yi fiyat seviyesiyle ORANTILI varsayar. Yani üretimin ima ettiği
`sd/L` L büyükken 0.1987'ye yakınsıyor -- L'ye bakılmaksızın.

Veri bu varsayımı **2023-2025 için destekliyor**:
gözlenen sd/L 0.207-0.238 aralığında (0.199 hedefinin çevresinde
±%20). 2022 kriz yılında bu 0.275'e çıkıyor (%38 sapma). **2026'da
tamamen kırılıyor**: sd/L 0.373-0.455, üretim varsayımının 1.9-2.3
katı.

Bu, sabit asinh yapının kısa vadede yetersiz kalabildiğini gösteriyor.
FW11 hiçbir modelin yeniden kestirimini yapmıyor; bu, gelecek
paketlerin (FW12+) konusu.

## Çıktılar

Hepsi `outputs/fw11_variance_stability/` altında:

- `variance_stability.csv`: 9 satır (8 tarih + 2026-only), tam metrik
  seti. Her tarih için pooled ve 12m_only varyantları için sd, kappa,
  q05, q95, ratio, normalize sd/L, sub-50 sub-analiz.
- `variance_stability.md`: kompakt tablo + tam detay.
- `figure_source.csv`: makale figürü için hazır. DejaVu Serif, 9.5
  pt, W = 5.5 inç, üst ve sağ çerçeve kapalı stil notu header'da.
  Sütunlar: date_label, L, sd_pooled, sd_12m_only, sd_prod, oranlar,
  normalize sd/L, sub-50 payı ve çıkarılınca sd, kappa.
- `hashes_before.txt` / `hashes_after.txt`: 9 donmuş invariantın
  bit karşılaştırması (aşağıda).

## Testler ve pytest

`tests/test_fw11_variance_stability.py` -- 3 test:

1. **2025-12-31 FW9 değerlerini yeniden üretir**: pooled sd =
   624 ± 1, 12m_only sd = 531 ± 1, kappa = 0.20 ± 0.01.
2. **Look-ahead yok**: valuation_utc'den sonraki hiçbir PTF gözlemi
   hesaba girmez; script 2027 tarihiyle çağrıldığında da mevcut
   veriyle sınırlı kalır ve bir hata / boş sonuç üretmez.
3. **sd_prod formülü doğru**: bilinen L değerinde formülün beklenen
   sonucu vermesi (L=2917.78'de sd_prod = 582.48 ± 0.1).

Tam pytest ve hash karşılaştırması aşağıda.

## Tam pytest ve hash sonuçları

- **Tam pytest** (yavaş testler dahil): **981 passed in 778.45s**
  (~13 dakika). Öncekine göre +3 yeni test (FW11). Skip yok, hata
  yok. Breakdown: 978 (FW10b sonrası) + 3 FW11 = 981.
- **9 dosyalık hash karşılaştırması**
  (`hashes_before.txt` vs `hashes_after.txt`): **birebir aynı**, 0
  dosya değişikliği. Bu manifest üretim yaml, `model_limitations.md`,
  `docs/PROJECT_STATUS_AND_FUTURE_WORK.md` ve diğer donmuş
  invariantları kapsıyor.
- Kabul edilmiş dört çıktı ağacına (outputs/market_calibration_final,
  outputs/forward_centered_diagnostics, outputs/scenario_sweep,
  outputs/tvtp2_experimental) yazılmadı. inputs/ altında hiçbir
  dosya değişmedi.

## Commit mesajı

Tek commit mesajı `Claude outputs/fw11_commit_msg.txt` dosyasında;
BOM'suz UTF-8, tamamen İngilizce, yalnızca ASCII, AI/asistan atfı
yok. Commit atılmadı.

