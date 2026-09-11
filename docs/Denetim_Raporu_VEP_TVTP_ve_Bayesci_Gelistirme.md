# VEP-Forward TVTP Opsiyon Fiyatlama — Veri Denetimi ve Bayesçi Geliştirme Değerlendirmesi

**Kapsam:** `vep-forward-tvtp-option-pricing-main` deposu (Faz 1–2 "tamamlandı" durumu), `inputs/market/realized_ptf_2026.csv` gerçekleşen PTF verisi ve Gupta & Reisinger (2012), *Robust Calibration of Financial Models Using Bayesian Estimators*.
**Tarih:** 11 Eylül 2026
**Yöntem:** Kod ve çıktılar okundu. `run_pde.py validate` (15/15 PASS) ve `run_pde.py price` yeniden koşturuldu. Backtest metrikleri ham veriden bağımsız olarak yeniden hesaplandı. Ek testler yapıldı: kapsama (coverage), CRPS, tavan/taban fiyat, gün içi profil, AR(1) ve opsiyon üst sınır testleri.
**Çalıştırılamayan:** pytest kurulamadı (paket deposuna erişim yok). Bu yüzden "169/172 test geçiyor" iddiasını bağımsız doğrulayamadım. Depoda tarihsel PTF serisi yok (sadece standartlaştırılmış RD `z` var), bu yüzden M9 kestirimini yeniden üretemedim.

---

## 1. Özet hüküm

| Katman | Durum | Açıklama |
|---|---|---|
| Forward eğrisi kalibrasyonu (VEP) | **Doğru** | 6 VEP kotasyonu 1e-12 TRY/MWh hassasiyetle yeniden üretiliyor. Bağımsız yeniden hesaplamada da doğrulandı. |
| Sayısal çözücü (PDE / moment ODE / MC) | **Doğru** | Put-call paritesi 1e-6'da tutuyor, E[P_t]=F(t) tam. PDE 687.04 ile MC 694.89 ± 4.30 uyumlu (|z|=1.82, sınırda). |
| Backtest raporundaki sayılar | **Doğru** | MAE 1312.39, RMSE 1575.43, bias −1019.04 TRY/MWh ve aylık tablo birebir tuttu. |
| **Rezidü dinamiği (σ, κ)** | **Hatalı, kritik** | Model belirsizliği 5–13 kat şişirilmiş. 50%'lik tahmin bandı gerçekleşen saatlerin %99.7'sini kapsıyor. |
| **Fiyat tavanı / tabanı** | **Modelde yok, kritik** | Değerleme tarihinde PTF tavanı 3400 TRY/MWh idi. Grid'deki 66 call'dan **54'ü** tavanın izin verdiği maksimum ödemeyi aşıyor. Model saatlerin ortalama **%37**'sinde negatif fiyat olasılığı veriyor, oysa PTF ≥ 0. |
| **Saatlik profil (HPFC)** | **Eksik, yüksek** | Rezidü varyansının ~%40'ı ay×saat deterministik profili (ay-seviyesi sapma hariç). Model bunu stokastik gürültü sayıyor. |
| Parametre provenansı (USD/TRY) | **Tutarsız, yüksek** | M9, USD bazlı koşudan (`markov_usd_final`) geliyor. Kendi adaptörünüz bu koşuyu "yalnız doğrulama için" diye işaretlemişken fiyatlamada kullanılıyor. |
| Risk-nötr Q1 kanalı | **İşlevsiz** | a₁ = −0.5 TRY/MWh/h bile fiyatı %0.0004 değiştiriyor. Risk primi fiilen sıfır. |
| Dokümantasyon | Küçük tutarsızlık | README/audit'te 72 saatlik benchmark 677.23 yazıyor, güncel kod 687.04 üretiyor. |

**Sonuç:** Mühendislik altyapısı sağlam, ama **ürettiği opsiyon fiyatları ekonomik olarak kullanılamaz**. Hataların çoğu kodda değil, modelin girdilerinde ve spesifikasyonunda. Paper'daki Bayesçi çerçeve projeye **anlamlı katkı sağlar**, ancak ancak bu yapısal hatalar düzeltildikten sonra. Aşağıda gerekçesi var.

---

## 2. Bulgular (verilerle ölçülmüş)

### B1. Tavan fiyat ihlali: no-arbitrage üst sınırı aşılıyor (KRİTİK, yeni bulgu)

Gerçekleşen veride PTF tavanı açıkça görülüyor:

| Dönem | Maks. PTF | Tavanda geçen saat | 0 TL'de geçen saat |
|---|---:|---:|---:|
| 2025-12 | 3400 | 8 | 0 |
| 2026-01 | 3400 | **225** | 0 |
| 2026-02 | 3400 | 57 | 20 |
| 2026-03 | 3400 | 39 | 22 |
| 2026-04 | **4500** | 19 | 95 |
| 2026-05 | 4500 | 3 | **214** |
| 2026-06 | 4500 | 2 | 40 |
| 2026-07 | 4500 | 13 | 0 |

Tavan Nisan 2026'da 3400'den 4500 TRY/MWh'ye çıkmış; EPDK kararıyla uyumlu ([yesilhaber](https://yesilhaber.net/epdk-elektrik-azami-fiyat-4500-tl-mwh/)). Taban 0.

Grid'deki bütün vadeler (24–720 saat) Ocak 2026'ya düşüyor, yani tavan 3400. O zaman call ödemesi en fazla (3400−K)⁺ olabilir ve fiyat ≤ DF·(3400−K)⁺ olmalı. 72 saatlik örnek:

| K | Model call | Üst sınır DF·(3400−K)⁺ | Ex-post Ocak ort. ödeme E[(P−K)⁺] |
|---:|---:|---:|---:|
| 2000 | 1274.6 | 1395.4 | 938.1 |
| 2600 | 896.5 | 797.4 ❌ | 428.6 |
| 3000 | **687.0** | **398.7** ❌ | **167.1** |
| 3400 | 512.4 | **0** ❌ | 0 |
| 4000 | 312.8 | **0** ❌ | 0 |

Sonuç: grid'deki **66 call'dan 54'ü imkânsız fiyat** veriyor. Ana benchmark olan K=3000, 72 saatlik call (687 TRY/MWh), tavanın izin verdiği maksimum ödemenin 1.7 katı. Tavan üstündeki strike'lar (K ≥ 3400) sıfır olmalıyken 313–2023 TRY/MWh arası değer alıyor. Put tarafı da bozuk: model negatif fiyatlara olasılık kütlesi koyuyor. MC'de 72 saatte P(P_T<0) = %5.5; Gauss yaklaşımıyla backtest ufkunda ortalama %37.

### B2. Rezidü volatilitesi aşırı şişirilmiş (KRİTİK, ekip kısmen biliyor)

Bağımsız hesap (5088 saat, 2026-01 → 2026-07):

| Test | Beklenen (iyi kalibre model) | Gerçekleşen |
|---|---:|---:|
| %50 tahmin bandı kapsaması | %50 | **%99.7** |
| %90 bandı | %90 | %99.9 |
| %95 bandı | %95 | %100 |
| Model sd / RMSE (aylık, *bias dahil*) | ≈1 | **5.3 – 10.0** |
| Ortalama CRPS | — | 2401 TRY/MWh |

Rapordaki "7–13×" oranı, ay-içi std'yi kullanıyor, yani ay-seviyesi sapmayı dışarıda bırakıyor. Adil karşılaştırma RMSE ile yapılmalı. O zaman oran 5–10× çıkıyor; yine de kabul edilemez.

Kök neden ekibin `half_life_reconciliation.md` teşhisiyle uyumlu, ama benim ölçümüm daha keskin:

| Seri | lag-1 AR | Yarı ömür |
|---|---:|---:|
| Modelin kullandığı κ (ham asinh-PTF, M9 φ) | 0.999996 | **168 720 saat (~19 yıl)** |
| Gerçekleşen P−F (2026) | 0.882 | **5.5 saat** |
| Ay×saat profili çıkarılmış P−F | 0.841 | **4.0 saat** |
| Günlük ortalama P−F | 0.782 | **2.8 gün** |

ACF(24 saat) = 0.37 ve ACF(168 saat) = 0.11. Yani rezidüde **iki zaman ölçeği** var: saatler mertebesinde hızlı bir bileşen ve gün/hafta mertebesinde yavaş bir bileşen. Tek faktörlü, rastgele yürüyüş gibi davranan bir OU bu yapıyı temsil edemez. Ekibin FW1 önerisi (κ ≈ 0.078/saat) doğru yönde ama tek başına yetmez: yavaş bileşen ve saatlik profil de gerekli.

Mekanizma: M9'un σ_stress = 0.0924/√saat değeri, ham saatlik asinh fiyat değişimlerinden kestirilmiş. Bu değişimlerin büyük kısmı gün içi deterministik salınım (gece–gündüz, güneş). Bu salınımı rastgele yürüyüş şoku gibi biriktirmek, 72 saatte sd = 1838 TRY/MWh üretiyor. Gerçekleşen ilk haftada fiyatın kendi std'si yalnızca 655 TRY/MWh.

### B3. Saatlik fiyat profili (HPFC) eksik (YÜKSEK)

Forward eğrisi ay içinde pürüzsüz ve gün içi şekli yok. P−F varyansının ayrışımı:

- %22: ay-seviyesi sapma (VEP'in kendi tahmin hatası, modele dışsal)
- **%40: ay×saat deterministik profil** (öğlen güneş çukuru, akşam pik)
- %38: gerçek stokastik kısım (sd ≈ 740 TRY/MWh)

Saatlik opsiyon (expiry-hour spot) fiyatlayan bir model için bu kabul edilemez. Saat 13:00'te vadesi dolan bir call ile 20:00'de dolan bir call bugün aynı fiyatı alıyor.

### B4. USD/TRY parametre karışımı (YÜKSEK, yeni bulgu)

- Bundle'daki M9 kestirimi `...\outputs\markov_usd_final\` klasöründen geliyor.
- M9'un doğrulama metrikleri `val_mae_price = 7.12` ve `val_mae_y = 0.090`. Oran 7.12/0.090 ≈ 79 ediyor, bu da ~70–80 birimlik fiyat seviyesi demek. Bu seviye **USD/MWh** ile tutarlı; TRY'de aynı oran ~2500–2900 olurdu.
- `markov_adapter.py` dokümantasyonu açıkça şunu söylüyor: "All PRICING dynamics are rebuilt from the authoritative TRY/M2 parameters — the shipped [USD M9] series is used only for validation". F2.1'de fiyatlama M9'a geçirilmiş, yani bu tasarım kararı sessizce tersine çevrilmiş.
- Sonuç: σ_y, USD fiyat üzerindeki asinh dönüşümünden geliyor ama TRY `scale_P = 282.48` ile TRY'ye çevriliyor. Fiyat ≫ scale olduğu sürece bu yaklaşık bir "göreli vol" kabulüdür ve etkisi sınırlıdır. Yine de birim tutarlılığı doğrulanamıyor. Ayrıca `z` standartlaştırması TRY eğitim penceresinde (≤2022-12-31, 61 361 satır) yeniden kurulmuş; M9 ise ~78 905 saatlik (≈2024 sonuna kadar) USD penceresinde kestirilmiş. Yani γ katsayıları farklı bir standartlaştırmaya uygulanıyor.

### B5. M9 kendi doğrulamasını geçemiyor (ORTA)

Aynı bundle'daki diagnostikler: `val_pit_ks_p = 5.9e-17`, Ljung-Box p = 0, ARCH-LM p = 0, genelleştirilmiş rezidü std = 10.7 (1 olmalı). Model seçimi (BIC) M9'u öne çıkarıyor ama M9'un öngörü dağılımı istatistiksel olarak reddediliyor. "Stress" rejiminin durağan olasılığı %67. Bu, rejimin bir piyasa stres hali değil, gün içi volatil saatleri temsil ettiğini gösteriyor. Etiket yanıltıcı.

### B6. Risk-nötr Q1 kanalı işlevsiz (ORTA, yeni yorum)

`risk_premium_sensitivity.csv` sonuçları: a = ±0.004 → fiyat değişimi ±0.0003%; a₁ = −0.5 → −0.0004%. Forward-centering ortalamayı sıfırladığı için sürüklenme (drift) kaydırması yalnız O(a²) varyans etkisi bırakıyor. Yani "risk primi düğmesi" pratikte hiçbir şey yapmıyor. Anlamlı bir fiyatlama ölçüsü değişikliği volatilite/rejim yoğunluğu kanalından (Q2: η_ij, σ-primi) gelmek zorunda.

### B7. Diğer

- α01 ve α10 kestirilmemiş, türetilmiş; standart hataları yok. RD_Ramp kovaryatı atılmış; AME'si RD_lag1'in ~5 katı. `pi_filtered` M2'den geliyor. Bunlar ekipte zaten belgelenmiş.
- 72 saatlik benchmark README'de, PROJECT_STATUS'ta ve calibration_audit §6'da 677.23 olarak geçiyor; güncel kod 687.04 üretiyor. Dokümanlar φ düzeltmesi öncesinden kalmış.
- Ocak "başarısı" (sMAPE %16.8) kısmen tavandan geliyor: Ocak'ta 225 saat fiyat 3400'e yapışmış ve bu dağılımı yukarıdan kesmiş.
- Yorumsal bir düzeltme: VEP'in 2026 ilkbaharındaki −%28 ile −%76 arası sapması **modelin hatası değil**. Ekibin bu tespiti doğru. Ama bu sapma bir risk ölçüsüne dönüştürülmeli (bkz. §3, P3).

---

## 3. Paper'ın yöntemleri projeyi geliştirebilir mi?

**Kısa cevap: evet, ama paper'daki haliyle değil, uyarlanarak. Ve yapısal düzeltmelerden (B1–B4) sonra.**

### 3.1 Doğrudan aktarılamayan kısım

Paper'da model parametresi (yerel volatilite yüzeyi σ(S,t)) likit **vanilla opsiyon kotasyonlarının bid-ask aralığına** kalibre ediliyor (denklem 6–9). Türkiye'de likit bir elektrik opsiyon piyasası yok. Proje dokümanları da bunu söylüyor: volatilite ve rejim primleri "not identified". Dolayısıyla paper'ın ana iddiası olan "vanilla fiyatlardan egzotiklere sağlam fiyat" bizde birebir kurulamaz. Volatilite tarafında piyasa likelihood'u yok.

### 3.2 Aktarılabilen ve değer katan kısımlar

| Paper bileşeni | Projedeki karşılığı | Kazanım |
|---|---|---|
| **Tikhonov = MAP denkliği (denk. 10–11)** | Mevcut `smooth_constrained` forward eğrisi (smoothness_weight, level_weight) tam olarak bir Tikhonov/MAP çözümü. Tüm σ, κ, α "point estimate". | Mevcut yaklaşımı Bayesçi çerçeveye oturtur. MAP bir özel durum olarak kalır, dolayısıyla geriye uyumlu. |
| **Bid-ask toleranslı likelihood (denk. 6–9)** | VEP kotasyonları şu anda KKT'de *kesin eşitlik*. Bunun yerine VEP min/maks/AOF aralığından veya VİOP elektrik vadelilerinin bid-ask'ından δᵢ toleransı alınıp truncated-Gauss likelihood kurulur. Ocak (kotasız ay) doğal olarak geniş belirsizlik alır. | Forward eğrisinin *dağılımı* elde edilir; saatlik şekil ve Ocak çapası belirsizliği fiyata yansır. |
| **Fonksiyonel parametre + log-spline + Sobolev prior (denk. 4–5, §4.1)** | σ(saat-of-day, ay) veya σ(t) "yerel vol yüzeyi" analoğu. Saatlik profil (HPFC) katsayıları da aynı şekilde. Pozitiflik log ile, pürüzsüzlük H¹ normu ile sağlanır. | B2–B3'ü doğrudan adresler: vol ve profil saat/aya göre değişir ama düzenlileştirilmiş kalır. |
| **Posterior örnekleme: MCMC Metropolis, prior-kovaryanslı adım θ'=θ+√(2du)Bξ, %23 kabul, m paralel zincir, burn-in/thinning (§4.2)** | θ = (log σᵢ, log κ_hızlı, log κ_yavaş, αᵢⱼ, γᵢⱼ, profil katsayıları, spike parametreleri). Likelihood tarihsel saatlik (P−F) rezidülerinin Hamilton-filtre log-olabilirliği artı forward kotasyon terimi. | α'lara standart hata *yerine* tam posterior gelir. M8/M9 seçimi yerine model ortalaması yapılır. Türetilmiş parametre sorunu (B7) çözülür. |
| **PSRF / Gelman-Rubin yakınsama (denk. 13)** | Parametreler ve anahtar opsiyon fiyatları için PSRF < 1.1 kabul kriteri. | Mevcut `validate`/test çerçevesine doğal biçimde eklenir. |
| **Bayes fiyatı (posterior ortalama, denk. 15) vs MAP fiyatı, fiyat pdf'i, güven aralığı (§6.1)** | Her opsiyon için tek sayı yerine "Bayes fiyatı + %68/%95 model-belirsizliği bandı". | "VEP-anchored, volatility-assumed" etiketini sayısallaştırır. Makale için güçlü bir katkı cümlesi sağlar. |
| **Sağlamlık testleri: diskretizasyon, prior (κ-normu), gürültü, kalibrasyon sayısı (§6.2–6.4)** | Grid/knot sayısı, smoothness_weight, δ, eğitim penceresi, rejim sayısı varyasyonları altında MAP ve Bayes fiyatlarının kararlılığı. | Hakemin "neden bu parametre?" sorusuna sistematik cevap. Paper'daki Şekil 7–9'un elektrik versiyonu. |
| **Yeniden kalibrasyon ve importance sampling ile posterior güncelleme (§5.3)** | Her hafta yeni VEP kotasyonları ve gerçekleşen PTF geldikçe örnekler yeniden ağırlıklandırılır. ESS düşünce resample-move (SMC) yapılır. | FW6'yı (backtest'i uzatma) bir *öğrenen* sisteme çevirir. |
| **Tanımlanamayan parametreler için prior-yayılımı** | η01, η10, σ-primi ve λᵢ için literatüre dayalı prior. Posterior ≈ prior olur ve bu fiyat aralığında görünür. Avellaneda/uncertain-vol ruhuna uygun bid/ask bandı çıkar. | B6'daki işlevsiz Q1 yerine *dürüst* bir risk-primi belirsizliği. |

### 3.3 Paper'ın kendi uyarıları (bizim için kritik)

1. **Yanlış model sınıfı düzelmez.** Paper'da model gerçeğe yakınsa sağlamlık elde ediliyor (§2, §6.2, Şekil 7'de gerçek değere yakınsamama). Tavan/taban içermeyen, gün içi profili olmayan, κ'sı yanlış bir modelin posterior'u da yanlış olur. Bu yüzden **önce B1–B4 düzeltilmeli**.
2. **Hesap maliyeti.** Paper'da 16 zincir × 25 000 iterasyon, her iterasyonda bir PDE çözümü var. Bizim PDE'miz 1D × 2 rejim ve ~5 saniye sürüyor (CLI dahil). Likelihood'u moment ODE ile ve Hamilton filtresiyle hesaplayıp PDE'yi yalnızca posterior örneklerinin fiyatlanmasında (N≈500–2000) kullanmak gerekiyor. Böylece hesap yapılabilir hale gelir.
3. **Egzotik analoğu.** Paper'da bariyer opsiyon, MAP'in en çok yanıldığı yer. Bizde doğal karşılıklar aylık baseload/peak **Asya tipi (ortalama) opsiyonlar**, **tavanla sınırlı call'lar** ve **swing/strip** opsiyonlar. MAP vs Bayes farkının en belirgin görüneceği yer burası.

---

## 4. Önerilen yol haritası (öncelik sırasıyla)

**Faz A – Yapısal düzeltmeler (zorunlu):**
- A1. Fiyat tavanı ve tabanı: ödeme P_T yerine min(max(P_T,0), cap(t)) olmalı, yani sınırlı dönüşüm veya yansıtan/emici sınır. Tavan takvimi: ≤ 2026-03: 3400, ≥ 2026-04: 4500. Arbitraj sınırı testleri eklenmeli.
- A2. HPFC: VEP aylık seviyesi × saat/gün-tipi/ay profili. Profil tarihsel PTF'den kestirilir. Aylık ortalama eşitliği korunur.
- A3. Rezidü dinamiğini P−F_HPFC üzerinde yeniden kestirmek: iki faktör (κ_hızlı ~0.1–0.2/saat, κ_yavaş ~0.01/saat), rejim-bağımlı σ, isteğe bağlı spike/sıfır-fiyat bileşeni.
- A4. Tek para birimi (TRY veya reel TRY) ile tek pencere. M9-USD parametreleri yalnız karşılaştırma için tutulur.
- A5. Dokümanlardaki 677.23 → 687.04 tutarsızlığı düzeltilir. Q1 kanalının etkisizliği belgelenir.

**Faz B – Paper'ın Bayesçi katmanı:** prior'lar, likelihood (tarihsel + forward bid-ask), MCMC, PSRF, Bayes vs MAP fiyatlar, sağlamlık testleri, importance-sampling ile yeniden kalibrasyon.

**Faz C – Değerlendirme:** yuvarlanan (rolling) out-of-sample backtest (her gün yeniden değerleme). Metrikler: kapsama, PIT, CRPS, pinball kaybı, opsiyon-ödeme backtesti. Benchmark'lar: Black-76, Lucia-Schwartz ve mevcut MAP modeli.

Bu yol haritasının ChatGPT'ye verilecek tam uygulama promptu ayrı dosyada: `ChatGPT_Proje_Promptu.md`.

---

### Kaynaklar
- Gupta, A. & Reisinger, C. (2012). *Robust Calibration of Financial Models Using Bayesian Estimators*.
- Depo çıktıları: `outputs/market_calibration_final/*`, `inputs/historical/archive/calibration_bundle/*`, `pde_option_model/markov_adapter.py`.
- Tavan 4500 TRY/MWh: [EPDK kararıyla elektrikte azami fiyat 4.500 TL/MWh oldu — Yeşil Haber](https://yesilhaber.net/epdk-elektrik-azami-fiyat-4500-tl-mwh/); [EPDK azami uzlaştırma fiyat mekanizması](https://www.epdk.gov.tr/Detay/Icerik/2-12999/azami-uzlastirma-fiyat-mekanizmasi-icin-uzatma-ka)
