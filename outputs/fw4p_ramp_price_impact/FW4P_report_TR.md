# FW4-P -- Ramp kovaryatının fiyat etkisi

**Türkçe rapor.** Bütün hesap ve kod İngilizce; yalnızca bu belge Türkçe.
Kurallar: yeni parametre kestirimi yok (FW9'un kestirdiği katsayılar olduğu
gibi alındı), look-ahead yok, sentetik veri yok, fiyatlama üretim
climatology kovaryat yoluyla yapıldı (FW12b kuralı), üretim yaml'ı ve
`model_limitations.md` ve `PROJECT_STATUS_AND_FUTURE_WORK.md`
değiştirilmedi, kabul edilmiş çıktı ağaçlarına yazılmadı.

## 1. Amaç

FW4'ün deneysel iki kovaryatlı modu, 72 saat `K = 3000` alım opsiyonunda
toplam ramp etkisini **-%1,008** olarak raporlamıştı. Ama o sayı *yeniden
kurulmuş* bir ramp serisi ve M9 paketinden *aktarılmış* eğimlerle
üretilmişti. FW9 daha sonra kendi ramp serisini tanımladı ve eğimleri
birlikte kestirdi (`outputs/fw9_self_estimation/TVTP_2cov.pkl`:
h01 = -0,079462, h10 = +0,381573; tek kovaryata karşı LR = 869,93).

FW4-P, aynı kontratları FW9'un ramp serisi ve FW9'un ramp katsayılarıyla
fiyatlar; diğer her şey üretimdeki gibi kalır.

## 2. İki ramp arasındaki gerçek fark

**Tanım aynı.** Her iki kurulumda da `t` saatine geçişi süren çift
`(z(t-1), z(t-1) - z(t-2))`. `CovariatePathBuilder` tam olarak bunu üretir,
FW9'un `z.diff().shift(1)`'i de aynı nesnedir. Fark standartlaştırma
penceresinde ve eğimlerde:

| | m_r | s_r | pencere sonu | h01 | h10 |
|---|---:|---:|---|---:|---:|
| FW4 (yaml) | 3,417142e-05 | 0,271450 | 2024-12-31 | -0,069395 | +0,405544 |
| FW9 (pkl) | 2,962489e-05 | 0,266400 | 2022-12-31 | -0,079462 | +0,381573 |

Aynı artış için FW9'un `r`'si **%1,90** daha büyük. Eğimler de farklı:
h01 mutlak değerce %14,5 daha büyük, h10 %5,9 daha küçük.

**Bir uyarı.** İki gelenek, kaydırmanın eğitim penceresi kesilmeden önce mi
sonra mı uygulandığı konusunda da ayrışıyor. Bu yüzden deponun
`fit_ramp_scaler`'ı FW9'un ölçekleyicisini aynı pencerede yeniden
üretemiyor ve `CovariatePathBuilder.verify_ramp_scaler` onu tasarım gereği
reddediyor. Kaynak olarak FW9'un kendi `scripts/fw9/build_ramp.py`'si
kullanıldı, tutarsızlık gizlenmek yerine manifest'e ve bir teste yazıldı
(`test_the_two_ramp_conventions_are_genuinely_different`).

## 3. Yöntem

**Beş konfigürasyon fiyatlandı:**

| id | açıklama | FW4 karşılığı |
|---|---|---|
| `production` | üretim 1D, depo yaml kesme terimleri | R0 |
| `base_1d_w9` | 1D, kesme terimleri W9'da yeniden türetilmiş | R1 |
| `fw4_ramp` | FW4 deneysel ramp (yaml olduğu gibi) | R3 |
| `fw9_ramp` | FW9 ramp + FW9 eğimleri, **üretim kesme terimleri** | -- |
| `fw9_ramp_matched` | FW9 ramp + FW9 eğimleri, kesme terimleri W9'da yeniden türetilmiş | R3'ün yapısal karşılığı |

**Neden iki FW9 varyantı?** Görevin lafzı "diğer bütün parametreler
üretimdeki gibi" diyor; bu `fw9_ramp`. Ama FW4'ün manşet sayısı R3 - R1'dir
ve orada **her iki bacak da** aynı örneklemde moment eşlemesiyle yeniden
türetilmiş kesme terimleri taşır, yani ortalama geçiş olasılıkları sabit
tutulur. Bu önemli: FW4'ün kendi ayrıştırmasında 72 saat `K = 3000`'de
-%2,35'lik eğim payı, +%1,38'lik kesme terimi telafisiyle büyük ölçüde
sönüyor. Kesme terimlerini üretimde bırakmak, ramp kanalını bir doluluk
kaymasıyla karıştırır: `fw9_ramp`'te vade sonu stres olasılığı 0,6008'den
0,6364'e çıkıyor. Bu yüzden `fw9_ramp_matched` da fiyatlandı ve
karşılaştırmalı okuma onun üzerinden yapılmalı.

**Izgara.** Bir vadedeki bütün varyantlar **tek bir sabit** 1201 düğümlü
artık ızgarasında fiyatlandı (varyantların otomatik ızgaralarının birleşimi).
Aksi halde ~0,03 TRY/MWh'lik ayrıklaştırma farkı, ~1,7 TRY/MWh'lik ramp
etkisini kirletirdi. Bu FW4'ün kendi yöntemi; FW4'ün yayımladığı ızgara
sınırlarında R0 ve R3 fiyatları 1e-6 içinde geri geliyor (slow test).

**Kovaryat yolu.** Üretim climatology (FW12b): z yolu eğitim penceresinin
climatology'si, ramp onun kendi saatlik artışı; sabit-geçiş (z = 0) sınırına
düşülmedi, sentetik kovaryat üretilmedi. Bütün 2D koşularda `max_s < 1`,
yani sürekli zamanlı zincir her yerde var.

## 4. Üç fiyat yan yana (TRY/MWh)

ATM satırları `K = F(T)`.

| vade | K | üretim | FW4 ramp | FW9 ramp | FW4 % | FW9 % |
|---:|---:|---:|---:|---:|---:|---:|
| 24 h | ATM 2917,24 | 202,295 | 198,845 | 198,231 | -1,705 | -2,009 |
| 24 h | 2000 | 927,108 | 926,712 | 926,484 | -0,043 | -0,067 |
| 24 h | 2500 | 479,533 | 477,675 | 477,104 | -0,387 | -0,506 |
| 24 h | 3000 | 163,935 | 160,586 | 159,966 | -2,043 | -2,421 |
| 24 h | 3500 | 36,635 | 35,458 | 34,992 | -3,212 | -4,484 |
| 24 h | 4000 | 5,589 | 5,373 | 5,232 | -3,866 | -6,395 |
| 48 h | ATM 2916,70 | 205,775 | 202,757 | 202,045 | -1,467 | -1,813 |
| 48 h | 2000 | 926,223 | 925,865 | 925,616 | -0,039 | -0,066 |
| 48 h | 2500 | 480,839 | 479,159 | 478,539 | -0,349 | -0,478 |
| 48 h | 3000 | 167,125 | 164,187 | 163,476 | -1,758 | -2,184 |
| 48 h | 3500 | 38,108 | 37,043 | 36,540 | -2,796 | -4,113 |
| 48 h | 4000 | 5,953 | 5,759 | 5,604 | -3,270 | -5,877 |
| 72 h | ATM 2916,16 | 205,597 | 202,588 | 201,875 | -1,463 | -1,810 |
| 72 h | 2000 | 924,699 | 924,341 | 924,092 | -0,039 | -0,066 |
| 72 h | 2500 | 479,907 | 478,229 | 477,609 | -0,350 | -0,479 |
| 72 h | 3000 | 166,757 | 163,829 | 163,116 | -1,756 | -2,183 |
| 72 h | 3500 | 38,014 | 36,953 | 36,451 | -2,792 | -4,112 |
| 72 h | 4000 | 5,937 | 5,743 | 5,588 | -3,265 | -5,876 |

`F(T)` bütün varyantlarda aynı ve `E^Q[P_T] = F(T)` her koşuda makine
hassasiyetinde sağlanıyor: ramp yalnızca geçiş yasasını değiştiriyor,
merkezlemeyi değil.

## 5. Doluluk sabit tutulduğunda ramp etkisi

Bu, FW4'ün manşet tanımıyla (R3 - R1) birebir aynı kontrast:

| vade | K | FW4 ramp | FW9 ramp (eşlenmiş) |
|---:|---:|---:|---:|
| 24 h | ATM | -%1,067 | -%1,083 |
| 24 h | 3000 | -%1,277 | -%1,297 |
| 48 h | ATM | -%0,846 | -%0,872 |
| 48 h | 3000 | -%1,012 | -%1,044 |
| 72 h | ATM | -%0,843 | -%0,869 |
| 72 h | 3000 | -%1,010 | -%1,042 |

**Sonuç: FW9'un kendi rampı ve kendi eğimleri, FW4'ün yeniden kurulmuş
rampıyla neredeyse aynı sayıyı veriyor.** 72 saat `K = 3000`'de -%1,010'a
karşı -%1,042; aradaki fark 0,03 puan. FW4'ün yayımladığı -%1,008 de
buradaki -%1,010 ile örtüşüyor (kalan 0,002 puan, ızgara birleşiminin beş
varyant üzerinden alınmasından geliyor).

Etki para-dışına doğru yüzde olarak büyüyor (`K = 4000`'de -%1,5 ile -%2,3),
ama orada opsiyonun kendisi 5-6 TRY/MWh olduğu için mutlak fark 0,1
TRY/MWh'nin altında. Para-içinde (`K = 2000`) etki %0,02'nin altına iniyor.

Kesme terimleri üretimde bırakılırsa (`fw9_ramp`) fark 72 saat `K = 3000`'de
-%2,18'e çıkıyor, ama bu sayı ramp kanalını doluluk kaymasıyla karıştırdığı
için ramp etkisi olarak okunmamalı; `fw4p_ramp_effect.csv` içinde
`occupancy_controlled = False` ile işaretli.

## 6. Çekinceler

**(a) FW9'un iki kovaryatlı fit'i birim kök sınırında.** Ham asinh seviyesi
üzerinde kestirilmiş ve uyan kalıcılık `phi = 0,9999987156`, yani yarı ömür
yaklaşık 62 yıl. Bu sınırda seviye, rejim sürecinin taşıyacağı düşük
frekanslı yapıyı da soğuruyor; dolayısıyla geçiş katsayıları -- h01 ve h10
dahil -- **yanlı olabilir**. Ayrıca optimizasyon gradyan toleransını
tutturmamış (`converged = False`, gradyan normu 96,6; n = 61 368), bu yüzden
bu eğimler kullanılabilir bir standart hata taşımıyor.

**(b) Eğimler geldikleri fit'in dışında kullanılıyor.** FW9, h01/h10'u kendi
gammalarıyla (-0,423428, +0,116406) birlikte kestirdi; burada görevin
belirttiği gibi üretim gammalarıyla (-0,583778, +0,077698) birleştiriliyor.
Tümüyle tutarlı bir koşu z kanalını da oynatırdı.

**(c) Beklenen büyüklük.** FW9'un kendi ayrıştırması
(`price_impact_v2_decomposition.csv`) 72 saat `K = 3000`'de toplam fiyat
farkının **%0,387**'sini geçiş katsayılarına atfediyor. Yani küçük bir etki
zaten beklenen sonuçtu; yukarıdaki sayılar bununla tutarlı.

**(d) Ramp deneysel kalmaya devam ediyor.** Burada hiçbir kabul edilmiş
sonuç değiştirilmiyor; kabul edilmiş çıktı ağaçlarına dokunulmadı.

## 7. Dosyalar ve testler

`outputs/fw4p_ramp_price_impact/` altında:

| dosya | içerik |
|---|---|
| `fw4p_runs.csv` | 90 koşunun tamamı (5 varyant x 3 vade x 6 kullanım fiyatı) |
| `fw4p_three_way.csv` | üç fiyat yan yana, fark ve yüzde sütunlarıyla |
| `fw4p_ramp_effect.csv` | doluluk kontrollü ve kontrolsüz kontrastlar |
| `fw4p_ramp_price_impact.md` | İngilizce sonuç notu |
| `run_manifest.json` | girdiler, ayarlar, FW9 fit kaydı, ölçekleyiciler, kesme terimleri |
| `hash_manifest_before.txt`, `hash_manifest_after.txt` | korunan 128 dosyanın SHA-256 özetleri |

## 8. Test çalıştırması hakkında bir not

Tam pytest (yavaş testler dahil): **1038 test toplandı, 1033 geçti, 5
düştü**. FW4-P'nin 28 testinin tamamı geçiyor.

Düşen 5 test `tests/test_fw2_data_loaders.py` içinde ve hepsi aynı sebeple:
pandas, TÜİK TÜFE `.xlsx` dosyasını okurken `openpyxl` için `ImportError`
atıyor. `openpyxl` ne `requirements.txt`'te ne `requirements-dev.txt`'te var
ve kurulu değil. Ortam eksiği; bu iş paketiyle ilgisi yok (FW4-P yalnızca
yeni dosya ekliyor).

`outputs/fw9_self_estimation/deseasonalized_ar1.json` bu koşuda **yeniden
yazılmadı**: tam pytest sonrası `git status` dosyada değişiklik göstermedi,
dolayısıyla `git checkout --` ile geri alma gerekmedi. Aynı durum bu
ortamdaki önceki iki tam koşuda da geçerliydi. Dosya kabul edilmiş dört
çıktı ağacının dışında olduğu için 128 dosyalık manifest zaten onu
kapsamıyor.

Testler: `tests/test_fw4p_ramp_price_impact.py` (28 test). Kapsam: iki ramp
geleneğinin gerçekten farklı olması, FW9 ölçekleyicisinin fit'le birebir
eşleşmesi, birim kök ve yakınsama uyarılarının doğrulanması, kesme terimi
türetiminin FW4'ün R1'ini ve yaml'ın R3'ünü yeniden üretmesi, FW4'ün
yayımladığı fiyatların kendi ızgarasında geri gelmesi, ölçekleyici
uyuşmazlığının reddedilmesi, climatology yolunun kullanıldığı, merkezleme
özdeşliği, gömülebilirlik ve çıktıların sıfırdan yeniden üretimi.
