# FW6a -- F2.5 karşılaştırmasının v2 kappa ile yeniden üretimi

**Türkçe rapor.** Bütün hesap ve kod İngilizce; yalnızca bu belge Türkçe.
Kurallar: yeni parametre kestirimi yok (üretim yaml'ı olduğu gibi
kullanıldı), look-ahead yok, sentetik veri yok, fiyatlama üretim
climatology kovaryat yoluyla yapıldı (FW12b kuralı), üretim yaml'ı ve
`model_limitations.md` ve `PROJECT_STATUS_AND_FUTURE_WORK.md`
değiştirilmedi, kabul edilmiş çıktı ağaçlarına yazılmadı.

## 1. Amaç

`outputs/market_calibration_final/model_comparison_pooled_vs_M9.md` (F2.5)
`K = 3000` Avrupa alım opsiyonunu 24 / 72 / 168 / 336 saatte üç artık
belirtimi altında fiyatlıyor:

| varyant | sigma_y | kappa |
|---|---|---|
| `pooled_M0_kappa` | havuzlanmış tek oynaklık | M0'ın kendi kappası |
| `pooled_M9_kappa` | havuzlanmış tek oynaklık | M9'un kappası |
| `M9_prod` | üretim iki rejimli TVTP | M9'un kappası |

Bu belgedeki M9 kappası taşıyan bütün sayılar `kappa = 4.108274e-06 /h`
(`phi = 0.999995891734`) ile üretilmiş, yani v2 öncesi değerle. Üretim
yaml'ı ise `kappa = 0.078394 /h` (`phi = 0.9246`) taşıyor -- 19.082 kat
daha hızlı. Bu haliyle F2.5 makalenin geri kalanıyla parametre bakımından
tutarsız. FW6a aynı karşılaştırmayı üretim kappasıyla yeniden üretir.

## 2. Yöntem

**Sabit tutulanlar (F2.5'teki gibi).** Vadeli eğri (`smooth_constrained`,
Ocak çapası `spot_to_next_linear`), `pi_filtered = (0.931977, 0.068023)`,
TVTP katsayıları, kullanım fiyatı 3000 TRY/MWh, dört vade,
`r_annual = 0.40`, havuzlanmış sigma inşası.

**Havuzlanmış sigma.** M0'ın iki oynaklığı M9'un durağan doluluk
oranlarıyla birleştirildi:

```
sigma_pooled = sqrt(0.325422 * 0.006123^2 + 0.674578 * 0.164255^2)
             = 0.134952
```

`FrozenM2Parameters` sınıfı `sigma_y[1] > sigma_y[0]` değişmezini zorunlu
kıldığı için (bkz. `pde_option_model/params_frozen.py`), havuzlanmış değer
F2.5'teki gibi +/- %0,01 ayrıldı. İki rejim böylece sayısal olarak ayırt
edilemez kalır ama değişmez ihlal edilmez.

**Değişenler.** Yalnızca iki M9-kappa varyantının kappası
(4.108274e-06 -> 0.078394 /h) ve artık durum ızgarasının üretim değerine
sabitlenmesi (1201 düğüm, `n_std = 6`, `n_time_steps = max(96, 2/saat)`).

**Kovaryat yolu.** `run_pde.py price` ile aynı: climatology `z(t-1)` yolu
`inputs/historical/rd_standardized.csv` üzerinden, eğitim sonu
2022-12-31 20:00 UTC, gecikme 1 saat, 0,25 saatlik ana ızgarada kurulup
çözücü ızgarasına doğrusal aradeğerlenir. Sabit-geçiş (z = 0) sınırına
hiçbir yerde düşülmedi.

**Kod.** `scripts/fw6a/f25_v2_kappa.py`. F2.5'i üreten adanmış bir betik
depoda yoktu -- F2.5 belgesinin kendisi "üretim kodu ellenmedi, havuzlama
yalnızca boru hattı yapılandırmasıyla yapıldı" diyor ve git geçmişinde
böyle bir betik bulunmuyor -- bu yüzden üretim boru hattını birebir
kullanan bir üretici yazıldı.

## 3. Yeniden üretimin doğrulanması

Üretici, iki dönemi de mevcut kayıtlara **basamak basamak** oturtuyor:

* v2 kappa ile:
  `outputs/market_calibration_final/model_comparison_pooled_vs_M9.csv`
  dosyasındaki 12 satırın beş sayısal sütununun tamamı 1e-4 içinde.
* v2 öncesi kappa ile: F2.5 markdown'ında **yazılı olan** sayıların tamamı
  (hem alım fiyatı hem artık sapma tabloları).

Yani hem eski hem yeni taraf bağımsız olarak doğrulandı; aradaki fark
yalnızca kappadan geliyor.

## 4. Yol boyunca çıkan bulgu: kabul edilmiş çiftin kendi içinde tutarsız olması

v2 kappa refit commit'i (`e1140a2`)
`model_comparison_pooled_vs_M9.csv` dosyasını yeniden üretmiş, ama
yanındaki `.md` dosyasını güncellememiş. Dolayısıyla:

* CSV **zaten** v2 sayılarını taşıyor,
* markdown ise v2 öncesi sayıları yazıyor ve kendi kaynağı olarak o CSV'yi
  gösteriyor.

Kappadan bağımsız olan `pooled_M0_kappa` satırları iki dosyada da aynı
kaldığı için tutarsızlık tam olarak M9-kappa satırlarına yerleşiyor. Bu
durum bir teste bağlandı
(`test_the_f25_markdown_is_stale_relative_to_its_own_csv`): markdown ile
CSV bir gün uyuşursa test FW6a'nın gerekçesinin kalmadığını söyleyerek
düşer. Kabul edilmiş ağaçtaki iki dosyaya da dokunulmadı.

## 5. Sonuçlar

### 5.1 Eski ve yeni yan yana (alım fiyatı, TRY/MWh)

| vade | varyant | eski | yeni | fark | % |
|---:|---|---:|---:|---:|---:|
| 24 h | `pooled_M0_kappa` | 724,09 | 724,09 | 0,00 | 0,00 |
| 24 h | `pooled_M9_kappa` | 731,58 | 353,49 | -378,09 | -51,68 |
| 24 h | `M9_prod` | 368,17 | 163,93 | -204,24 | -55,48 |
| 72 h | `pooled_M0_kappa` | 1254,65 | 1254,65 | 0,00 | 0,00 |
| 72 h | `pooled_M9_kappa` | 1292,87 | 356,69 | -936,18 | -72,41 |
| 72 h | `M9_prod` | 687,04 | 166,75 | -520,29 | -75,73 |
| 168 h | `pooled_M0_kappa` | 1854,30 | 1854,30 | 0,00 | 0,00 |
| 168 h | `pooled_M9_kappa` | 1985,65 | 353,84 | -1631,80 | -82,18 |
| 168 h | `M9_prod` | 1073,90 | 164,95 | -908,95 | -84,64 |
| 336 h | `pooled_M0_kappa` | 2450,70 | 2450,70 | 0,00 | 0,00 |
| 336 h | `pooled_M9_kappa` | 2799,39 | 348,91 | -2450,48 | -87,54 |
| 336 h | `M9_prod` | 1525,78 | 161,84 | -1363,94 | -89,39 |

`pooled_M0_kappa` M0'ın kendi kappasını taşıdığı için iki dönemde de aynı:
kappa düzenlemesini yalıtan kontrol satırı budur. Artık sapma sütunları da
aynı yönde hareket ediyor (v2 ile %49-87 daralma); `F(T)` her varyantta
değişmedi.

### 5.2 Havuzlanmış ile M9 arasındaki fark nasıl değişti

v2 öncesi (yayımlanmış F2.5):

| vade | `M9` vs `pooled_M0_kappa` | `M9` vs `pooled_M9_kappa` |
|---:|---:|---:|
| 24 h | -%49,15 | -%49,67 |
| 72 h | -%45,24 | -%46,86 |
| 168 h | -%42,09 | -%45,92 |
| 336 h | -%37,74 | -%45,50 |

v2 kappa:

| vade | `M9` vs `pooled_M0_kappa` | `M9` vs `pooled_M9_kappa` |
|---:|---:|---:|
| 24 h | -%77,36 | -%53,63 |
| 72 h | -%86,71 | -%53,25 |
| 168 h | -%91,10 | -%53,38 |
| 336 h | -%93,40 | -%53,61 |

## 6. Yorum

**(a) Fark artık vadeye göre düz.** `pooled_M9_kappa`'ya karşı iskonto v2
öncesinde 24-336 h boyunca 4,18 puanlık bir aralık tarıyordu; üretim
kappasıyla aralık 0,37 puana iniyor. Yarı ömür 8,84 saat olduğu için artık
varyansı daha 24 saatte durağan değerinin %97,7'sine ulaşıyor; tablodaki
bütün vadeler pratikte aynı durağan karışım varyansını fiyatlıyor.
`M9_prod` alım fiyatı buna paralel olarak vadeye göre düz (ortalamasının
%3,0'ü kadar yayılım: 163,93 / 166,75 / 164,95 / 161,84). F2.5'in yorum
bölümündeki **3. madde** -- yüzde farkın 24 saatten 336 saate 8 puan
daralmasının "rejim koşullamasının nicel imzası" olduğu iddiası -- üretim
parametreleriyle ayakta kalmıyor.

**(b) Kalan düz fark bir oynaklık *seviyesi* farkı; F2.5'in "rejim
koşullamasının değerini yalıtıyor" iddiası geri çekiliyor.** Karşılaştırmanın
iki tarafı aynı oynaklık seviyesini hiç paylaşmıyor: havuzlanmış taban
sigmasını M0'ın kestirimlerinden (0,006123 / 0,164255) alıp 0,134952'ye
varıyor, `M9_prod` ise M9'un kendi sigmalarıyla (0,003535 / 0,092407)
çalışıyor ve bunların aynı doluluk ağırlıklarıyla etkin oynaklığı
**0,075923**. Oran 0,5626. Vade sonundaki gerçekleşen artık sapma oranı ise
0,531-0,533. Yani düz farkın tamamı bu sigma oranı. Ölçülen şey, M0 ile M9
kestirimlerinin oynaklık seviyesi konusundaki anlaşmazlığı; süzülmüş bir
rejim inancı tutmanın değeri değil. v2 öncesinde bu karışıklık vade yapısı
tarafından örtülüyordu; varyans en kısa vadenin içinde doyunca fark sigma
oranına çöküyor ve karışıklık görünür hale geliyor.

**(c) Rejim karışımının etkisi nerede ölçülüyor.** FW9 f turu, *aynı durağan
varyansta* tek rejimli bir OU fiyatlıyor
(`kappa_sensitivity_isovariance_v2.md`): 72 saat ATM'de 183,4 TRY/MWh'ye
karşı üretimin 166,75'i, yani eşit varyansta iki rejimli karışım yaklaşık
-%9 değerinde. Buradaki sayıyla aynı işarette ama bir mertebe küçük: karışım
kanalı ince-kuyruk düzeltmesi, ~-%53'ün kalan ~44 puanı ise M0-M9 oynaklık
seviyesi farkı.
Zamanla değişen geçişlerin desteği de bu karşılaştırmadan değil, FW9'un
olabilirlik oranı testinden geliyor (`lr_test_TVTP_vs_constant.csv`:
LR = 1178,66, df = 2).

**(d) `pooled_M0_kappa` üçüncü bir kanalı da içeri katıyor.** Kappası M0'ın
kendisi: v2 öncesinde M9'unkinden ~200 kat *hızlıydı*, şimdi üretimden
~96 kat *yavaş*, dolayısıyla artık varyansı vadeyle büyümeye devam ediyor.
-%77 -> -%93 sütunu bu yüzden oynaklık seviyesinin üstüne bir de ortalamaya
dönme hızını karşılaştırıyor; o da bir rejim koşullaması ölçümü değil.

**(e) F2.5'in 4. maddesi tersine dönüyor.** O madde, yalnız kappa
değişiminin (aynı havuzlanmış sigma ile `pooled_M0_kappa` ->
`pooled_M9_kappa`) alım fiyatını %1-14 yükselttiğini, yani rejim
koşullamasına atfettiğinden bir mertebe küçük olduğunu söylüyor. Üretim
kappasıyla aynı değişim fiyatı -%51 ile -%86 arasında oynatıyor. Ortalamaya
dönme zaman ölçeği artık iki kanaldan **büyük** olanı, ikincil olanı değil.

**(f) Değişmeyenler.** F2.5'in (a), (b) ve (d) çekinceleri aynen geçerli:
havuzlanmış sigma hâlâ M9'un durağan doluluğunu vekil alıyor,
karşılaştırma hedefi hâlâ kestirilmiş M0 değil tek rejimli havuzlanmış
taban (M0'ın alpha/gamma'ları bu handoff'ta yok, dolayısıyla TVTP'ye özgü
ayrıştırma hâlâ ertelenmiş durumda), ve `E^Q[P_t] = F(t)` özdeşliği bütün
varyantlarda ve iki dönemde de makine hassasiyetinde sağlanıyor (azami
mutlak hata 0,0 TRY/MWh). F2.5'in (c) çekincesi ("kappa uyuşmazlığı") ise
yukarıdaki (d) maddesi olarak çok daha güçlü okunmalı.

F2.5 markdown'ı kabul edilmiş bir çıktı ağacında olduğu için
değiştirilmedi; dokümantasyon güncellemesi onu `outputs/f25_v2_kappa/`
tarafından geçersiz kılınmış olarak işaretlemeli.

## 7. Dosyalar ve testler

`outputs/f25_v2_kappa/` altında:

| dosya | içerik |
|---|---|
| `f25_v2_kappa.csv` | v2 kappa ile 12 satır (yeni) |
| `f25_pre_v2_reproduction.csv` | v2 öncesi kappa ile 12 satır (eski, yeniden üretilmiş) |
| `f25_old_vs_new.csv` | yan yana, fark ve yüzde sütunlarıyla |
| `f25_gap_v2.csv`, `f25_gap_pre_v2.csv` | havuzlanmış-M9 farkı, vade bazında |
| `f25_v2_kappa.md` | İngilizce sonuç notu |
| `run_manifest.json` | girdiler, ayarlar, kappa ve havuzlama sağlaması |
| `hash_manifest_before.txt`, `hash_manifest_after.txt` | korunan 128 dosyanın SHA-256 özetleri |

Testler: `tests/test_fw6a_f25_v2_kappa.py` (26 test). Kapsam: havuzlama
aritmetiği, varyant kablolaması, kovaryat yolunun üretim CLI'siyle birebir
eşitliği, korunan dizinlere yazma reddi, kabul edilmiş CSV ve yayımlanmış
markdown ile eşleşme, merkezleme özdeşliği, ve iki dönemin sıfırdan
yeniden üretimi (`slow` işaretli).

Eski dosyaların hiçbiri silinmedi veya değiştirilmedi.
