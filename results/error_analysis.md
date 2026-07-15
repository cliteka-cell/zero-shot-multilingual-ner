# Error Analysis (checkpoint: `ner_checkpoint.pt`)


## tr

Per-class recall: {'O': 0.9316489476176129, 'PER': 0.9217663761872124, 'ORG': 0.8968831402137789, 'LOC': 0.8405101654249274}

Confusion matrix (rows=true, cols=pred, order=['O', 'PER', 'ORG', 'LOC']):
```
[[43290   347  1944   885]
 [  155  9414   533   111]
 [  292   261  9985   595]
 [  258   124   881  6656]]
```


### Qualitative errors (8 examples)


**Example 1:**

| Token | True | Pred |
|---|---|---|
| Eğitimini | O | O |
| o | O | O |
| kentte | O | O |
| bitirdikten | O | O |
| sonra | O | O |
| , | O | O |
| Bağdat | B-LOC | B-ORG **<-- mismatch** |
| Nizamiye | B-ORG | I-ORG |
| Medresesi'ne | I-ORG | I-ORG |
| devam | O | O |
| etti | O | O |
| . | O | O |

**Example 2:**

| Token | True | Pred |
|---|---|---|
| Daha | O | O |
| sonra | O | O |
| Cumhurbaşkanı | O | B-ORG **<-- mismatch** |
| Michel | B-PER | B-PER |
| Djotodia | I-PER | I-PER |
| kabinesinde | O | B-ORG **<-- mismatch** |
| görev | O | O |
| yaptı | O | O |
| ve | O | O |
| Başbakan | O | B-ORG **<-- mismatch** |
| olarak | O | O |
| atanmadan | O | O |
| önce | O | O |
| geçici | O | B-ORG **<-- mismatch** |
| Cumhurbaşkanı | O | B-ORG **<-- mismatch** |
| Catherine | B-PER | B-PER |
| Samba-Panza'nın | I-PER | I-PER |
| bir | O | O |
| danışmanı | O | O |
| olarak | O | O |
| görev | O | O |
| yaptı | O | O |
| . | O | O |

**Example 3:**

| Token | True | Pred |
|---|---|---|
| YÖNLENDİRME | O | B-ORG **<-- mismatch** |
| Eurovision | B-LOC | I-ORG **<-- mismatch** |
| Çocuk | I-LOC | I-ORG **<-- mismatch** |
| Şarkı | I-LOC | I-ORG **<-- mismatch** |
| Yarışması'nda | I-LOC | I-ORG **<-- mismatch** |
| Beyaz | I-LOC | I-ORG **<-- mismatch** |
| Rusya | I-LOC | I-LOC |

**Example 4:**

| Token | True | Pred |
|---|---|---|
| Yaygın | O | O |
| olduğu | O | O |
| eyalet | O | B-LOC **<-- mismatch** |
| Los | B-ORG | B-LOC **<-- mismatch** |
| Santos'dur | I-ORG | I-LOC **<-- mismatch** |
| . | O | O |

**Example 5:**

| Token | True | Pred |
|---|---|---|
| Yönetim | O | O |
| merkezi | O | O |
| , | O | O |
| en | O | O |
| büyük | O | O |
| kenti | O | B-LOC **<-- mismatch** |
| ve | O | O |
| eski | O | O |
| merkezi | O | O |
| Ahmedabad | B-LOC | B-LOC |
| yakınlarındaki | O | O |
| Gandhinagar'dır | B-LOC | B-LOC |
| . | O | O |

**Example 6:**

| Token | True | Pred |
|---|---|---|
| Dallas | B-ORG | B-ORG |
| ( | I-ORG | I-ORG |
| dizi | I-ORG | I-ORG |
| , | I-ORG | I-ORG |
| 2012 | I-ORG | I-ORG |
| ) | I-ORG | I-ORG |
| , | O | O |
| 2012'de | O | O |
| çekilmeye | O | O |
| başlanan | O | O |
| yeniden | O | O |
| yapım | O | I-ORG **<-- mismatch** |
| televizyon | O | I-ORG **<-- mismatch** |
| dizisi | O | I-ORG **<-- mismatch** |
| . | O | O |

**Example 7:**

| Token | True | Pred |
|---|---|---|
| YÖNLENDİRME | O | B-ORG **<-- mismatch** |
| 1996 | B-LOC | B-ORG **<-- mismatch** |
| UEFA | I-LOC | I-ORG **<-- mismatch** |
| Intertoto | I-LOC | I-ORG **<-- mismatch** |
| Kupası | I-LOC | I-ORG **<-- mismatch** |

**Example 8:**

| Token | True | Pred |
|---|---|---|
| YÖNLENDİRME | O | B-ORG **<-- mismatch** |
| Office | B-ORG | I-ORG |
| of | I-ORG | I-ORG |
| Film | I-ORG | I-ORG |
| and | I-ORG | I-ORG |
| Literature | I-ORG | I-ORG |
| Classification | I-ORG | I-ORG |
| ( | I-ORG | I-ORG |
| Avustralya | I-ORG | I-ORG |
| ) | I-ORG | I-ORG |

## de

Per-class recall: {'O': 0.9528942232062225, 'PER': 0.9151793450008664, 'ORG': 0.8462745098039216, 'LOC': 0.8292413793103448}

Confusion matrix (rows=true, cols=pred, order=['O', 'PER', 'ORG', 'LOC']):
```
[[65420   146  2217   871]
 [  243 10563   575   161]
 [  447   271  8632   850]
 [  232   182   824  6012]]
```


### Qualitative errors (8 examples)


**Example 1:**

| Token | True | Pred |
|---|---|---|
| WEITERLEITUNG | O | O |
| Hu | B-LOC | O **<-- mismatch** |
| ( | I-LOC | O **<-- mismatch** |
| Xi’an | I-LOC | B-LOC |
| ) | I-LOC | O **<-- mismatch** |

**Example 2:**

| Token | True | Pred |
|---|---|---|
| Runde | O | O |
| rammte | O | O |
| Lorenzo | B-PER | B-PER |
| Bandini | I-PER | I-PER |
| an | O | O |
| zweiter | O | O |
| Stelle | O | O |
| liegend | O | O |
| die | O | O |
| Streckenbegrenzung | O | O |
| aus | O | O |
| Strohballen | O | B-LOC **<-- mismatch** |
| . | O | O |

**Example 3:**

| Token | True | Pred |
|---|---|---|
| Alexandrasittiche | O | O |
| kommen | O | O |
| außerdem | O | O |
| in | O | O |
| den | O | O |
| Galeriewäldern | B-LOC | B-LOC |
| entlang | O | O |
| von | O | O |
| Wasserläufen | O | B-LOC **<-- mismatch** |
| vor | O | O |
| . | O | O |

**Example 4:**

| Token | True | Pred |
|---|---|---|
| Tour | B-LOC | B-ORG **<-- mismatch** |
| d’Aï | I-LOC | I-ORG **<-- mismatch** |
| – | O | O |
| 2332 | O | O |
| m | O | O |

**Example 5:**

| Token | True | Pred |
|---|---|---|
| WM | B-ORG | B-ORG |
| 1995 | I-ORG | I-ORG |
| in | O | I-ORG **<-- mismatch** |
| Thunder | B-LOC | I-ORG **<-- mismatch** |
| Bay | I-LOC | I-LOC |

**Example 6:**

| Token | True | Pred |
|---|---|---|
| ' | O | O |
| '' | O | O |
| Heiliges | B-ORG | B-LOC **<-- mismatch** |
| Römisches | I-ORG | I-LOC **<-- mismatch** |
| Reich | I-ORG | I-LOC **<-- mismatch** |
| '' | O | O |
| ' | O | O |

**Example 7:**

| Token | True | Pred |
|---|---|---|
| Im | O | O |
| Endspiel | O | O |
| unterlag | O | O |
| sie | O | O |
| im | O | O |
| Alter | O | O |
| von | O | O |
| 40 | O | O |
| Jahren | O | O |
| im | O | O |
| `` | O | O |
| ältesten | O | O |
| '' | O | O |
| WTA-Finale | O | B-ORG **<-- mismatch** |
| aller | O | O |
| Zeiten | O | O |
| überraschend | O | O |
| , | O | O |
| wenn | O | O |
| auch | O | O |
| knapp | O | O |
| , | O | O |
| der | O | O |
| 33-jährigen | O | O |
| Tamarine | B-PER | B-PER |
| Tanasugarn | I-PER | I-PER |
| . | O | O |

**Example 8:**

| Token | True | Pred |
|---|---|---|
| Die | O | O |
| Geschäftsführung | O | O |
| sitzt | O | O |
| in | O | O |
| Münster | B-LOC | B-LOC |
| ( | I-LOC | I-LOC |
| Westfalen | I-LOC | I-LOC |
| ) | I-LOC | O **<-- mismatch** |
| . | O | O |

## ar

Per-class recall: {'O': 0.7675642453808628, 'PER': 0.6858403325697803, 'ORG': 0.901930671347082, 'LOC': 0.5899789932311522}

Confusion matrix (rows=true, cols=pred, order=['O', 'PER', 'ORG', 'LOC']):
```
[[19982   437  4256  1358]
 [  178  8084  3063   462]
 [  255   254 12333   832]
 [  255   160  4855  7583]]
```


### Qualitative errors (8 examples)


**Example 1:**

| Token | True | Pred |
|---|---|---|
| تعلم | O | O |
| في | O | O |
| جامعة | B-ORG | B-ORG |
| نورث | I-ORG | I-ORG |
| وسترن | I-ORG | I-ORG |
| في | I-ORG | O **<-- mismatch** |
| . | O | O |

**Example 2:**

| Token | True | Pred |
|---|---|---|
| تحويل | O | B-ORG **<-- mismatch** |
| ده‌شهر | B-LOC | I-ORG **<-- mismatch** |
| ( | I-LOC | I-LOC |
| مقاطعة | I-LOC | I-LOC |
| كلاردشت | I-LOC | I-LOC |
| ) | I-LOC | I-LOC |

**Example 3:**

| Token | True | Pred |
|---|---|---|
| تحويل | O | B-ORG **<-- mismatch** |
| ويلفريد | B-PER | I-ORG **<-- mismatch** |
| أورباين | I-PER | I-ORG **<-- mismatch** |
| إيلفيس | I-PER | I-ORG **<-- mismatch** |
| إندزانغا | I-PER | I-ORG **<-- mismatch** |

**Example 4:**

| Token | True | Pred |
|---|---|---|
| تحويل | O | B-LOC **<-- mismatch** |
| مقاطعة | B-LOC | I-LOC |
| روش | I-LOC | I-LOC |
| ( | I-LOC | I-LOC |
| كانساس | I-LOC | I-LOC |
| ) | I-LOC | I-LOC |

**Example 5:**

| Token | True | Pred |
|---|---|---|
| تحويل | O | B-ORG **<-- mismatch** |
| قائمة | B-LOC | I-ORG **<-- mismatch** |
| المواضيع | I-LOC | I-ORG **<-- mismatch** |
| الأساسية | I-LOC | I-ORG **<-- mismatch** |
| في | I-LOC | I-ORG **<-- mismatch** |
| علم | I-LOC | I-ORG **<-- mismatch** |
| الاجتماع | I-LOC | I-ORG **<-- mismatch** |

**Example 6:**

| Token | True | Pred |
|---|---|---|
| بشير | B-PER | B-PER |
| الجميل | I-PER | I-PER |
| - | O | O |
| الرئيس | O | B-ORG **<-- mismatch** |
| السابق | O | I-ORG **<-- mismatch** |
| المنتخب، | O | I-ORG **<-- mismatch** |
| مؤسس | O | O |
| القوات | B-ORG | B-ORG |
| اللبنانية | I-ORG | I-ORG |
| ، | O | O |
| نجل | O | O |
| بيار | B-PER | B-PER |
| الجميل | I-PER | I-PER |
| . | O | O |

**Example 7:**

| Token | True | Pred |
|---|---|---|
| ' | O | O |
| '' | O | O |
| راغون | O | B-LOC **<-- mismatch** |
| يسنيتس | O | I-LOC **<-- mismatch** |
| '' | O | O |
| ' | O | O |
| هي | O | O |
| بلدة | B-LOC | B-LOC |
| ألمانية | I-LOC | I-LOC |
| تقع | O | O |
| في | O | O |
| ألمانيا | B-LOC | B-LOC |
| في | O | O |
| ساكسونيا | B-LOC | B-LOC |
| أنهالت | I-LOC | O **<-- mismatch** |
| . | O | O |

**Example 8:**

| Token | True | Pred |
|---|---|---|
| تحويل | O | B-LOC **<-- mismatch** |
| أتيا | B-LOC | I-LOC |
| ( | I-LOC | I-LOC |
| سرقسطة | I-LOC | I-LOC |
| ) | I-LOC | I-LOC |