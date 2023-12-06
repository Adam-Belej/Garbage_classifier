**Středoškolská odborná činnost**
**Obor č. 18: Informační technologie**

**Použití neuronových sítí pro rozpoznávání tříděného odpadu z obrázků**

**Adam Belej**
**Olomoucký kraj**
**Šternberk 2023**

***

**Středoškolská odborná činnost**
**Obor č. 18: Informační technologie**

**Použití neuronových sítí pro rozpoznávání tříděného odpadu z obrázků**
**Usage of neural networks in garbage classification from images**

**Autoři:** Adam Belej
** Škola:** Gymnázium, Šternberk, Horní náměstí 5, Šternberk
**Kraj:** Olomoucký
**Konzultant:** 

Šternberk 2023

***

**Prohlášení**
Prohlašuji, že jsem svou práci SOČ vypracoval samostatně a použil jsem pouze prameny
a literaturu uvedené v seznamu bibliografických záznamů.
Prohlašuji, že tištěná verze a elektronická verze soutěžní práce SOČ jsou shodné.
Nemám závažný důvod proti zpřístupňování této práce v souladu se zákonem č. 121/2000 Sb.,
o právu autorském, o právech souvisejících s právem autorským a o změně některých zákonů
(autorský zákon) ve znění pozdějších předpisů.
Ve Šternberku dne  ………………………………………………
Adam Belej


**Poděkování**

***

**Anotace**

**Klíčová slova**

Neuronové sítě, Rozpoznávání obázků, Počítačové vidění

**Annotation**

**Keywords**

Neural networks, Image recognition, Computer vision

***

Obsah

[toc]

***

# 1 Úvod


# 2 Teoretická Část

## 2.1 Neuronová síť
Neuronová síť je složena z mnoha jednotek - neuronů, které jsou mezi sebou propojeny a komunikují mezi sebou. U počítačových neuronových sítí je mnoho druhů, v této části se však budu věnovat pouze vícevrstvým perceptronům (MLP - Multilayer perceptron), které jsem pro tuto práci použil.

### 2.1.1 Perceptron
Perceptron je základní jednotkou počítačových neuronových sítí, a jeho jádrem je algoritmus, který pro matici vstupních dat $x$ o délce $k$ spočítá skalární součin s vektorem váh (weights) $w$ a přičte k nim práh (bias) $b$, a následně na toto číslo použije aktivační funkci $g$. Vzorcem se to dá vyjádřit jako:
$f(x) = g(\sum_{i=1}^k w_i  x_i + b)$
Mezi nejčastěji používané aktivační funkce patří:

- logistická sigmoida $g(z) = \frac{1}{1 + e^{-z}}$ (obr. 2.1), 
- ReLU (Rectified Linear Unit) $g(z) = max(0, z)$ (obr. 2.2), 
- hyperbolický tangens $g(z) = tanh(z)$ (obr. 2.3),
- softmax[^1].

[^1]: LaTeXový vzorce a grafy přidám později (+ testuju footnotes)

### 2.1.2 Vícevrstvý perceptron
Vícevrstvý perceptron, někdy také nazýván jako dopředná neuronová síť, se skládá z několika vrstev perceptronů, které jsou na sebe napojeny. První (vstupní) vrstva dostává jako vstup přímo původní vstupní data, další (skryté) vrstvy pak výstupy z předchozích vrstev. Poslední (výstupní) vrstva většinou u klasifikace do více kategorií má tolik neuronů, kolik je kategorií.  

### 2.1.3 Husté neuronové sítě
Husté neuronové sítě jsou druh sítí, kde každý neuron v dané vrstvě dostává jako vstup celou vstupní matici z předchozí vrstvy (v případě první vrstvy vstup od uživatele), a pro vstupní matici o délce $k$ má $k + 1$ paramterů (váhy pro každé $x_i$ a práh), a vstup pro n+1 vstrvu je matice výstupů n-té vrstvy o délce $l$, kde $l$ je počet neuronů  n-té vrstvy. Tento druh sítí je pro účely rozpoznávání obrázků silně neefektivní, o čemž svědčí i výsledky testování takovéto sítě (graf na obrázku 2.1), kde se po druhé epoše přesnost modelu na trénovacím datasetu zastavila na 0.5827, a výsledná přesnost na testovacích datech byla (ve vlaku není wifi, pak to číslo z kagglu dopíšu zpětně). Implmentace této sítě v kódu vypadá následovně:

```
tf.keras.models.Sequential(
            [
                tf.keras.layers.Rescaling(1. / 255),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=128, activation='relu'),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dense(units=32, activation='relu'),
                tf.keras.layers.Dense(units=16, activation='relu'),
                tf.keras.layers.Dense(units=num_of_classes)
            ])

```

![Obrázek 2.1 - graf přesnosti a ztráty trénovacích a validačních dat během trénování](/images/dense_graph.png)

# 3 Implementace

## 3.1 Software pro tvorbu a augmentaci vstupních dat

### 3.1.1 Vstupní data
Nejdříve bylo potřeba vytvořit vstupní data v podobě fotek odpadků. Všechny fotky byly foceny na pozadí, které jsem vytvořil z papíru a grafitu, a které má napodobovat pohybující se pás na třídící lince.

### 3.1.2 Zpracování a tvorba dat
Bylo potřeba sjednotit formát dat, aby bylo možné je použít jako vstup pro neuronové sítě. Zároveň jsem zvolil poměr stran 1:1, aby byla jednodušší následná augmentace. Jelikož se jedná o velký objem dat, bylo potřeba zautomatizovat celý proces přeformátování fotek na velikost 512x512 pixelů, která by měla dostatečně zachovat objekty na fotkách, ale zároveň nebýt tak velká, aby velikost dat výrazně neztížila proces trénování sítí. K tomuto jsem nejprve použil knihovnu [Katna][https://pypi.org/project/katna/], která s využitím umělé inteligence hledá důležitou část obrázku tak, aby při ořezávání došlo k co možná nejmenší ztrátě dat. S její pomocí jsem obrázky přeformároval na poměr stran 1:1. Pak jsem využil knihovny  [Pillow][https://pypi.org/project/Pillow/] ke konverzi do formátu png a zmenšení obrázků na formát 512x512 pixelů.

### 3.1.3 Datová augmentace
Datová augmentace je pomětně rozšířená technika, při které se původní dataset rozšíří tím, že se mírně poupraví nebo pozmění původní data.

K augmentaci obrázků, které již byly v požadovaném formátu jsem použil opět knihovnu [Pillow][https://pypi.org/project/Pillow/]. Všechny obrázky byly nejdříve horizontálně převráceny, a pak otočeny o 90, 180 a 270°. Tímto způsobem jsem efektivně zosminádobil vstupní data pro trénování.

Celý proces od přeformátování až po augmentaci je zautomatizován v programu 'dataset-creator.py', který je co nejvíce generalizován tak, aby jej bylo možné využívat i pro budoucí projekty s jiným požadovaným formátem či velikostí obrázků. Tento program je spustitelný z terminálu, a jako vstupní parametry přijímá:
- šířku -W nebo --width (integer),
- výšku -H nebo --height (integer),
- cestu ke složce se vstupními soubory -i nebo --input_dir (string),
- cestu, kam následně uložit upravené fotky -o nebo --output_dir (string),
- formát, ve kterém budou fotky uloženy -e nebo --extension (string),
- zda augmentovat data, nebo je jen konvertovat na požadovanou velikost -a nebo --augmentation (bool)


# 4 Závěr
