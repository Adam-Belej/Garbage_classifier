**Středoškolská odborná činnost**
**Obor č. 18: Informační technologie**

**Použití neuronových sítí pro rozpoznávání tříděného odpadu z obrázků**

**Adam Belej**
**Olomoucký kraj**
**Šternberk 2023**


**Středoškolská odborná činnost**
**Obor č. 18: Informační technologie**

**Použití neuronových sítí pro rozpoznávání tříděného odpadu z obrázků**
**Usage of neural networks in garbage classification from images**

**Autoři:** Adam Belej
** Škola:** Gymnázium, Šternberk, Horní náměstí 5, Šternberk
**Kraj:** Olomoucký
**Konzultant:** 

Šternberk 2023


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


**Anotace**

**Klíčová slova**

Neuronové sítě, Rozpoznávání obázků, Počítačové vidění

**Annotation**

**Keywords**

Neural networks, Image recognition, Computer vision


Obsah

[toc]


# 1 Úvod


# 2 Teoretická Část
## 2.1 Neuronová síť
Neuronová síť je složena z mnoha jednotek - neuronů, které jsou mezi sebou propojeny a komunikují mezi sebou. U počítačových neuronových sítí je mnoho druhů, v této části se však budu věnovat pouze vícevrstvým perceptronům (MLP - Multilayer perceptron), které jsem pro tuto práci použil.
### 2.1.1 Perceptron
Perceptron je základní jednotkou počítačových neuronových sítí, a jeho jádrem je algoritmus, který pro matici vstupních dat $x$ o délce $k$ spočítá skalární součin s vektorem váh (weights) $w$ a přičte k nim práh (bias) $b$, a následně na toto číslo použije aktivační funkci $f$. Vzorcem se to dá vyjádřit jako:
$y = f(\sum_{i=1}^k w_i  x_i + b)$
Mezi nejčastěji používané aktivační funkce patří logistická sigmoida[^1], ReLU (Rectified Linear Unit)[^1], hype rbolický tangens[^1] a softmax[^1]. 

[^1]: LaTeXový vzorce a grafy přidám později (+ testuju footnotes)
### 2.1.2 Vícevrstvý perceptron
Vícevrstvý perceptron, někdy také nazýván jako dopředná neuronová síť, se skládá z několika vrstev perceptronů, které jsou na sebe napojeny. První (vstupní) vrstva dostává jako vstup přímo původní vstupní data, další (skryté) vrstvy pak výstupy z předchozích vrstev. Poslední (výstupní) vrstva většinou u klasifikace do více kategorií má tolik neuronů, kolik je kategorií.  
### 2.1.3 Husté neuronové sítě
Husté neuronové sítě jsou druh sítí, kde každý neuron v dané vrstvě dostává jako vstup celou vstupní matici z předchozí vrstvy (v případě první vrstvy vstup od uživatele), a pro vstupní matici o délce $k$ má $k + 1$ paramterů (váhy pro každé $x_i$ a práh), a vstup pro n+1 vstrvu je matice výstupů n-té vrstvy o délce $l$, kde $l$ je počet neuronů  n-té vrstvy. Tento druh sítí je pro účely rozpoznávání obrázků silně neefektivní, o čemž svědčí i výsledky testování takovéto sítě (wip, přidám výsledky trénování, testování, grafy). Implmentace této sítě vypadá následovně: 
`tf.keras.models.Sequential(
            [
                tf.keras.layers.Rescaling(1. / 255),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=128, activation='relu'),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dense(units=32, activation='relu'),
                tf.keras.layers.Dense(units=16, activation='relu'),
                tf.keras.layers.Dense(units=num_of_classes)
            ])`

# 3 Implementace
## 3.1 Software pro tvorbu a augmentaci vstupních dat
### 3.1.1 Vstupní data
Nejdříve bylo potřeba vytvořit vstupní data v podobě fotek odpadků. Všechny fotky byly foceny na pozadí, které jsem vytvořil z papíru a grafitu, a které by mělo napodobovat pohybující se pás na třídící lince.
### 3.1.2 Zpracování a augmentace dat
Jelikož se jedná o velký objem dat, bylo potřeba zautomatizovat proces přeformátování fotek na velikost 512x512 pixelů. K tomuto jsem použil knihovnu [Katna][https://pypi.org/project/katna/], která za pomocí umělé inteligence hledá důležitou část obrázku tak, aby při ořezávání došlo k co možná nejmenší ztrátě dat. 

K augmentaci jsem použil knihovnu [Pillow][https://pypi.org/project/Pillow/]. Všechny obrázky byly nejdříve horizontálně převráceny, a pak otočeny o 90, 180 a 270°, což ve výsledku zosminásobilo objem vstupních dat.
# 4 Závěr