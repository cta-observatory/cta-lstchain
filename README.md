# cta-lstchain [![Build Status](https://github.com/cta-observatory/cta-lstchain/workflows/CI/badge.svg?branch=master)](https://github.com/cta-observatory/cta-lstchain/actions?query=workflow%3ACI+branch%3Amaster) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6344673.svg)](https://doi.org/10.5281/zenodo.6344673)

Repository for the high level analysis of the LST.
The analysis is heavily based on [ctapipe](https://github.com/cta-observatory/ctapipe), adding custom code for mono reconstruction.

- **Source code:** https://github.com/cta-observatory/cta-lstchain
- **Documentation:** https://cta-observatory.github.io/cta-lstchain/

Note that notebooks are currently not tested and not guaranteed to be up-to-date.   
In doubt, refer to tested code and scripts: basic functions of lstchain (reduction steps R0-->DL1 and DL1-->DL2) 
are unit tested and should be working as long as the build status is passing.

## Install

- You will need to install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or [anaconda](https://www.anaconda.com/distribution/#download-section) first. 


### As user

```
LSTCHAIN_VER=0.9.6  (or the version you want to install - usually the latest release)
wget https://raw.githubusercontent.com/cta-observatory/cta-lstchain/v$LSTCHAIN_VER/environment.yml
conda env create -n lst -f environment.yml
conda activate lst
pip install lstchain==$LSTCHAIN_VER
rm environment.yml
```

### As developer

- Create and activate the conda environment:
```
git clone https://github.com/cta-observatory/cta-lstchain.git
cd cta-lstchain
conda env create -f environment.yml
conda activate lst-dev
```

**Note**: To prevent packages you installed with `pip install --user` from taking precedence over the conda environment, run:
```
conda env config vars set PYTHONNOUSERSITE=1 -n <environment_name>
```

To update the environment (e.g. when depenencies got updated), use:
```
conda env update -n lst-dev -f environment.yml
```

- Install lstchain in developer mode:

```
pip install -e .
```

To run some of the tests, some non-public test data files are needed.
These tests will not be run locally if the test data is not available,
but are always run in the CI.

To download the test files locally, run `./download_test_data.sh`.
It will ask for username and password and requires `wget` to be installed.
Ask one of the project maintainers for the credentials. If 
you are a member of the LST collaboration you can also obtain them here:

https://ctaoobservatory.sharepoint.com/:i:/r/sites/ctan-onsite-it/Shared%20Documents/General/information_2.jpg?csf=1&web=1&e=suUkV6

To run the tests that need those private data file, add `-m private_data`
to the pytest call, e.g.:

```
pytest -m private_data -v lstchain
```

To run all tests, run
```
pytest -m 'private_data or not private_data' -v lstchain
```

## Contributing

All contribution are welcomed.

Guidelines are the same as [ctapipe's ones](https://cta-observatory.github.io/ctapipe/development/index.html)    
See [here](https://cta-observatory.github.io/ctapipe/development/pullrequests.html) for the general guidelines on how to make a pull request to contribute to the repository. Since the addition of the private data, the CI tests for Pull Requests from forks are not working, therefore we would like to ask to push your modified branches directly to the main cta-lstchain repo. If you do not have writing permissions in the repo, please contact one of the main developers. 


## Report issue / Ask a question

Use [GitHub Issues](https://github.com/cta-observatory/cta-lstchain/issues).

## Cite

If you use lstchain in a publication, please cite the exact version you used from Zenodo _Cite as_, see https://doi.org/10.5281/zenodo.6344673

Please also cite the following proceedings:

```
@inproceedings{Abe:2021Sq,
  author = "Abe, Hyuga  and  Aguasca, Arnau  and  Agudo, Ivan  and  Antonelli, Lucio Angelo  and  Aramo, Carla  and  Armstrong, Thomas  and  Artero, Manuel  and  Asano, Katsuaki  and  Ashkar, Halim  and  Aubert, Pierre  and  Baktash, Ali  and  Bamba, Aya  and  Baquero Larriva, Andres  and  Baroncelli, Leonardo  and  Barres de Almeida, Ulisses  and  Barrio, Juan Abel  and  Batković, Ivana  and  Becerra Gonzalez, Josefa  and  Bernardos, Mabel  and  Berti, Alessio  and  Biederbeck, Noah  and  Bigongiari, Ciro  and  Blanch, Oscar  and  Bonnoli, Giacomo  and  Bordas, Pol  and  Bose, Debanjan  and  Bulgarelli, Andrea  and  Burelli, Irene  and  Buscemi, Mario  and  Cardillo, Martina  and  Caroff, Sami  and  Carosi, Alessandro  and  Cassol, Franca  and  Cerruti, Matteo  and  Chai, Yating  and  Cheng, Ks  and  Chikawa, Michiyuki  and  Chytka, Ladislav  and  Contreras, Jose Luis  and  Cortina, Juan  and  Costantini, Heide  and  Dalchenko, Mykhailo  and  De Angelis, Alessandro  and  de Bony de Lavergne, Mathieu  and  Deleglise, Guillaume  and  Delgado, Carlos  and  Delgado Mengual, Jordi  and  Della Volpe, Domenico  and  Depaoli, Davide  and  Di Pierro, Federico  and  Di Venere, Leonardo  and  Díaz, Carlos  and  Dominik, Rune Michael  and  Dominis Prester, Dijana  and  Donini, Alice  and  Dorner, Daniela  and  Doro, Michele  and  Elsässer, Dominik  and  Emery, Gabriel  and  Escudero, Juan  and  Fiasson, Armand  and  Foffano, Luca  and  Fonseca, Maria Victoria  and  Freixas Coromina, Lluis  and  Fukami, Satoshi  and  Fukazawa, Yasushi  and  Garcia, Enrique  and  Garcia López, Ramon  and  Giglietto, Nicola  and  Giordano, Francesco  and  Gliwny, Pawel  and  Godinovic, Nikola  and  Green, David  and  Grespan, Pietro  and  Gunji, Shuichi  and  Hackfeld, Jonas  and  Hadasch, Daniela  and  Hahn, Alexander  and  Hassan, Tarek  and  Hayashi, Kohei  and  Heckmann, Lea  and  Heller, Matthieu  and  Herrera Llorente, Javier  and  Hirotani, Kouichi  and  Hoffmann, Dirk  and  Horns, Dieter  and  Houles, Julien  and  Hrabovsky, Miroslav  and  Hrupec, Dario  and  Hui, David  and  Hütten, Moritz  and  Inada, Tomohiro  and  Inome, Yusuke  and  Iori, Maurizio  and  Ishio, Kazuma  and  Iwamura, Yuki  and  Jacquemont, Mikael  and  Jiménez Martínez, Irene  and  Jouvin, Léa  and  Jurysek, Jakub  and  Kagaya, Mika  and  Karas, Vladimir  and  Katagiri, Hideaki  and  Kataoka, Jun  and  Kerszberg, Daniel  and  Kobayashi, Yukiho  and  Kong, Albert  and  Kubo, Hidetoshi  and  Kushida, Junko  and  Lamanna, Giovanni  and  Lamastra, Alessandra  and  Le Flour, Thierry  and  Longo, Francesco  and  Lopez-Coto, Ruben  and  López-Moya, Marcos  and  López-Oramas, Alicia  and  Luque-Escamilla, Pedro L.  and  Majumdar, Pratik  and  MAKARIEV, Martin  and  Mandat, Dusan  and  Manganaro, Marina  and  Mannheim, Karl  and  Mariotti, Mosè  and  Marquez, Patricia  and  Marsella, Giovanni  and  Martí, Josep  and  Martinez, Oibar  and  Martínez, Gustavo  and  Martinez, Manel  and  Marusevec, Petra  and  Mas, Alvaro  and  Maurin, Gilles  and  Mazin, Daniel  and  Mestre Guillen, Enrique  and  Mićanović, Saša  and  Miceli, Davide  and  Miener, Tjark  and  Miranda, Jose Miguel  and  Miranda, Luis David Medina  and  Mirzoyan, Razmik  and  Mizuno, Tsunefumi  and  Molina, Edgar  and  Montaruli, Teresa  and  Monteiro, Inocencio  and  Moralejo, Abelardo  and  Morcuende, Daniel  and  Moretti, Elena  and  Morselli, Aldo  and  Mrakovcic, Karlo  and  Murase, Kohta  and  Nagai, Andrii  and  Nakamori, Takeshi  and  Nickel, Lukas  and  Nieto, Daniel  and  Nievas, Mireia  and  Nishijima, Kyoshi  and  Noda, Koji  and  Nosek, Dalibor  and  Nöthe, Maximilian  and  Nozaki, Seiya  and  Ohishi, Michiko  and  Ohtani, Yoshiki  and  Oka, Tomohiko  and  Okazaki, Nao  and  Okumura, Akira  and  Orito, Reiko  and  Otero-Santos, Jorge  and  Palatiello, Michele  and  Paneque, David  and  Paoletti, Riccardo  and  Paredes, Josep Maria  and  Pavletić, Lovro  and  Pech, Miroslav  and  Pecimotika, Mario  and  Poireau, Vincent  and  Polo, Miguel  and  Prandini, Elisa  and  Prast, Julie  and  Priyadarshi, Chaitanya  and  Prouza, Michael  and  Rando, Riccardo  and  Rhode, Wolgang  and  Ribó, Marc  and  Rizi, Vincenzo  and  Rugliancich, Andrea  and  Ruiz, Jose Enrique  and  Saito, Takayuki  and  Sakurai, Shunsuke  and  Sanchez, David  and  Šarić, Toni  and  Saturni, Francesco Gabriele  and  Scherpenberg, Juliane  and  Schleicher, Bernd  and  Schubert, Jan Lukas  and  Schüssler, Fabian  and  Schweizer, Thomas  and  Seglar Arroyo, Monica  and  Shellard, Ronald Cintra  and  Sitarek, Julian  and  Sliusar, Vitalii  and  Spolon, Alessia  and  Strišković, Jelena  and  Strzys, Marcel  and  Suda, Yusuke  and  Sunada, Yuji  and  Tajima, Hiroyasu  and  Takahashi, Mitsunari  and  Takahashi, Hiromitsu  and  Takata, Jumpei  and  Takeishi, Ryuji  and  Tam, P. Thomas  and  Tanaka, Shuta  and  Tateishi, Dai  and  Tejedor, Luis Ángel  and  Temnikov, Petar  and  Terada, Yukikatsu  and  Terzic, Tomislav  and  Teshima, Masahiro  and  Tluczykont, Martin  and  Tokanai, Fuyuki  and  Torres, Diego F.  and  Travnicek, Petr  and  Truzzi, Stefano  and  Vacula, Martin  and  VAZQUEZ ACOSTA, Monica  and  VERGUILOV, Vassil  and  Verna, Gaia  and  Viale, Ilaria  and  Vigorito, Carlo Francesco  and  Vitale, Vincenzo  and  Vovk, Ievgen  and  Vuillaume, Thomas  and  Walter, Roland  and  Will, Martin  and  Yamamoto, Tokonatsu  and  Yamazaki, Ryo  and  Yoshida, Tatsuo  and  Yoshikoshi, Takanori  and  Zarić, Darko",
  title = "{Physics Performance of the Large Size Telescope prototype of the Cherenkov Telescope Array}",
  doi = "10.22323/1.395.0806",
  booktitle = "Proceedings of 37th International Cosmic Ray Conference {\textemdash} PoS(ICRC2021)",
  year = 2021,
  volume = "395",
  pages = "806"
}
```
