# Online Learning Application Project - Social Influence & Advertising

## Team 15
---
| Surname            | Name      | Contact Info                       |
|:-------------------|:----------|:-----------------------------------|
| Ambrosini Barzaghi | Riccardo  | riccardo1.ambrosini@mail.polimi.it |
| Bonalumi           | Marco     | marco1.bonalumi@mail.polimi.it     |
| Careddu            | Gianmario | gianmario.careddu@mail.polimi.it   |
| Giaccaglia         | Pablo     | pablo.giaccaglia@mail.polimi.it    |
| Zoccheddu          | Sara      | sara.zoccheddu@mail.polimi.it      |

## Description

This repository contains the implementation of the project Advertising and Social Influence for the 2021/2022 edition of the Online Learning Applications course at Politecnico di Milano.

## Usage

In the simulations folders there are 6 runnable files, one for each required step. They all call [SimulationHandler](Project/simulations/SimulationHandler.py), that has a number of settable parameters. The most notable one is ```plot_regressor_progress```, which accepts as parameter the name of one learner to dynamically print the learnt curve. e.g. ```plot_regressor_progress=BanditNames.GPTS_Learner.name```. See [learners](Project/learners) for the other learners.

## Installation
Clone and install: 
```sh
git clone https://github.com/gccianmario/Online-learning-application-projects/
pip install -r requirements.txt
```

## Requirements
* scikit-learn>=1.0.2
* seaborn>=0.10.1
* numpy>=1.21.5
* pandas>=1.4.2
* scipy>=1.7.3
* matplotlib>=3.5.1
* tqdm==4.64.0
* progressbar2==3.37.1 
