# :fire: Fantasy Name Generation With Power Of RNNs :fire:
**Text generation with character level tokenization using [pytorch-lightning](https://www.pytorchlightning.ai/) ⚡**

![alt text](https://cutewallpaper.org/21/frozen-throne-wallpaper/warcraft-3-frozen-throne-wallpaper-Google-Search-in-2019-.jpg)

## Contents
* [Setup](#setup)
* [Pretrained Model Usage](#pretrained-model-usage)
* [Training From Scratch](#training)
    - [Using Config YAMLs](#using-config-yamls)
    - [With Custom Data](#with-custom-data)⚡

## Setup
Get Repository
```script
git clone https://github.com/borhanMorphy/fantasy-name-generator.git
cd fantasy-name-generator
```

Install the dependencies

```script
pip install -r requirements.txt
```

## Pretrained Model Usage

```script
```

## Training

### Using Config YAMLs

Training **NaiveGRU** model, using baseline_gru.yaml configs, **wow names** and **dota names** with batch size of 64 and auto learning rate finder

```script
python train.py -y configs/baseline_gru.yaml -bs 64 -ds wowdb dotadb lotrdb --auto-lr
```

### With Custom Data