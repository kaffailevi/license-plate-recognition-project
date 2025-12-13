# License Plate Recognition (PyTorch)

Egyszerű PyTorch projekt rendszámtábla felismeréshez az
[andrewmvd/car-plate-detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
adatkészlet felhasználásával.

## Követelmények

- Python 3.12
- PyTorch és Torchvision (lásd `requirements.txt`)

Telepítés:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Adatletöltés

A Kaggle CLI használatával töltsd le és csomagold ki az adatokat:

```bash
mkdir -p data
kaggle datasets download -d andrewmvd/car-plate-detection -p data
unzip data/car-plate-detection.zip -d data/car-plate-detection
```

Az elérési út így néz ki: `data/car-plate-detection/images` és
`data/car-plate-detection/annotations`.

## Tanítás

```bash
python train.py --data-dir data/car-plate-detection --epochs 5 --batch-size 2
```

A kimeneti súlyok alapértelmezés szerint a `checkpoints/model.pt` fájlba kerülnek.

## Inferencia (előrejelzés)

```bash
python predict.py --weights checkpoints/model.pt --image path/to/sample.jpg --save-path outputs/prediction.jpg
```

A futás kiírja a megtartott dobozokat és opcionálisan elmenti a vizualizációt.
