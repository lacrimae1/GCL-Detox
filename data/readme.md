# Data

The processed datasets are **not** versioned in this repository — only this README is tracked.

Download them from Google Drive:

<https://drive.google.com/drive/folders/1Xxd2-T5Ax3KNF9E3ywRV4m4pKDAHIiWb?usp=sharing>

Expected layout after downloading:

```
data/
└── clean/
    ├── ml-10M/   (~306 MB)
    ├── douban/   (~20 MB)
    └── Yelp/     (~40 MB)
```

Each dataset folder contains the split files (`train.txt`, `val.txt`, `test.txt`), `sample.pickle`, and, for ML-10M, the raw `ratings.dat`.
