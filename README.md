### Environment

```bash
git clone {Repository Url}
```

```bash
python3 -m venv ./AiSolutionLabMinitask
source ./AiSolutionLabMinitask/bin/activate
```

```bash
(AiSolutionLabMinitask) cd {Repository Name}
(AiSolutionLabMinitask) make install
(AiSolutionLabMinitask) make test
```

### Data

- `pred_{k}.txt`는 예측 polygon 예측 문장을, `gt_{k}.txt`는 정답 polygon 과 정답 문장을 저장합니다.
- 각 파일은 다음과 같은 형태로 생겼습니다.

```
119,800,132,804,120,1032,108,1028,CARTOLER
102,1150,122,1159,120,1193,100,1187,###
416,1254,433,1255,433,1261,416,1262,###
237,1117,242,1117,241,1171,236,1171,###
227,1119,235,1118,235,1151,226,1150,T
377,1213,402,1210,402,1223,377,1226,###
154,1073,174,1074,174,1096,151,1094,###
1215,347,1259,295,1266,366,1221,423,12
33,1109,70,1138,67,1175,28,1161,###
556,600,607,607,607,873,558,862,PISTONE
481,1148,504,1115,504,1145,482,1174,###
57,1283,67,1283,67,1293,57,1292,###
58,1247,70,1246,70,1262,57,1262,###
```

- `./data/preprocessed` 에 `pred_{k(int>=0)}.txt`, `gt_{k(int>=0)}.txt` 형태로 데이터를 업로드합니다.
- 예를 들면 다음과 같습니다.

```
.
├── README.md
├── data
│   ├── preprocessed
│   │   ├── gt_0.txt
│   │   ├── gt_1.txt
│   │   ├── gt_2.txt
│   │   ├── ...
│   │   ├── pred_0.txt
│   │   ├── pred_1.txt
│   │   ├── pred_2.txt
│   │   └── ...
│   └── ...
├── src
│   └── ...
├── Makefile
├── requirements.txt
├── main.py
└── ... 
```

NOTE
- `pred_{k}.txt` 파일은 `gt_{k}.txt` 파일과 같은 이름을 가진 `pred_{k}.txt` 파일이 존재해야 합니다.
- 파일명 속 `{k(int>=0)}`은 0부터 가장 큰 정수까지 순차적으로 증가해야 합니다.

### Start

```bash
python3 main.py
```