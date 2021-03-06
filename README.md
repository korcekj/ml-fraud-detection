# ML Fraud detection

Fraud detection system built using Machine Learning

## Installation

### Docker

#### 1. Define image names and environment variables

```yml
    version: "3.9"
    services:
      cli:
        image: korcekj/ml-fraud-detection:<tag>
        depends_on:
          - api
        environment:
          - MS_DISTANCE_URL=<msdistance url>
        volumes:
          - <data dir path>:/home/cli_user/<dir>
          - <visuals dir path>:/home/cli_user/<dir>
          - <models dir path>:/home/cli_user/<dir>
        stdin_open: true
        tty: true
      api:
        image: korcekj/ms-distance:<tag>
        ports:
          - "8080:8080"
        environment:
          - MONGO_URI=<mongodb uri>
          - MONGO_DB=<mongodb name>
          - DM_API_KEY=<distancematrix api key>
          - GC_API_KEY=<geocode api key>
```

#### 2. Compose container

```bash
  $ docker compose up -d
  $ docker exec -it <container-name-or-id> bash
```

## Usage

```bash
  $ python main.py --help
```

### Dataset cleanup

Clean the dataset from unwanted columns and empty cells **:param data_import:**
path to input file **:param data_export:** path to output file **:param target:**
column name **:param rows:** number of rows to pick up **:param columns:** column names to be removed

```bash
  $ python main.py data-cleanup -di <dataset_in> -de <dataset_out> -t <column_name>
```

| Parameter               | Type      | Description                |
|:---------------------|:----------| :------------------------- |
| `-di, --data-import` | `PATH`    | Data file path for import  [required] |
| `-de, --data-export` | `PATH`    | Data file path for export  [required] |
| `-t, --target` | `TEXT`    | Name of target column  [required] |
| `-r, --rows` | `INTEGER` | Number of rows to be processed |
| `-c, --columns` | `TEXT`    | Columns to be removed |
| `--help` |           | Show this message and exit |

### Fraud detection (Microservices)

Detect fraud transactions using microservices **:param data_import:** path to input file **:param data_export:** path to
output file **:param target:** column name **:param rows:** number of rows to pick up

```bash
  $ python main.py microservices -di <dataset_in> -de <dataset_out> -t <column_name>
```

| Parameter               | Type      | Description                |
|:---------------------|:----------| :------------------------- |
| `-di, --data-import` | `PATH`    | Data file path for import  [required] |
| `-de, --data-export` | `PATH`    | Data file path for export  [required] |
| `-t, --target` | `TEXT`    | Name of target column  [required] |
| `-r, --rows` | `INTEGER` | Number of rows to be processed |
| `--help` |           | Show this message and exit |

### Fraud detection (Neural Network)

Detect fraud transactions using neural network **:param train_data:** path to training data **:param test_data:** path
to testing data **:param module_import:**
path to module for import **:param module_export:** path to module for export
**:param visuals_export:** path to visuals folder **:param valid_split:** ratio of "valid" data **:param batch_size:**
size of the batch **:param learning_rate:**
learning rate **:param epochs:** number of epochs **:param target:** column name
**:param visuals:** boolean

```bash
  $ python main.py neural-network -tnd <dataset_train> -tts <dataset_test> -t <column_name>
```

| Parameter               | Type      | Description                |
|:---------------------|:----------| :------------------------- |
| `-tnd, --train-data` | `PATH`    | Training data file path  [required] |
| `-ttd, --test-data` | `PATH`    | Testing data file path  [required] |
| `-mi, --module-import` | `PATH` | Module file path for import |
| `-me, --module-export` | `PATH` | Module folder path for export |
| `-ve, --visuals-export` | `PATH` | Visualizations dir path for export |
| `-vs, --valid-split` | `FLOAT RANGE` | Validation split  [0<=x<=1] |
| `-bs, --batch-size` | `INTEGER RANGE` | Batch size  [1<=x<=32768] |
| `-lr, --learning-rate` | `FLOAT RANGE` | Learning rate  [0<=x<=1] |
| `-e, --epochs` | `INTEGER RANGE` | Number of epochs  [1<=x<=10000] |
| `-t, --target` | `TEXT`    | Name of target column  [required] |
| `-v, --visuals` |  | Show visuals |
| `--help` |           | Show this message and exit |

### Fraud detection (Decision Tree)

Detect fraud transactions using decision tree **:param train_data:** path to training data **:param test_data:** path to
testing data **:param module_import:**
path to module for import **:param module_export:** path to module for export
**:param visuals_export:** path to visuals folder **:param max_depth:** maximum depth of the tree **:param
min_samples_split:**
minimum number of samples to split a node **:param min_samples_leaf:** minimum number of samples at a leaf node **:param
criterion:** split quality function **:param target:** column name
**:param visuals:** boolean

```bash
  $ python main.py decision-tree -tnd <dataset_train> -tts <dataset_test> -t <column_name>
```

| Parameter               | Type                           | Description                               |
|:---------------------|:-------------------------------|:------------------------------------------|
| `-tnd, --train-data` | `PATH`                         | Training data file path  [required]       |
| `-ttd, --test-data` | `PATH`                         | Testing data file path  [required]        |
| `-mi, --module-import` | `PATH`                         | Module file path for import               |
| `-me, --module-export` | `PATH`                         | Module folder path for export             |
| `-ve, --visuals-export` | `PATH`                         | Visualizations dir path for export        |
| `-md, --max-depth` | `INTEGER`                      | Maximum depth of the tree                 |
| `-ms, --min-samples-split` | `INTEGER`                      | Minimum number of samples to split a node |
| `-ml, --min-samples-leaf` | `INTEGER`                      | Minimum number of samples at a leaf node  |
| `-c, --criterion` | <code>gini&vert;entropy</code> | Quality function |
| `-t, --target` | `TEXT`                         | Name of target column  [required]         |
| `-v, --visuals` |                                | Show visuals                              |
| `--help` |                                | Show this message and exit                |

### Fraud detection (Random Forest)

Detect fraud transactions using random forest **:param train_data:** path to training data **:param test_data:** path to
testing data **:param module_import:**
path to module for import **:param module_export:** path to module for export
**:param visuals_export:** path to visuals folder **:param max_depth:** maximum depth of the tree **:param
min_samples_split:** minimum number of samples to split a node **:param min_samples_leaf:** minimum number of samples at
a leaf node **:param n_estimators:** number of trees **:param criterion:** split quality function **:param target:**
column name **:param visuals:** boolean

```bash
  $ python main.py random-forest -tnd <dataset_train> -tts <dataset_test> -t <column_name>
```

| Parameter                  | Type                           | Description                               |
|:---------------------------|:-------------------------------|:------------------------------------------|
| `-tnd, --train-data`       | `PATH`                         | Training data file path  [required]       |
| `-ttd, --test-data`        | `PATH`                         | Testing data file path  [required]        |
| `-mi, --module-import`     | `PATH`                         | Module file path for import               |
| `-me, --module-export`     | `PATH`                         | Module folder path for export             |
| `-ve, --visuals-export`    | `PATH`                         | Visualizations dir path for export        |
| `-md, --max-depth`         | `INTEGER`                      | Maximum depth of the tree                 |
| `-ms, --min-samples-split` | `INTEGER`                      | Minimum number of samples to split a node |
| `-ml, --min-samples-leaf`  | `INTEGER`                      | Minimum number of samples at a leaf node  |
| `-ne, --n-estimators`      | `INTEGER`                               | Number of trees |
| `-c, --criterion`          | <code>gini&vert;entropy</code> | Quality function |
| `-t, --target`             | `TEXT`                         | Name of target column  [required]         |
| `-v, --visuals`            |                                | Show visuals                              |
| `--help`                   |                                | Show this message and exit                |