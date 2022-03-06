# ML Fraud detection

Fraud detection system built using Machine Learning

## Installation

### Manual

```bash
  $ git clone https://github.com/korcekj/ml-fraud-detection.git
  $ cd ml-fraud-detection
  $ python setup.py install
```

## Usage

```bash
  $ ml-fraud-detection --help
```

### Dataset cleanup

Clean the dataset from unwanted columns and empty cells **:param data_import:**
path to input file **:param data_export:** path to output file **:param target:**
column name **:param rows:** number of rows to pick up **:param columns:** column names to be removed

```bash
  $ ml-fraud-detection cd -di <dataset_in> -de <dataset_out> -t <column_name>
```

| Parameter               | Type      | Description                |
|:---------------------|:----------| :------------------------- |
| `-di, --data-import` | `PATH`    | Data file path for import  [required] |
| `-de, --data-export` | `PATH`    | Data file path for export  [required] |
| `-t, --target` | `TEXT`    | Name of target column  [required] |
| `-r, --rows` | `INTEGER` | Number of rows to be processed |
| `-c, --columns` | `TEXT`    | Columns to be removed |
| `--help` |           | Show this message and exit |

### Fraud detection (Microservice)

Detect fraud transactions using microservice **:param data_import:** path to input file **:param data_export:** path to
output file **:param target:** column name **:param rows:** number of rows to pick up

```bash
  $ ml-fraud-detection fd -di <dataset_in> -de <dataset_out> -t <column_name>
```

| Parameter               | Type      | Description                |
|:---------------------|:----------| :------------------------- |
| `-di, --data-import` | `PATH`    | Data file path for import  [required] |
| `-de, --data-export` | `PATH`    | Data file path for export  [required] |
| `-t, --target` | `TEXT`    | Name of target column  [required] |
| `-r, --rows` | `INTEGER` | Number of rows to be processed |
| `--help` |           | Show this message and exit |

### Neural Network

Detect fraud transactions using neural network **:param train_data:** path to training data **:param test_data:** path to
testing data **:param module_import:**
path to module for import **:param module_export:** path to module for export
**:param visuals_export:** path to verbose directory **:param valid_split:** ratio of "valid" data **:param batch_size:** size of
the batch **:param learning_rate:**
learning rate **:param epochs:** number of epochs **:param target:** column name
**:param visuals:** boolean

```bash
  $ ml-fraud-detection nn -tnd <dataset_train> -tts <dataset_test> -t <column_name>
```

| Parameter               | Type      | Description                |
|:---------------------|:----------| :------------------------- |
| `-tnd, --train-data` | `PATH`    | Training data file path  [required] |
| `-ttd, --test-data` | `PATH`    | Testing data file path  [required] |
| `-mi, --module-import` | `PATH` | Module file path for import |
| `-me, --module-export` | `PATH` | Module file path for export |
| `-ve, --visuals-export` | `PATH` | Visualizations dir path for export |
| `-vs, --valid-split` | `FLOAT RANGE` | Validation split  [0<=x<=1] |
| `-bs, --batch-size` | `INTEGER RANGE` | Batch size  [1<=x<=32768] |
| `-lr, --learning-rate` | `FLOAT RANGE` | Learning rate  [0<=x<=1] |
| `-e, --epochs` | `INTEGER RANGE` | Batch size  [1<=x<=10000] |
| `-t, --target` | `TEXT`    | Name of target column  [required] |
| `-v, --visuals` |  | Name of target column  [required] |
| `--help` |           | Show this message and exit |