import pandas as pd


def load_data():
    train_data = pd.read_csv('data/fraudTrain.csv')
    test_data = pd.read_csv('data/fraudTest.csv')
    return train_data, test_data


def main():
    train_data, test_data = load_data()


if __name__ == '__main__':
    main()
