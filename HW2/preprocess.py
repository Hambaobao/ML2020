import pandas as pd

train_file_path = "./data/train.csv"
test_file_path = "./data/test_no_label.csv"


def preprocess(file_path):
    raw_data = pd.read_csv(file_path, encoding='big5')

    num_rows = raw_data.shape[0]
    num_cols = raw_data.shape[1]

    # drop_col_list = [num_cols - 1, ]
    drop_col_list = []
    for c in range(0, num_cols):
        print("Handling column ", str(c))
        if raw_data.iloc[:, c].dtype == 'int64':
            continue
        else:
            kind_list = []
            drop_col_list.append(c)
            for r in range(0, num_rows):
                # print(str(r) + "," + str(c), ": ", raw_data.iloc[r, c])
                # continue
                if raw_data.iloc[r, c] in kind_list:
                    continue
                else:
                    kind_list.append(raw_data.iloc[r, c])
                    raw_data[raw_data.iloc[r, c]] = 0
                    # print(str(r) + "," + str(c), "\tAdd", raw_data.iloc[r, c], "into kind list")

    drop_col_list.sort()
    back_count = len(drop_col_list)
    for c in drop_col_list:
        back_count = back_count - 1
        print("Remaining: " + str(back_count))
        for r in range(0, num_rows):
            label = raw_data.iloc[r, c]
            raw_data.loc[r, label] = 1
            # print("Remaining: " + str(back_count), str(r) + "," + str(c), label)

    drop_col_list.append(0)
    raw_data.drop(raw_data.columns[drop_col_list], axis=1, inplace=True)
    raw_data.drop([' ?', ' Yes', ' No'], axis=1, inplace=True)
    if 'y' in raw_data.columns.values.tolist():
        raw_data.drop(['y', ], axis=1, inplace=True)

    return raw_data


if __name__ == '__main__':
    train_data = preprocess(train_file_path)
    test_data = preprocess(test_file_path)

    train_data_col_values = train_data.columns.values.tolist()
    test_data_col_values = test_data.columns.values.tolist()

    insert_values = set(train_data_col_values) & set(test_data_col_values)
    not_in_train_col_values = list(set(train_data_col_values) - insert_values)
    not_in_test_col_values = list(set(test_data_col_values) - insert_values)

    # print("Not in train column values")
    # print(not_in_train_col_values)
    # print("Not in test column values")
    # print(not_in_test_col_values)

    not_in_train_col_values.remove(' 50000+.')
    not_in_train_col_values.remove(' - 50000.')

    train_data.drop(not_in_train_col_values, axis=1, inplace=True)
    test_data.drop(not_in_test_col_values, axis=1, inplace=True)

    train_data.rename(columns={' 50000+.': '50000+'})
    train_data.rename(columns={' - 50000.': '50000-'})

    train_data.sort_index(axis=1)
    test_data.sort_index(axis=1)

    train_data.to_csv("./train_data.csv", index=False, sep=',')
    test_data.to_csv("./test_data.csv", index=False, sep=',')
