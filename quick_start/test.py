from model.StringMatching import StringMatching


def run(data_name, data_path, model):
    """
        :param data_name: the name of the dataset
        :param data_path: the local path of the dataset
        :param model: the name of the model

        :return:
    """
    if model == "StringMatching":
        sm = StringMatching(data_name, data_path)
        sm.run()


if __name__ == '__main__':
    run("mesa", "../dataset/mesa_dataset/", "StringMatching")
