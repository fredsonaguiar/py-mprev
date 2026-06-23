import os


def list_documents(path:str):
    documents = []

    filenames = os.listdir(path)

    for filename in filenames:
        filename = os.path.join(path, filename)
        if os.path.isfile(filename):
            documents.append(filename)
        elif os.path.isdir(filename):
            documents += list_documents(filename)

    return documents


