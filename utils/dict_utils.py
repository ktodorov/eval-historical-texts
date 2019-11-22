def update_dictionaries(dict1: dict, dict2: dict):
    for key, value in dict2.items():
        if key not in dict1.keys():
            dict1[key] = []

        dict1[key].extend(value)
