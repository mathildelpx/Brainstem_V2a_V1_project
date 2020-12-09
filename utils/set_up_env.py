import os


def set_up_env_fish(output_path, fishlabel):
    for folder_name in ['dataset', 'np_array', 'csv_files', 'fig']:
        try:
            os.mkdir(output_path + folder_name + '/')
        except FileExistsError:
            if folder_name == 'dataset':
                print('IDE already set up')
            else:
                pass
        else:
            print('First IDE set up')
        try:
            os.mkdir(output_path + folder_name + '/' + fishlabel)
        except FileExistsError:
            if folder_name == 'dataset':
                print('Fish already analyzed')
            else:
                pass


def set_up_env_plane(output_path, fishlabel, depth):
    for folder_name in ['dataset', 'np_array', 'csv_files', 'fig']:
        try:
            os.mkdir(output_path + folder_name + '/' + fishlabel + '/' + depth)
        except FileExistsError:
            pass