import json
import os
from typing import Dict, List, Tuple

from torch.utils.data import DataLoader, Dataset


DATASET_PATH = '/content/drive/MyDrive/Study/3_курс/Проект 3 курс/dataset_indexed'

COMMENT_PATH = DATASET_PATH + '/comment_directory_indexed'
DESCRIPTION_PATH = DATASET_PATH + '/description_directory_indexed'
REVIEW_PATH = DATASET_PATH + '/review_directory_indexed'
GAMENAME_IDX_PATH = '/content/drive/MyDrive/Study/3_курс/Проект 3 курс/gamename_to_idx.json'


class GameGenerationDataset(Dataset):
    def __init__(
        self,
        path_to_comment: str,
        path_to_description: str,
        path_to_review: str,
        path_to_gamename_to_idx: str
    ) -> None:

    # Read 'idx - game_name' json file for iterating on game folders

        with open(path_to_gamename_to_idx, 'r') as idx_game_file:
            json_data_dict: Dict[str, str] = json.load(idx_game_file)
            self.idx_game_dict: Dict[int, str] = {int(key): value for key, value in json_data_dict.items()}

        self.path_to_comment: str = path_to_comment
        self.path_to_description: str = path_to_description
        self.path_to_review: str = path_to_review

        self.path_verbose_name: Dict[str, str] = {
            self.path_to_description: 'description',
            self.path_to_comment: 'comment',
            self.path_to_review: 'review'
        }

    def __getitem__(self, index: int) -> Tuple[str, Dict[str, List[Dict[str, str]]]]:
        game_verbose_name: str | None = self.idx_game_dict.get(index)

        if game_verbose_name is None:
            raise IndexError('index out of range')

        reivew_description_comment: Dict[str, List[Dict[str, str]]] = {value: [] for key, value in self.path_verbose_name.items()}

        for path, verbose_name in self.path_verbose_name.items():
            game_path: str = f'{path}/{index}'
            for root, dirs, json_files in os.walk(game_path):
                for idx, json_file in enumerate(json_files):
                    with open(f'{game_path}/{json_file}', 'r') as f:
                        json_data = json.load(f)
                        reivew_description_comment.get(verbose_name).append(json_data)


        return game_verbose_name, reivew_description_comment


    def __len__(self) -> int:
        return len(self.idx_game_dict)


dataset = GameGenerationDataset(
    COMMENT_PATH,
    DESCRIPTION_PATH,
    REVIEW_PATH,
    GAMENAME_IDX_PATH
)

# dataset[10] result:
# ('.hack//Mutation Part 2',
#  {'description': [{'name': '10',
#     'URL': 'https://www.metacritic.com/game/playstation-2/hackmutation-part-2/details',
#     'meta_score': '76',
#     'meta_count': '24',
#     'user_score': '8.5',
#     'user_count': '83',
#     'data': 'May  6, 2003',
#     'platforms': ['PlayStation 2'],
#     'rating': 'T',
#     'official_site': None,
#     'developer': 'CyberConnect2',
#     'genre': ['Role-Playing,', 'Action', 'RPG'],
#     'text': 'In part 2 of this episodic, 4-part double-disc masterpiece, your remarkable party members meet up with the characters from .hack//SIGN television series, to support you through your painstaking journey to the truth. What is the developer of ~The World~ hiding? Kite continues his quest, armed with the forbidden abilities of Data Drain and Gate Hacking. Will he be able to save Orca and the other coma victims...? [Bandai]'}],
#   'comment': [],
#   'review': []})