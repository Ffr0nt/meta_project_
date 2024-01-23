from pathlib import Path
import json
from collections.abc import Callable


def default_true_predicate(dict):
    """
    Default predicate that allways return true
    :param dict: dict
        No matter what that is, return always true
    :return: bool
        true
    """
    return True


def generator_over_review(
        path_folder: str,
        review_predicate: Callable[[dict], bool] = default_true_predicate,
        game_search_list: list[str] = [],
        platform_search_list: list[str] = [],
        add_info: bool = False
):
    """
    Returns a generator, that iterates over game reviews.
    =====================================================
    :param path_folder: str
        Path to the outer folder (critic_reviews)

    :param review_predicate: Callable[[dict],bool]
        Predicate that terminate what review you need to iterate over.
        Default option returns True value always.

    :param game_search_list:
        List of games that you need to iterate over.
        Default option - []. Iterates over every game.

    :param platform_search_list:
        List of platforms that you need to iterate over.
        Default option - []. Iterates over every platform.

    :param add_info:
        Terminates if only text of review is returned (if false),
        or additional info is required (if true). In that case, will return
        (   text_of_review ,
            {"belongings":
            (name, platform, review_name),
             "path": absolute_path}
        )
    :return:
        Generator
    """
    names = Path(path_folder).glob('*')
    for n in names:
        if not (not game_search_list or n.name in game_search_list):
            continue

        platforms = Path(n).glob('*')

        for p in platforms:
            if not (not platform_search_list or p.name in platform_search_list):
                continue

            reviews = Path(p).glob('*')
            for rev in reviews:

                with open(rev) as f:
                    templates = json.load(f)
                    if review_predicate(templates):
                        if add_info:
                            yield templates, {"belongings": (n.name, p.name, rev.name),
                                              "path": str(rev)}
                        else:
                            yield templates
