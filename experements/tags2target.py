from torch import tensor

target_dict_meta = {
    "Action": 1,
    "Adventure": 2,
    "General": 3,
    "Shooter": 4,
    "Shoot-'Em-Up": 4,
    "Role-Playing": 5,
    "RPG": 5,
    "Strategy": 6,
    "Tactics": 6,
    "Tactical": 6,
    "Arcade": 7,
    "Platformer": 8,
    "Miscellaneous": 9,
    "Puzzle": 10,
    "Sports": 11,
    "Soccer": 11,
    "Basketball": 11,
    "Football": 11,
    "Golf": 11,
    "Boxing": 11,
    "Tennis": 11,
    "Motorcycle": 11,
    "Hockey": 11,
    "Motocross": 11,
    "Skateboarding": 11,
    "Snowboarding": 11,
    "Biking": 11,
    "Skate": 11,
    "Skateboard": 11,
    "Simulation": 12,
    "Sim": 12,
    "Driving": 13,
    "Racing": 13,
    "Automobile": 13,
    "GT": 13,
    "Car": 13,
    "Fighting": 14,
    "Beat-'Em-Up": 14
}


def tags2target(tags, meta=True):
    """
    Convert list of tags to the target

    :param meta: bool
    flag to show if tags belong to meta-critics or steam

    :param tags: list of strings
    list that contains names of tags from meta-critics or steam

    :return: torch.Tensor of float
    vector of 0.,1., where 0. - absence of feacher, 1. - presence
    """
    if meta:
        answer = tensor([0.] * 14)
        for t in tags:
            if t in target_dict_meta.keys():
                answer[target_dict_meta[t]] = 1.
        return answer
    else:
        answer = tensor([0.] * 14)
        return answer
