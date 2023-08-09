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

# dataset[15] result: ('007 Racing',
# {'description': [
# {
# 'name': '15', '
# URL':
# 'https://www.metacritic.com/game/playstation/007-racing/details',
# 'meta_score': '51',
# 'meta_count': '16',
# 'user_score': '5.0',
# 'user_count': '21',
# 'data': 'Nov 20, 2000',
# 'platforms': ['PlayStation'],
# 'rating': 'T',
# 'official_site': None,
# 'developer': 'Eutechnyx',
# 'genre': ['Driving,', 'Mission-based'],
# 'text': "Pay attention, 007. This is for your eyes only. Q-Branch has re-engineered some of your favorite automobiles
# with additional gadgets and firepower. You'll need everything they've got, plus your own uncanny ability to beat the
# odds. The free world is depending on you. Oh, and one other thing before you go, 007 - try to bring them back in
# one piece."
# }],
#
# 'comment': [
# {'text of comment':
# 'The concept itself may have been exciting at the time, and thats the only good
# thing that I can say about this game. The controls leave a lot to be desired. No effort was put into the story and
# writing',
# 'score': '2',
# 'date': 'Jul 23, 2016',
# 'game_name': '15'},
# {'text of comment':
# 'I like this game because
# it has interesting and cool missions, and because it was my childhood gameThe things I dont like is the shortness
# of the game and this is the only bad thing in my opinion',
# 'score': '7',
# 'date': 'Sep 29, 2021',
# 'game_name': '15'},
# {'text of comment':
# "After Nintendo sold off the rights to make games about James Bond, EA sought to
# capitalize on the success of the fantastic Goldeneye 007, and made a series of Bond games that weren't so bad,
# except for this turd of a game. This is my personal choice for worst PSone game. The game is an unbelievably
# horrible mess. The gameplay is atrocious thanks to the slippery controls and the fact After Nintendo sold off the
# rights to make games about James Bond, EA sought to capitalize on the success of the fantastic Goldeneye 007,
# and made a series of Bond games that weren't so bad, except for this turd of a game. This is my personal choice for
# worst PSone game. The game is an unbelievably horrible mess. The gameplay is atrocious thanks to the slippery
# controls and the fact your sports cars are slow. Your car will constantly swerve out of control randomly and you
# will have extreme difficulty trying to properly use a weapon. The storyline is horrible as well and the dialogue is
# terrible. This game features some of the most tedious and frustrating missions ever put into a game. Avoid at all
# costs, because this ride just crashes and burns.",
# 'score': '1',
# 'date': 'Jun  7, 2009',
# 'game_name': '15'},
# {'text of comment':
# 'Atmosphere: 6Enjoyment: 10Gameplay: 9Graphics: 7Music: 7Story: 8Total: 7.83',
# 'score': '8',
# 'date': 'Dec 28, 2021',
# 'game_name': '15'}],
#
# 'review':
# [{
# 'name': '15',
# 'text': "Electronic Arts' latest James Bond
# licensed game, 007 Racing, is one of those games in which you try desperately to have fun despite the game getting
# in your way. The strangest thing about 007 Racing is that it is essentially a slightly more complicated 3D version
# of the classic old coin-op Spy Hunter. And Spy Hunter is a knock-off from movies and TV shows like 007 and Mission
# Impossible. So we've come full circle. Perhaps even more serendipitous is that the 20-something-old Spy Hunter is
# in many ways better than 007 Racing.   EA's 007 Racing gives gamers the chance to play as British agent James Bond,
# and more specifically to drive his cars. Players get to handle Bond's many stealthy, gadget-laden cars,
# shooting enemies down with bullets and missiles, utilizing smoke screens, oil slicks, and the like to knock them
# out of commission, and foisting his enemies as best as the stiff-@ss Brits know how. There are hints of Spy Hunter
# and even Twisted Metal, but nothing here ever bests those games. Perhaps if the game looked -- and played -- like a
# fifth-generation PlayStation game, instead of a first-generation game, it would be more likeable. But that's not
# the case.   Gameplay EA's  007 Racing is probably one of the best game ideas ever. It's also one of the best ideas
# for a game that's been horrendously hacked to mincemeat. Everyone I know wants to drive a bitchin', expensive,
# highly modified weapon-of-a-car. It's a fundamental need, like eating, sleeping, and having sex. Well...it comes
# right after those, anyway. But the game provides immediate satisfaction (and an extra amount of dissatisfaction)
# because players jump right into several of Bond's cars -- Aston Martin DB 5, BMW Z3 Roadster, and a handful of
# others.   Taking a nod from the massive library of 19 Bond films, MI6's ~Q Branch~ dishes up a good assortment of
# weaponry and gadgets. The laundry list include machineguns, surface-to-air missiles, spike and mine dispensers,
# rocket launchers, smoke screens, an oil slick generator, tire shredders and bulletproof windows. Cars take physical
# -- visual -- damage, and they are based on a four-point model, which means that there are four collision points on
# the car, enabling it to take to the air better, spin, twist, flip, etc. The missions take place in locales such as
# Eastern Europe, Amsterdam, South America, Mexico, Monte Carlo, Louisiana, and New York.   What 007 Racing boils
# down to is simple arcade racing action. You get the chance to thwart enemies by altering their plans to destroy the
# world, via a number of objective-based missions. You get your objectives, and you must meet them to beat the level,
# which is based on a scoring system. To be fair, 007 Racing has a wide variety of mission levels. One of the best
# levels in the game is ~Escape,~ which is a straight-out racing level in which you have to beat out Koskov's
# hot-looking assassin through the jungles of Mexico. You need speed and a decent knowledge of the course to beat it,
# and of course it's highly reminiscent of Spy Hunter. I personally would have been happy to see a lot more levels
# like this, but it wasn't in the cards.   One of the levels that should have been a great level, because of all of
# the objectives, is Air Strike. Do you know how many times I played that frickin' level before I actually understood
# what to do? Call me a dumb-@ss if you will (and I there are those of you who exist), but detonating those mines
# isn't what I call intuitive. Also, what's up with losing a little bit if health each time blow up a mine and that
# hightail it out of there? Even when I perfectly dropped set the bomb off, and then left on time, I still lost
# health. And then I lose health by breaking up tents and then gain the same amount back? Ughhh! Levels like this and
# Ambush are potentially great levels, but they're badly balanced and clumsily constructed. Highway Hazard was kind
# of cool, too -- you get the chance to pierce each of the tires of an 18-wheeler with a high-tech laser,
# and once you've done it you win.It reminds me of Speed Racer and Spy Hunter, and yet it's not as fun to play or
# watch.   Yeah, the level design isn't only unbalanced, it's awkward. While it seems like a good idea to have a
# minimum score to beat a level in the final analysis, to beat some of these levels and then miss the minimum score
# was an exercise in frustration.   Strangely enough, while I have qualms with the level design and the game's
# overall imbalance, the actual driving isn't bad. The engine sounds and looks of each car were pretty spot-on. The
# brakes and acceleration are solid, and the e-brake makes for some killer power-slides. Using the e-brake is a lot
# of fun. And despite an eye-piercingly painful framerate, the sense of speed wasn't too bad in some situations. It's
# almost impossible to have a bad framerate and a good sense of speed, but at least in the Escape level,
# the speed level was high. Strange, that.   Graphics After five years of the PlayStation growing and growing as a
# system and developers pushing its limits, it's remarkable that any company could create a horrible set of graphics
# such as these. I mean, this game wouldn't make a top 500 best-looking games list. Everything is haywire here:
# pop-up, really mind-inducing perspective correction problems, terrible (terrible!) collision detection, and murky,
# grainy, low resolution visuals. Seams pop-up everywhere and the particle system -- if you want to call it that --
# is straight from the 16-bit years. Ouch.   The characters are small and indistinct, and the collision detection is
# total rubbish. I mean, sure, if I wasn't supposed to hit a scientist that's fine. But when I shoot them the bullets
# go right through. If I run them down, they turn invisible. That's just one of many examples.   Still, if this game
# were simply a great one based solely on straightforward gameplay, I would say so. But it's not. Neither the
# graphics nor the gameplay are that exceptional and most kids who have any sense about them will look at this game
# and wonder what year this was made. (~This was made in 2000? No way!~)   Sound The weird thing about the sound is
# that parts of sound are good, and some of them are simply really hard to hear. I had to turn the sound almost
# entirely off to hear the characters speaking during several levels. There appears to have been some interesting
# things going on in the production of this game, and it didn't sounding like Gerswin. Thankfully, Eutechnyx edited
# out a lot of the stock 007 sounds that hammered gamers ears in the alpha version of this game. The repetitive
# da-daaaa-duuhhh! clips are still too numerous. Q and his helper, done by John Cleese are perfect. But Bond is as
# weak as he was in The World Is Not Enough. M sounds good, though, and numerous other voice actors do a decent
# job.EA's 007 Racing is a decent little game, as long as you don't expect too much from it. As you might have
# suspected, 007 Racing ain't the Sean Connery of Bond games, it's the Timothy Dalton version. It's not original,
# nor is it good looking. It's filled with awkward spots and questionable areas (like when I reached the broken
# bridge in Escape and the vocals chimed in after it was too late to launch my parachute), and it becomes a chore
# rather than fun. Occasionally, there are little flashes of goodness (Escape and Gimme a Break are examples),
# but the game never really reaches any new planes of play that weve did already experience in Spy Hunter,
# back in the early 1980s. I mean if you're simply dying to drive Bond cars, rent this game, but don't buy it full
# price. Now, if you don't mind, I've got an old-school arcade to find.",
# 'date': 'Nov 22, 2000',
# 'grade': None},
# {'name': '15',
# 'ref': 'https://www.gamespot.com/reviews/007-racing-review/1900-2657324/',
# 'text': "Over the years,
# the James Bond movies have provided the movie-going public with a glimpse into the life of a playboy superspy - the
# gadgets, the espionage, the fast women, and the faster cars. While the Bond movies have been turned into games
# countless times, none of the games have provided the focus of 007 Racing. EA's latest Bond game centers on 007's
# vehicular endeavors, giving you a heavily armed and armored car and a laundry list of objectives. While the game
# sounds like an interesting and excellent idea, in execution, it struggles in too many areas to warrant a
# recommendation. The game lets you drive various Bond cars that have appeared throughout his movie exploits,
# such as the Aston Martin DB5, BMW 750, BMW Z8, Lotus Esprit, and BMW Z3. Each car handles slightly differently,
# though none of them handle particularly well. Each car can be outfitted with different items, such as rockets,
# missiles, machine guns, shields, smoke screens, and oil slicks. Only one item can be equipped at a time,
# and you automatically switch to an item when you pick it up. This leads to problems when you're trying to line up a
# rocket shot and accidentally run over a new weapon - you're forced to cycle through the weapons one at a time,
# wasting precious seconds. The weapon troubles don't stop there. Your rockets are side-mounted, alternating fire
# from the left and right sides of your car. If you don't keep track of exactly which side of your car is the next to
# fire, lining up even midrange rocket shots is needlessly difficult. A set of crosshairs would have been a big help
# here. Add the aiming difficulties to the sluggish nature of the game's various vehicles, and you've got a recipe
# for frustration. Even when traveling at high speeds, the cars can't whip around fast enough for you to line up an
# accurate shot. Since taking an extra second or two to line up a shot usually leads to the cars getting too close to
# your car, you'll end up damaging yourself with the close explosions. The game has a couple of two-player modes: a
# standard deathmatch mode and a pass-the-bomb mode. Neither mode is particularly entertaining, because the game
# doesn't have the solid arcade-style control of your average car combat game. 007 Racing looks decent but definitely
# fails to impress. There's a lot of texture warping, and most of the game's textures look extremely muddy. It's easy
# to forgive these flaws when you're driving at high speeds, but considering you spend quite a bit of the game moving
# at average or low speeds, you have plenty of time to marvel at the game's subpar graphics. The sound has its
# plusses and minuses as well. The Pierce Brosnan sound-alike does an admirable job of imitating the big-screen 007,
# and most of the game's voices are well done. However, the in-mission chidings from Q are extremely repetitive and
# frequent enough to make you want to simply turn the game's sound off altogether. The music, however, delivers a
# decent soundtrack of tunes appropriate for the Bond universe. The game's varied mission objectives occasionally
# give it a Driver-like feel, but the clunky control issues really manage to take you out of the game. The heavily
# modified Need for Speed engine is great for the fast action, fast driving missions, but the slower-paced,
# more combat-heavy levels suffer from the game's rough control. Overall, 007 Racing isn't polished enough to fill
# the needs of objective-based driving game fans. Fans of these types of games would be better served by Driver 2.",
# 'date': 'November 22, 2000',
# 'grade': None}]})
