"""A small set of real five-letter words used only by the unit tests.

The tests never depend on the downloaded Wordle lists, so they run anywhere.
It includes words with repeated letters (geese, speed, sweep, llama, ...) because
those are where feedback bugs usually hide.
"""

_TEXT = """
about abuse actor acute admit adopt adult after again agent agree ahead alarm album alert
alike alive allow alone along alter among anger angle angry apart apple apply arena argue
arise armor array aside asset avoid awake award aware badly baker bases basic beach began
begin being below bench birth black blame blank blast blind block blood board boost booth
bound brain brand bread break breed brief bring broad broke brown build built buyer cable
carry catch cause chain chair chart chase cheap check chest chief child chose civil claim
class clean clear click clock close coach coast could count court cover craft crash cream
crime cross crowd crown curve cycle daily dance dealt death debut delay depth doing doubt
dozen draft drama drawn dream dress drill drink drive drove dying eager early earth eight
elite empty enemy enjoy enter entry equal error event every exact exist extra faith false
fault fiber field fifth fifty fight final first fixed flash fleet floor fluid focus force
forth forty forum found frame frank fraud fresh front fruit fully funny geese ghost giant
given glass globe going grace grade grand grant grass great green gross group grown guard
guess guest guide happy heart heavy hence horse hotel house human hurry image index inner
input issue joint judge known label large laser later laugh layer learn lease least leave
legal level light limit lives local logic loose lower lucky lunch lying magic major maker
march match maybe mayor meant media metal might minor minus mixed model money month moral
motor mount mouse mouth movie music needs never newly night noise north noted novel nurse
occur ocean offer often order other ought paint panel paper party peace phase phone photo
piece pilot pitch place plain plane plant plate point pound power press price pride prime
print prior prize proof proud prove queen quick quiet quite radio raise range rapid ratio
reach ready refer right rival river robin rough round route royal rural scale scene scope
score sense serve seven shall shape share sharp sheet shelf shell shift shirt shock shoot
short shown sight since sixth sixty sized skill sleep slide small smart smile smoke solid
solve sorry sound south space spare speak speed spend spent split spoke sport staff stage
stake stand start state steam steel stick still stock stone stood store storm story strip
stuck study stuff style sugar suite super sweet table taken taste teach teeth thank theft
their theme there these thick thing think third those three threw throw tight times tired
title today topic total touch tough tower track trade train treat trend trial tried tries
truck truly trust truth twice under union unity until upper upset urban usage usual valid
value video virus visit vital voice waste watch water wheel where which while white whole
whose woman women world worry worse worst worth would wound write wrong wrote yield young
youth abide eerie llama sheep sweep steep creep sissy gloss mamma added array
"""

WORDS = sorted({w for w in _TEXT.split() if len(w) == 5 and w.isalpha()})
# the first 3/4 act as hidden answers, all of them are accepted as guesses
ANSWERS = WORDS[: (len(WORDS) * 3) // 4]
ALLOWED = WORDS
