Load model from: https://huggingface.co/monologg/bert-base-cased-goemotions-original 

run load_model.py to take input csv in this case "test_input.csv" and it will output
a new csv called "test_output.csv" 

The py file goes through a for loop iternating each cell with the "text" header and
the output will have a new header "predicted_emotion" added to it.

The emotions will in represented in numbers like so:

0: admiration
1: amusement
2: anger
3: annoyance
4: approval
5: caring
6: confusion
7: curiosity
8: desire
9: disappointment
10: disapproval
11: disgust
12: embarrassment
13: excitement
14: fear
15: gratitude
16: grief
17: joy
18: love
19: nervousness
20: optimism
21: pride
22: realization
23: relief
24: remorse
25: sadness
26: surprise
27: neutral