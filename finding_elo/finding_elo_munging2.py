#Script to extract features from chess game data file data_uci.pgn

import pandas as pd
import re

chess_df = pd.read_csv("score_features.csv")

raw_game_data = open("data_uci.pgn").read()

formatted_result = []
result = re.findall( "Result[^\]]+", raw_game_data)
for r in result:
    if "1-0" in r:
        formatted_result.append("WhiteWin")
    elif "0-1" in r:
        formatted_result.append("BlackWin")
    else:
        formatted_result.append("Draw")

WhiteElo = re.findall( "\[WhiteElo.+\]", raw_game_data)
BlackElo = re.findall( "\[BlackElo.+\]", raw_game_data)
WhiteElo = [int(elo[-6:-2]) for elo in WhiteElo]
BlackElo = [int(elo[-6:-2]) for elo in BlackElo]
raw_moves = re.findall("^[a-z].+", raw_game_data, re.M)
opening_2_moves = []
opening_4_moves = []
opening_6_moves = []
opening_8_moves = []
opening_3_moves = []
opening_5_moves = []
opening_7_moves = []
opening_9_moves = []
opening_10_moves = []

opening_14_moves = []
opening_25_moves = []
opening_36_moves = []

opening_13_moves = []
opening_24_moves  = []
opening_35_moves  = []
opening_46_moves  = []

for game in raw_moves:
    if len (game) >= 9:
        opening_2_moves.append(game[0:9])
    else:
        opening_2_moves.append("Ended_Early")
    if len (game) >= 19:
        opening_4_moves.append(game[10:19])
    else:
        opening_4_moves.append("Ended_Early")
    if len (game) >= 29:
        opening_6_moves.append(game[20:29])
    else:
        opening_6_moves.append("Ended_Early")
    if len (game) >= 39:
        opening_8_moves.append(game[30:39])
    else:
        opening_8_moves.append("Ended_Early")
    if len (game) >= 49:
        opening_10_moves.append(game[40:49])
    else:
        opening_10_moves.append("Ended_Early")

    if len (game) >= 14:
        opening_3_moves.append(game[5:14])
    else:
        opening_3_moves.append("Ended_Early")
    if len (game) >= 24:
        opening_5_moves.append(game[15:24])
    else:
        opening_5_moves.append("Ended_Early")
    if len (game) >= 34:
        opening_7_moves.append(game[25:34])
    else:
        opening_7_moves.append("Ended_Early")
    if len (game) >= 44:
        opening_9_moves.append(game[35:44])
    else:
        opening_9_moves.append("Ended_Early")
    if len (game) >= 19:
        opening_14_moves.append(game[0:19])
    else:
        opening_14_moves.append("Ended_Early")
    if len (game) >= 24:
        opening_25_moves.append(game[5:24])
    else:
        opening_25_moves.append("Ended_Early")
    if len (game) >= 29:
        opening_36_moves.append(game[10:29])
    else:
        opening_36_moves.append("Ended_Early")

    if len (game) >= 14:
        opening_13_moves.append(game[0:14])
    else:
        opening_13_moves.append("Ended_Early")
    if len (game) >= 19:
        opening_24_moves.append(game[5:19])
    else:
        opening_24_moves.append("Ended_Early")
    if len (game) >= 24:
        opening_35_moves.append(game[10:24])
    else:
        opening_35_moves.append("Ended_Early")
    if len (game) >= 29:
        opening_46_moves.append(game[15:29])
    else:
        opening_46_moves.append("Ended_Early")




chess_df["result"] = formatted_result
chess_df["opening_2_moves"] = opening_2_moves
chess_df["opening_4_moves"] = opening_4_moves
chess_df["opening_6_moves"] = opening_6_moves
chess_df["opening_8_moves"] = opening_8_moves
chess_df["opening_3_moves"] = opening_3_moves
chess_df["opening_5_moves"] = opening_5_moves
chess_df["opening_7_moves"] = opening_7_moves
chess_df["opening_9_moves"] = opening_9_moves
chess_df["opening_10_moves"] = opening_10_moves

chess_df["opening_14_moves"] = opening_14_moves
chess_df["opening_25_moves"] = opening_25_moves
chess_df["opening_36_moves"] = opening_36_moves

chess_df["opening_13_moves"] = opening_13_moves
chess_df["opening_24_moves"] = opening_24_moves
chess_df["opening_35_moves"] = opening_35_moves
chess_df["opening_46_moves"] = opening_46_moves


chess_df.to_csv("score_features2.csv")


#Create a separate data frame for ELO values of first 25000 games (training set)
elo_dict = {"WhiteElo":WhiteElo,"BlackElo":BlackElo}
chess_ELO = pd.DataFrame(elo_dict, index=[x for x in range(1,25001)])
chess_ELO.index.name = "Event"

chess_ELO.to_csv("chess_ELO.csv")




