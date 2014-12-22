#Script to extract features from chess game data file data.pgn

import pandas as pd
import re

chess_df = pd.read_csv("score_features2.csv")

raw_game_data = open("data.pgn").read()

game_moves = raw_game_data.split("\n")
formatted_games = []

#Extracts game moves for each game
game = []
active_game = False
for m in game_moves:
    if "1." in m:
        game.append(m)
        active_game = True
    elif 'Event' in m:
        if game != []:
            formatted_games.append(" ".join(game))
            game = []
            active_game = False
    elif active_game:
        game.append(m)
formatted_games.append(" ".join(game))

white_total_checks = []
black_total_checks =[]
white_queen_moves_avg = []
black_queen_moves_avg = []
white_rook_moves_avg = []
black_rook_moves_avg = []
white_bishop_moves_avg = []
black_bishop_moves_avg = []
white_king_moves_avg = []
black_king_moves_avg = []
white_knight_moves_avg = []
black_knight_moves_avg = []
white_castle_turn_num = []
black_castle_turn_num = []
white_castle_side = []
black_castle_side = []
white_promotions = []
black_promotions = []
white_check_rate = []
black_check_rate = []

for g in formatted_games:
    moves = re.findall(' [a-zA-Z][^ ]+',g)
    game_len = len(moves)
    turn = 1
    white_checks = 0
    black_checks = 0
    white_queen_moves = 0
    black_queen_moves = 0
    white_rook_moves= 0
    black_rook_moves= 0
    white_bishop_moves= 0
    black_bishop_moves= 0
    white_king_moves= 0
    black_king_moves= 0
    white_knight_moves= 0
    black_knight_moves= 0
    white_castle_turn = "NoCastle"
    black_castle_turn = "NoCastle"
    white_castle_type = "NoCastle"
    black_castle_type = "NoCastle"
    white_promotion= 0
    black_promotion = 0
    player = 0
    for m in moves:
        if player == 0:
            if "+" in m:
                white_checks+=1
            if "=" in m:
                white_promotion+=1
            if "Q" in m:
                white_queen_moves+=1
            elif "R" in m:
                white_rook_moves+=1
            elif "B" in m:
                white_bishop_moves+=1
            elif "K" in m:
                white_king_moves+=1
            elif "N" in m:
                white_knight_moves+=1
            if "O" in m:
                white_castle_turn = turn
                if "O-O-O" in m:
                    white_castle_type = "QueenSide"
                else:
                    white_castle_type = "KingSide"
        if player == 1:
            if "+" in m:
                black_checks+=1
            if "=" in m:
                black_promotion+=1
            if "Q" in m:
                black_queen_moves+=1
            elif "R" in m:
                black_rook_moves+=1
            elif "B" in m:
                black_bishop_moves+=1
            elif "K" in m:
                black_king_moves+=1
            elif "N" in m:
                black_knight_moves+=1
            if "O" in m:
                black_castle_turn = turn
                if "O-O-O" in m:
                    black_castle_type = "QueenSide"
                else:
                    black_castle_type = "KingSide"
        turn+=1
        player= abs(player-1)
    white_total_checks.append(white_checks)
    black_total_checks.append(black_checks)
    white_check_rate.append(white_checks/game_len)
    black_check_rate.append(black_checks/game_len)
    white_queen_moves_avg.append(white_queen_moves/game_len)
    black_queen_moves_avg.append(black_queen_moves/game_len)
    white_rook_moves_avg.append(white_rook_moves/game_len)
    black_rook_moves_avg.append(black_rook_moves/game_len)
    white_bishop_moves_avg.append(white_bishop_moves/game_len)
    black_bishop_moves_avg.append(black_bishop_moves/game_len)
    white_king_moves_avg.append(white_king_moves/game_len)
    black_king_moves_avg.append(black_king_moves/game_len)
    white_knight_moves_avg.append(white_knight_moves/game_len)
    black_knight_moves_avg.append(black_knight_moves/game_len)
    white_castle_turn_num.append(white_castle_turn)
    black_castle_turn_num.append(black_castle_turn)
    white_castle_side.append(white_castle_type)
    black_castle_side.append(black_castle_type)
    white_promotions.append(white_promotion)
    black_promotions.append(black_promotion)




chess_df["white_total_checks"] = white_total_checks
chess_df["white_queen_moves_avg"] = white_queen_moves_avg
chess_df["white_bishop_moves_avg"] = white_bishop_moves_avg
chess_df["white_king_moves_avg"] = white_king_moves_avg
chess_df["white_knight_moves_avg"] = white_knight_moves_avg
chess_df["white_rook_moves_avg"] = white_rook_moves_avg
chess_df["white_castle_turn_num"] = white_castle_turn_num
chess_df["white_castle_side"] = white_castle_side
chess_df["white_promotions"] = white_promotions
chess_df["white_check_rate"] = white_check_rate

chess_df["black_total_checks"] = black_total_checks
chess_df["black_queen_moves_avg"] = black_queen_moves_avg
chess_df["black_bishop_moves_avg"] = black_bishop_moves_avg
chess_df["black_king_moves_avg"] = black_king_moves_avg
chess_df["black_knight_moves_avg"] = black_knight_moves_avg
chess_df["black_rook_moves_avg"] = black_rook_moves_avg
chess_df["black_castle_turn_num"] = black_castle_turn_num
chess_df["black_castle_side"] = black_castle_side
chess_df["black_promotions"] = black_promotions
chess_df["black_check_rate"] = black_check_rate

chess_df.to_csv("score_features3.csv")





