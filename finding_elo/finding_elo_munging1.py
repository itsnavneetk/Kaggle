#Script to extract features from chess score data file stockfish.csv
import numpy as np
import pandas as pd

#Load in and format raw chess game scoring data
raw_scores = [line.strip().split(",")[1].split() for line in open("stockfish.csv")][1:]

#Initialize containers for features to extract
game_length = []
average_score = []
score_stdev = []
largest_gain = []
largest_drop = []
max_score = []
min_score = []
ending_score = []
white_avg_improve = []
black_avg_improve = []
white_median_improve = []
black_median_improve = []
white_q1_improve =[]
white_q2_improve =[]
white_q3_improve =[]
white_q4_improve =[]
black_q1_improve =[]
black_q2_improve =[]
black_q3_improve =[]
black_q4_improve =[]

game_score10 = []
game_score20 = []
game_score30 = []
game_score40 = []
game_score50 = []
game_score60 = []
game_score70 = []
game_score80 = []
game_score90 = []
game_score100 = []

white_q1_max =[]
white_q2_max =[]
white_q3_max =[]
white_q4_max =[]
black_q1_max =[]
black_q2_max =[]
black_q3_max =[]
black_q4_max =[]

white_q1_min =[]
white_q2_min =[]
white_q3_min =[]
white_q4_min =[]
black_q1_min =[]
black_q2_min =[]
black_q3_min =[]
black_q4_min =[]

white_q1_stdev =[]
white_q2_stdev =[]
white_q3_stdev =[]
white_q4_stdev =[]
black_q1_stdev =[]
black_q2_stdev =[]
black_q3_stdev =[]
black_q4_stdev =[]

white_5_improve = []
white_10_improve = []
white_15_improve = []
white_20_improve = []
white_25_improve = []
white_30_improve = []
white_35_improve = []
white_40_improve = []
white_45_improve = []
white_50_improve = []
white_55_improve = []
white_60_improve = []
white_65_improve = []
white_70_improve = []
white_75_improve = []

black_5_improve = []
black_10_improve = []
black_15_improve = []
black_20_improve = []
black_25_improve = []
black_30_improve = []
black_35_improve = []
black_40_improve = []
black_45_improve = []
black_50_improve = []
black_55_improve = []
black_60_improve = []
black_65_improve = []
black_70_improve = []
black_75_improve = []


#Loop through game data, calculate and append new features to feature containers
for game in raw_scores:
    game_len = len(game)+1    # Add 1 to game length to avoid divide by zero errors caused by empty games
    total = 0
    prev = None
    player = 1
    max_so_far = -100
    min_so_far = 100
    max_drop = 0
    max_gain = 0
    white_improve = [0]
    black_improve = [0]
    game_nums = [0]
    for score in game:
        if score != "NA":
            score = int(score)
            game_nums.append(score)
            total+=score
            if prev != None:
                change = score - prev
                if change < max_drop:
                    max_drop = change
                if change > max_gain:
                    max_gain = change
                if player == 1:
                    black_improve.append(change)
                else:
                    white_improve.append(change)
                player = abs(player-1)
            prev = score
            if score > max_so_far:
                max_so_far = score
            if score < min_so_far:
                min_so_far = score

    #Add computed values to feature containers
    white_avg = sum(white_improve)/(game_len/2)
    black_avg = sum(black_improve)/(game_len/2)
    game_length.append(game_len)
    average_score.append(total/game_len)
    score_stdev.append(np.std(np.array(game_nums)))
    largest_gain.append(max_gain)
    largest_drop.append(max_drop)
    max_score.append(max_so_far)
    min_score.append(min_so_far)
    white_avg_improve.append(white_avg)
    black_avg_improve.append(black_avg)
    white_median_improve.append(sorted(white_improve)[len(white_improve)//2])
    black_median_improve.append(sorted(black_improve)[len(black_improve)//2])

    white_q1_improve.append( sum(white_improve[0:len(white_improve)//4])/len(white_improve)//4 )
    white_q2_improve.append( sum(white_improve[len(white_improve)//4 : (len(white_improve)//4)*2])/len(white_improve)//4 )
    white_q3_improve.append( sum(white_improve[(len(white_improve)//4)*2 : (len(white_improve)//4)*3])/len(white_improve)//4 )
    white_q4_improve.append( sum(white_improve[(len(white_improve)//4)*3 : ])/len(white_improve)//4 )
    black_q1_improve.append( sum(black_improve[0:len(black_improve)//4])/len(black_improve)//4 )
    black_q2_improve.append( sum(black_improve[len(black_improve)//4 : (len(black_improve)//4)*2])/len(black_improve)//4 )
    black_q3_improve.append( sum(black_improve[(len(black_improve)//4)*2 : (len(black_improve)//4)*3])/len(black_improve)//4 )
    black_q4_improve.append( sum(black_improve[(len(black_improve)//4)*3 : ])/len(black_improve)//4 )

    white_q1_max.append(max(white_improve[0:1+len(white_improve)//4]))
    white_q2_max.append(max(white_improve[len(white_improve)//4 : 1+(len(white_improve)//4)*2]))
    white_q3_max.append(max(white_improve[(len(white_improve)//4)*2 : 1+(len(white_improve)//4)*3]))
    white_q4_max.append(max(white_improve[(len(white_improve)//4)*3 : ]))
    black_q1_max.append(max(black_improve[0:1+len(black_improve)//4]))
    black_q2_max.append(max(black_improve[len(black_improve)//4 : 1+(len(black_improve)//4)*2]))
    black_q3_max.append(max(black_improve[(len(black_improve)//4)*2 : 1+(len(black_improve)//4)*3]))
    black_q4_max.append(max(black_improve[(len(black_improve)//4)*3 : ]))

    white_q1_min.append(min(white_improve[0:1+len(white_improve)//4]))
    white_q2_min.append(min(white_improve[len(white_improve)//4 : 1+(len(white_improve)//4)*2]))
    white_q3_min.append(min(white_improve[(len(white_improve)//4)*2 : 1+(len(white_improve)//4)*3]))
    white_q4_min.append(min(white_improve[(len(white_improve)//4)*3 : ]))
    black_q1_min.append(min(black_improve[0:1+len(black_improve)//4]))
    black_q2_min.append(min(black_improve[len(black_improve)//4 : 1+(len(black_improve)//4)*2]))
    black_q3_min.append(min(black_improve[(len(black_improve)//4)*2 : 1+(len(black_improve)//4)*3]))
    black_q4_min.append(min(black_improve[(len(black_improve)//4)*3 : ]))

    white_q1_stdev.append(np.std(np.array((white_improve[0:len(white_improve)//4]))))
    white_q2_stdev.append(np.std(np.array((white_improve[len(white_improve)//4 : (len(white_improve)//4)*2]))))
    white_q3_stdev.append(np.std(np.array((white_improve[(len(white_improve)//4)*2 : (len(white_improve)//4)*3]))))
    white_q4_stdev.append(np.std(np.array((white_improve[(len(white_improve)//4)*3 : ]))))
    black_q1_stdev.append(np.std(np.array((black_improve[0:len(black_improve)//4]))))
    black_q2_stdev.append(np.std(np.array((black_improve[len(black_improve)//4 : (len(black_improve)//4)*2]))))
    black_q3_stdev.append(np.std(np.array((black_improve[(len(black_improve)//4)*2 : (len(black_improve)//4)*3]))))
    black_q4_stdev.append(np.std(np.array((black_improve[(len(black_improve)//4)*3 : ]))))

    if len(white_improve) >=5:
        white_5_improve.append( sum(white_improve[0:5])/5 )
    else:
        white_5_improve.append(white_avg)
    if len(white_improve) >=10:
        white_10_improve.append( sum(white_improve[5:10])/5 )
    else:
        white_10_improve.append(white_avg)
    if len(white_improve) >=15:
        white_15_improve.append( sum(white_improve[10:15])/5 )
    else:
        white_15_improve.append(white_avg)
    if len(white_improve) >=20:
        white_20_improve.append( sum(white_improve[15:20])/5 )
    else:
        white_20_improve.append(white_avg)
    if len(white_improve) >=25:
        white_25_improve.append( sum(white_improve[20:25])/5 )
    else:
        white_25_improve.append(white_avg)
    if len(white_improve) >=30:
        white_30_improve.append( sum(white_improve[25:30])/5 )
    else:
        white_30_improve.append(white_avg)
    if len(white_improve) >=35:
        white_35_improve.append( sum(white_improve[30:35])/5 )
    else:
        white_35_improve.append(white_avg)
    if len(white_improve) >=40:
        white_40_improve.append( sum(white_improve[35:40])/5 )
    else:
        white_40_improve.append(white_avg)
    if len(white_improve) >=45:
        white_45_improve.append( sum(white_improve[40:45])/5 )
    else:
        white_45_improve.append(white_avg)
    if len(white_improve) >=50:
        white_50_improve.append( sum(white_improve[45:50])/5 )
    else:
        white_50_improve.append(white_avg)
    if len(white_improve) >=55:
        white_55_improve.append( sum(white_improve[50:55])/5 )
    else:
        white_55_improve.append(white_avg)
    if len(white_improve) >=60:
        white_60_improve.append( sum(white_improve[55:60])/5 )
    else:
        white_60_improve.append(white_avg)
    if len(white_improve) >=65:
        white_65_improve.append( sum(white_improve[60:65])/5 )
    else:
        white_65_improve.append(white_avg)
    if len(white_improve) >=70:
        white_70_improve.append( sum(white_improve[65:70])/5 )
    else:
        white_70_improve.append(white_avg)
    if len(white_improve) >=75:
        white_75_improve.append( sum(white_improve[70:75])/5 )
    else:
        white_75_improve.append(white_avg)

    if len(black_improve) >=5:
        black_5_improve.append( sum(black_improve[0:5])/5 )
    else:
        black_5_improve.append(black_avg)
    if len(black_improve) >=10:
        black_10_improve.append( sum(black_improve[5:10])/5 )
    else:
        black_10_improve.append(black_avg)
    if len(black_improve) >=15:
        black_15_improve.append( sum(black_improve[10:15])/5 )
    else:
        black_15_improve.append(black_avg)
    if len(black_improve) >=20:
        black_20_improve.append( sum(black_improve[15:20])/5 )
    else:
        black_20_improve.append(black_avg)
    if len(black_improve) >=25:
        black_25_improve.append( sum(black_improve[20:25])/5 )
    else:
        black_25_improve.append(black_avg)
    if len(black_improve) >=30:
        black_30_improve.append( sum(black_improve[25:30])/5 )
    else:
        black_30_improve.append(black_avg)
    if len(black_improve) >=35:
        black_35_improve.append( sum(black_improve[30:35])/5 )
    else:
        black_35_improve.append(black_avg)
    if len(black_improve) >=40:
        black_40_improve.append( sum(black_improve[35:40])/5 )
    else:
        black_40_improve.append(black_avg)
    if len(black_improve) >=45:
        black_45_improve.append( sum(black_improve[40:45])/5 )
    else:
        black_45_improve.append(black_avg)
    if len(black_improve) >=50:
        black_50_improve.append( sum(black_improve[45:50])/5 )
    else:
        black_50_improve.append(black_avg)
    if len(black_improve) >=55:
        black_55_improve.append( sum(black_improve[50:55])/5 )
    else:
        black_55_improve.append(black_avg)
    if len(black_improve) >=60:
        black_60_improve.append( sum(black_improve[55:60])/5 )
    else:
        black_60_improve.append(black_avg)
    if len(black_improve) >=65:
        black_65_improve.append( sum(black_improve[60:65])/5 )
    else:
        black_65_improve.append(black_avg)
    if len(black_improve) >=70:
        black_70_improve.append( sum(black_improve[65:70])/5 )
    else:
        black_70_improve.append(black_avg)
    if len(black_improve) >=75:
        black_75_improve.append( sum(black_improve[70:75])/5 )
    else:
        black_75_improve.append(black_avg)

    if len(game_nums)>10:
        game_score10.append(game_nums[10])
    else:
        game_score10.append(0)
    if len(game_nums)>20:
        game_score20.append(game_nums[20])
    else:
        game_score20.append(0)
    if len(game_nums)>30:
        game_score30.append(game_nums[30])
    else:
        game_score30.append(0)
    if len(game_nums)>40:
        game_score40.append(game_nums[40])
    else:
        game_score40.append(0)
    if len(game_nums)>50:
        game_score50.append(game_nums[50])
    else:
        game_score50.append(0)
    if len(game_nums)>60:
        game_score60.append(game_nums[60])
    else:
        game_score60.append(0)
    if len(game_nums)>70:
        game_score70.append(game_nums[70])
    else:
        game_score70.append(0)
    if len(game_nums)>80:
        game_score80.append(game_nums[80])
    else:
        game_score80.append(0)
    if len(game_nums)>90:
        game_score90.append(game_nums[90])
    else:
        game_score90.append(0)
    if len(game_nums)>100:
        game_score100.append(game_nums[100])
    else:
        game_score100.append(0)

    if prev:
        ending_score.append(prev)
    else:
        ending_score.append(0)

chess_dict = {"game_length":game_length,"average_score":average_score,"score_stdev":score_stdev,"largest_gain":largest_gain,
              "largest_drop":largest_drop,"max_score":max_score,"min_score":min_score,
              "ending_score":ending_score, "white_avg_improve":white_avg_improve,
              "black_avg_improve":black_avg_improve,"white_median_improve":white_median_improve,
              "black_median_improve":black_median_improve,"white_q1_improve":white_q1_improve,
              "white_q2_improve":white_q2_improve,
              "white_q3_improve":white_q3_improve,
              "white_q4_improve":white_q4_improve,"black_q1_improve":black_q1_improve,
              "black_q2_improve":black_q2_improve,
              "black_q3_improve":black_q3_improve,
              "black_q4_improve":black_q4_improve,
              'white_5_improve': white_5_improve,
                'white_10_improve': white_10_improve,
                'white_15_improve': white_15_improve,
                'white_20_improve': white_20_improve,
                'white_25_improve': white_25_improve,
                'white_30_improve': white_30_improve,
                'white_35_improve': white_35_improve,
                'white_40_improve': white_40_improve,
                'white_45_improve': white_45_improve,
                'white_50_improve': white_50_improve,
                'white_55_improve': white_55_improve,
                'white_60_improve': white_60_improve,
                'white_65_improve': white_65_improve,
                'white_70_improve': white_70_improve,
                'white_75_improve': white_75_improve,
                'black_5_improve': black_5_improve,
                'black_10_improve': black_10_improve,
                'black_15_improve': black_15_improve,
                'black_20_improve': black_20_improve,
                'black_25_improve': black_25_improve,
                'black_30_improve': black_30_improve,
                'black_35_improve': black_35_improve,
                'black_40_improve': black_40_improve,
                'black_45_improve': black_45_improve,
                'black_50_improve': black_50_improve,
                'black_55_improve': black_55_improve,
                'black_60_improve': black_60_improve,
                'black_65_improve': black_65_improve,
                'black_70_improve': black_70_improve,
                'black_75_improve': black_75_improve,

                'white_q1_max': white_q1_max,
                'white_q2_max': white_q2_max,
                'white_q3_max': white_q3_max,
                'white_q4_max': white_q4_max,
                'black_q1_max': black_q1_max,
                'black_q2_max': black_q2_max,
                'black_q3_max': black_q3_max,
                'black_q4_max': black_q4_max,

                'white_q1_min': white_q1_min,
                'white_q2_min': white_q2_min,
                'white_q3_min': white_q3_min,
                'white_q4_min': white_q4_min,
                'black_q1_min': black_q1_min,
                'black_q2_min': black_q2_min,
                'black_q3_min': black_q3_min,
                'black_q4_min': black_q4_min,

                'white_q1_stdev': white_q1_stdev,
                'white_q2_stdev': white_q2_stdev,
                'white_q3_stdev': white_q3_stdev,
                'white_q4_stdev': white_q4_stdev,
                'black_q1_stdev': black_q1_stdev,
                'black_q2_stdev': black_q2_stdev,
                'black_q3_stdev': black_q3_stdev,
                'black_q4_stdev': black_q4_stdev,

                'game_score10':game_score10,
                'game_score20':game_score20,
                'game_score30':game_score30,
                'game_score40':game_score40,
                'game_score50':game_score50,
                'game_score60':game_score60,
                'game_score70':game_score70,
                'game_score80':game_score80,
                'game_score90':game_score90,
                'game_score100':game_score100
}

#Create feature data frame
chess_df = pd.DataFrame(chess_dict, index=[x for x in range(1,50001)])
chess_df.index.name = "Event"

#Write the new feature data frame to CSV
chess_df.to_csv("score_features.csv")

