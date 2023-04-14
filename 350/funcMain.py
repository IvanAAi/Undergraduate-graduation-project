import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import math
from scipy import stats
import statistics as st

import operator

# print('请输入你需要推荐的id:')

# aa = input()
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-aa", help="id")

args = parser.parse_args()
aa = args.aa
print("id",aa)
steam_raw = pd.read_csv('KNN-steam-200k-cleaned.csv', header=None,
                        names=["User_ID", "Game", "Interaction", "Hours", "Ignore"])

steam_raw = steam_raw.drop("Ignore", axis=1)
steam1 = steam_raw
# clean up the table so that it only shows the hours put into each game
# if a user has purchased but not played the game, the hours will be 0
steam1 = steam_raw[steam_raw['Interaction'] == "purchase"]
steam2 = steam_raw[steam_raw['Interaction'] == "play"]
steam3 = pd.merge(steam1, steam2, how='left', left_on=['User_ID', 'Game'], right_on=['User_ID', 'Game'])
steam3['Hours_y'] = steam3['Hours_y'].fillna(0)

# put it all into a clean table
steam_clean = steam3.drop(['Interaction_x', 'Interaction_y', 'Hours_x'], axis=1)
steam_clean.head()
steam_clean.info()

print("Max " + str(steam_clean['Hours_y'].max())
      + "\nMean " + str(steam_clean['Hours_y'].mean())
      + "\nMedian " + str(steam_clean['Hours_y'].median())
      + "\nMode " + str(steam_clean['Hours_y'].mode()))



#user_hours is a list of all the hours that each user put into the game we are calculating the frequency for
#hours_i is the hours for that specific user for the game
def frequency(hours_i, user_hours):
    if user_hours == 0:
        return 0
    elif hours_i == user_hours:
        return 1
    return hours_i/(user_hours - hours_i)

#frequency_list is the list of all the frequencies between the user k and the top user
#frequency is the calculated frequency for that user
def rating(frequency_sum):
    return 4*(1-frequency_sum)+1



#print(steam_clean2.iloc[steam_length-1])
#print(steam_length)
b = 0
for b in range(0, 2):
    print(b)

#first we need to create a table with all of the games and the total hours put into each
game_hours = steam_clean.groupby(['Game'])['Hours_y'].sum().reset_index()

#now we can use this information in the frequency function
steam_clean2 = steam_clean
steam_length = int(len(steam_clean2.index))
i = 0
steam_clean2['Frequency'] = np.nan
for i in range(0, steam_length):
    hours_i = steam_clean2.iloc[i][2]
    user_hours = game_hours[game_hours['Game'] == steam_clean2.iloc[i][1]].iloc[0][1] #- steam_clean2.iloc[i][2]
#    print(hours_i)
#    print(user_hours)
#    steam_clean2.iloc[i]['Frequency'] = hours_i/user_hours
    steam_clean2.at[i, 'Frequency'] = frequency(hours_i, user_hours)
#    print(steam_clean2.iloc[i]['Frequency'])
    #steam_clean2.iloc[i]['Frequency'] = frequency(hours_i, user_hours)
    #    steam_clean2.iloc[i]['Frequency'] = frequency(steam_clean2.iloc[i][2], game_hours[game_hours['Game'] == steam_clean2.iloc[i][1]].iloc[0][1] - steam_clean2.iloc[i][2])


# print(steam_clean.iloc[1][1])
# print(game_hours[game_hours['Game'] == 'Alan Wake'].iloc[0][1])
# print(game_hours)
steam_clean3 = steam_clean2

steam_clean3['Rating'] = np.nan
steam_clean3.sort_values(by=['Game', 'Frequency'], inplace=True, ascending=False, ignore_index=True)

print(steam_clean3)
# for i in range(0, steam_length):
# need a way to get all of the frequencies greater than the frequency for that user for that game at i

store = ""  # flag to see if we have moved on to a new game
sum_f = 0  # running frequency sum
last_f = 0  # flag to see if there are mutliple users with the same frequency
last_r = 0  # if multiple users with same frequency, want to give same rating

for i in range(0, steam_length):  # go through the entire dataframe
    temp = steam_clean3.iloc[i][1]  # find out which game we are one
    f_i = steam_clean3.iloc[i][3]  # find out the frequency for that game and user

    if temp != store:  # if it's the first time we are going over the game
        store = temp  # indicate we are now on a new game and save it's name
        if f_i == 0.0:  # if there are no hours for the game
            sum_f = 0  # reset running sum
            last_f = 0  # reset last frequency
            last_r = 1  # set last rating
            steam_clean3.at[i, 'Rating'] = 1  # if there are no hours, give it the lowest rating
        else:  # if there are hours for the game, must be the top user
            sum_f = f_i  # first time on this game so sum is the frequency
            last_f = f_i  # set last frequency
            last_r = 5  # set last rating
            steam_clean3.at[i, 'Rating'] = 5  # return the highest rating b/c top user

    else:  # it's not the first time on this game
        if f_i == 0:  # multiple users have zero hours
            last_r = 1  # set last rating
            steam_clean3.at[i, 'Rating'] = 1  # return lowest score; don't need to reset anything
        elif last_f == f_i:  # if there are multiple users with the same frequency
            sum_f += f_i  # add to running sum
            steam_clean3.at[i, 'Rating'] = last_r  # return last rating
        else:
            rating_f = rating(sum_f)  # calculate the rating
            sum_f += f_i  # update the sum
            last_f = f_i  # update the last frequency
            last_r = rating_f  # update the last rating
            steam_clean3.at[i, 'Rating'] = rating_f



steam_clean4 = steam_clean3
steam_clean4.sort_values(by = ['User_ID', 'Game'], inplace = True, ignore_index = True)
#remove the games with no hours played
steam_clean4 = steam_clean4[steam_clean4['Rating'] != 1]
steam_clean4=pd.DataFrame(steam_clean4)
print(steam_clean4)


# distance function; takes array q and p and calculates modified euclidean distance
def distance(q, p):
    total = 0
    for i in range(0, len(q)):
        total += (q[i] - p[i]) ** 2
    #    print(total)
    return math.sqrt(total) / len(q)


# find the k nearest neighbors
def neighbors(df, k_neighbors, user):
    distances = []
    # subset of the original table that contains only the rows for specific user
    user_games = df[df['User_ID'] == user]
    # subset of the original table minus the rows for specific user
    df_subset = df[df['User_ID'] != user]
    # temporary list to hold the ratings for the specific user
    user_temp = []
    # temporary list to hold the ratings for the user we are currently indexed on
    temp = []
    # flag to see if we have moved on to a new user id
    temp_id = 0
    # iterate through the entire subset
    for index, row in df_subset.iterrows():
        # if the game at that particual row is a game that the specific user has
        if row['Game'] in set(user_games['Game']):
            # if it is, check to see if we are on a new user or not
            if row['User_ID'] == temp_id:
                # if not, add the rating to the temp list
                temp.append(row['Rating'])
                # also add the rating to the user temp list
                user_temp.append(user_games.loc[user_games['Game'] == row['Game'], 'Rating'].iloc[0])
            # if it's the first time running the loop; set temp_id, add temp_id, game, and ratings
            # but do not calculate distance
            elif temp_id == 0:
                temp_id = row['User_ID']
                temp.append(row['Rating'])
                user_temp.append(user_games.loc[user_games['Game'] == row['Game'], 'Rating'].iloc[0])
            # not the first time running the loop
            # new user
            else:
                # calculate distance for previous user
                dist = distance(user_temp, temp)
                # add that to distances along with the id
                distances.append((temp_id, dist))
                # set the flag to the new id
                temp_id = row['User_ID']
                # reset temp and user_temp
                temp = []
                temp.append(row['Rating'])
                user_temp = []
                user_temp.append(user_games.loc[user_games['Game'] == row['Game'], 'Rating'].iloc[0])
    # once we finish for loop, sort distances so smallest are first
    distances.sort(key=operator.itemgetter(1))
    neighbor_list = []
    # insert neighbors into the list, smallest distance first up to the kth neighbor
    for i in range(k_neighbors):
        neighbor_list.append(distances[i])
    # return the list of k neighbors
    return neighbor_list


# recommend games based on the neighbors' ratings
def recommend(user, neighbor_list, df):
    # which games the user already has
    user_games = df[df['User_ID'] == user]
    dissim_games = []
    # go through all the neighbors
    for neighbor in neighbor_list:
        # make a temporary table containing all of the games that the neighbor has but the user does not
        temp = df[(df['User_ID'] == neighbor[0]) & (~df['Game'].isin(user_games['Game']))]
        # loop through the games in temp
        for index, game in temp.iterrows():
            # add the game and its rating to the dissimilar games list
            dissim_games.append((game['Game'], game['Rating']))
    # sort the dissimilar games list by the game name
    dissim_games.sort(key=operator.itemgetter(0))
    # flag to see if moved on to a new game
    flag = ""
    # running sum of all the ratings
    running_sum = 0
    # list we will add the recomendations to
    rec_list = []
    # count of how many times the game was in dissim_games
    count = 0
    # loop through all of the games
    for dis in dissim_games:
        # if it's the first time the game has come up in the loop
        if flag != dis[0]:
            # if it's not the first time the loop has run
            # if it was then we do not want to append anything
            if flag != "":
                # append the last game name and the average rating
                rec_list.append((flag, running_sum / count))
            # set the flag to the new gae
            flag = dis[0]
            # set the running sum to the current rating
            running_sum = dis[1]
            # reset the counter
            count = 1
        # multiple ratings for the same game
        else:
            # add the current rating to the running sum
            running_sum += dis[1]
            # increment the counter
            count += 1
    # sort the list of recommended games with the highest rating first
    sort_list = sorted(rec_list, key=operator.itemgetter(1), reverse=True)
    return (sort_list)

def rec_games(rec_tuple):
    games = []
    for pair in rec_tuple:
        games.append(pair[0])
    return games


aa = float(aa)
test_neighbors = neighbors(steam_clean4, 5, aa)
print(test_neighbors)

recs = recommend(aa, test_neighbors, steam_clean4)
recommended_games = rec_games(recs)




import random
from collections import Counter
from sklearn.metrics import roc_curve, auc, average_precision_score
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
path = 'steam-200k-cleaned.csv'
#path = 'steam-200k.csv'
df = pd.read_csv(path, header = None,
                 names = ['UserID', 'Game', 'Action', 'Hours', 'Other'])
df.head()

df['Hours_Played'] = df['Hours'].astype('float32')

df.loc[(df['Action'] == 'purchase') & (df['Hours'] == 1.0), 'Hours_Played'] = 0


df.UserID = df.UserID.astype('int')
df = df.sort_values(['UserID', 'Game', 'Hours_Played'])

clean_df = df.drop_duplicates(['UserID', 'Game'], keep = 'last').drop(['Action', 'Hours', 'Other'], axis = 1)

# every transaction is represented by only one record now
clean_df.head()



n_users = len(clean_df.UserID.unique())
n_games = len(clean_df.Game.unique())

print('There are {0} users and {1} games in the data'.format(n_users, n_games))

sparsity = clean_df.shape[0] / float(n_users * n_games)
print('{:.2%} of the user-item matrix is filled'.format(sparsity))
user_counter = Counter()
for user in clean_df.UserID.tolist():
    user_counter[user] +=1

game_counter = Counter()
for game in clean_df.Game.tolist():
    game_counter[game] += 1

user2idx = {user: i for i, user in enumerate(clean_df.UserID.unique())}
idx2user = {i: user for user, i in user2idx.items()}
print(user2idx)
print(idx2user)
game2idx = {game: i for i, game in enumerate(clean_df.Game.unique())}
idx2game = {i: game for game, i in game2idx.items()}
user_idx = clean_df['UserID'].apply(lambda x: user2idx[x]).values
game_idx = clean_df['gameIdx'] = clean_df['Game'].apply(lambda x: game2idx[x]).values
hours = clean_df['Hours_Played'].values


zero_matrix = np.zeros(shape = (n_users, n_games)) # Create a zero matrix
user_game_pref = zero_matrix.copy()
user_game_pref[user_idx, game_idx] = 1 # Fill preference matrix

user_game_interactions = zero_matrix.copy()
# Confidence matrix
user_game_interactions[user_idx, game_idx] = hours + 1


k = 10

# Count the number of purchases for each user
purchase_counts = np.apply_along_axis(np.bincount, 1, user_game_pref.astype(int))
buyers_idx = np.where(purchase_counts[:, 1] >= 2 * k)[0] #find the users who purchase 2 * k games
print('{0} users bought {1} or more games'.format(len(buyers_idx), 2 * k))



test_frac = 0.2 # Let's save 10% of the data for validation and 10% for testing.
test_users_idx = np.random.choice(buyers_idx,
                                  size = int(np.ceil(len(buyers_idx) * test_frac)),
                                  replace = False)



val_users_idx = test_users_idx[:int(len(test_users_idx) / 2)]
test_users_idx = test_users_idx[int(len(test_users_idx) / 2):]


def data_process(dat, train, test, user_idx, k):
    for user in user_idx:
        purchases = np.where(dat[user, :] == 1)[0]
        mask = np.random.choice(purchases, size=k, replace=False)

        train[user, mask] = 0
        test[user, mask] = dat[user, mask]
    return train, test



train_matrix = user_game_pref.copy()
test_matrix = zero_matrix.copy()
val_matrix = zero_matrix.copy()

# Mask the train matrix and create the validation and test matrices
train_matrix, val_matrix = data_process(user_game_pref, train_matrix, val_matrix, val_users_idx, k)
train_matrix, test_matrix = data_process(user_game_pref, train_matrix, test_matrix, test_users_idx, k)



test_matrix[test_users_idx[0], test_matrix[test_users_idx[0], :].nonzero()[0]]

train_matrix[test_users_idx[0], test_matrix[test_users_idx[0], :].nonzero()[0]]
tf.reset_default_graph() # Create a new graphs

pref = tf.placeholder(tf.float32, (n_users, n_games))  # Here's the preference matrix
interactions = tf.placeholder(tf.float32, (n_users, n_games)) # Here's the hours played matrix
users_idx = tf.placeholder(tf.int32, (None))

n_features = 30

# The X matrix represents the user latent preferences with a shape of user x latent features
X = tf.Variable(tf.truncated_normal([n_users, n_features], mean = 0, stddev = 0.05))

# The Y matrix represents the game latent features with a shape of game x latent features
Y = tf.Variable(tf.truncated_normal([n_games, n_features], mean = 0, stddev = 0.05))

# Here's the initilization of the confidence parameter
conf_alpha = tf.Variable(tf.random_uniform([1], 0, 1))

#user bias
user_bias = tf.Variable(tf.truncated_normal([n_users, 1], stddev = 0.2))

# Concatenate the vector to the user matrix
X_plus_bias = tf.concat([X,
                         #tf.convert_to_tensor(user_bias, dtype = tf.float32),
                         user_bias,
                         tf.ones((n_users, 1), dtype = tf.float32)], axis = 1)

# game bias
item_bias = tf.Variable(tf.truncated_normal([n_games, 1], stddev = 0.2))

# Cocatenate the vector to the game matrix
Y_plus_bias = tf.concat([Y,
                         tf.ones((n_games, 1), dtype = tf.float32),
                         item_bias],
                         axis = 1)



pred_pref = tf.matmul(X_plus_bias, Y_plus_bias, transpose_b=True)

# Construct the confidence matrix with the hours played and alpha paramter
conf = 1 + conf_alpha * interactions



cost = tf.reduce_sum(tf.multiply(conf, tf.square(tf.subtract(pref, pred_pref))))
l2_sqr = tf.nn.l2_loss(X) + tf.nn.l2_loss(Y) + tf.nn.l2_loss(user_bias) + tf.nn.l2_loss(item_bias)
lambda_c = 0.01
loss = cost + lambda_c * l2_sqr



lr = 0.05
optimize = tf.train.AdagradOptimizer(learning_rate = lr).minimize(loss)


# This is a function that helps to calculate the top k precision
def top_k_precision(pred, mat, k, user_idx):
    precisions = []

    for user in user_idx:
        rec = np.argsort(-pred[user, :])  # Found the top recommendation from the predictions

        top_k = rec[:k]
        labels = mat[user, :].nonzero()[0]

        precision = len(set(top_k) & set(labels)) / float(k)  # Calculate the precisions from actual labels
        precisions.append(precision)
    return np.mean(precisions)
iterations = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        sess.run(optimize, feed_dict = {pref: train_matrix,
                                        interactions: user_game_interactions})

        if i % 10 == 0:
            mod_loss = sess.run(loss, feed_dict = {pref: train_matrix,
                                                   interactions: user_game_interactions})
            mod_pred = pred_pref.eval()
            train_precision = top_k_precision(mod_pred, train_matrix, k, val_users_idx)
            val_precision = top_k_precision(mod_pred, val_matrix, k, val_users_idx)
            print('Iterations {0}...'.format(i),
                  'Training Loss {:.2f}...'.format(mod_loss),
                  'Train Precision {:.3f}...'.format(train_precision),
                  'Val Precision {:.3f}'.format(val_precision)
                )

    rec = pred_pref.eval()
    test_precision = top_k_precision(rec, test_matrix, k, test_users_idx)
    print('\n')
    print('Test Precision{:.3f}'.format(test_precision))

    n_examples = 1
    users = np.random.choice(test_users_idx, size=n_examples, replace=False)
    rec_games = np.argsort(-rec)

for user in users:
    print(user)
    print('用户id为 #{0} 的推荐结果为'.format(idx2user[user]))
    purchase_history = np.where(train_matrix[user, :] != 0)[0]
    recommendations = rec_games[user, :]

    new_recommendations = recommendations[~np.in1d(recommendations, purchase_history)][:k]

    print('推荐游戏列表')
    print(', '.join([idx2game[game] for game in new_recommendations]))
    print('\n')
    print('实际购买游戏列表')
    print(', '.join([idx2game[game] for game in np.where(test_matrix[user, :] != 0)[0]]))
    print('\n')
    print('Precision of {0}'.format(
        len(set(new_recommendations) & set(np.where(test_matrix[user, :] != 0)[0])) / float(k)))
    print('--------------------------------------')
    print('\n')
str = str(aa).format(idx2user[user])
str = int(float(str))
str = user2idx[str]
print(str)
recommendations = rec_games[str, :]
purchase_history = np.where(train_matrix[str, :] != 0)[0]
new_recommendations = recommendations[~np.in1d(recommendations, purchase_history)][:k]
print('KNN算法的推荐结果是：')
print(recommended_games)
print('SVD算法的结果是：')
print(', '.join([idx2game[game] for game in new_recommendations]))
with open("data.txt","w") as f:
    f.write(", ".join(recommended_games)+"\n")
    f.write(', '.join([idx2game[game] for game in new_recommendations])+"\n")