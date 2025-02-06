import pandas as pd
import math
import sys
import time

def count_w(s):
    return s.count('W')

def preprocess(csv_file):
    df = pd.read_csv(csv_file)

    df['home_wins'] = df['home_wl_pre5'].apply(count_w)
    df['away_wins'] = df['away_wl_pre5'].apply(count_w)

    df['off_home'] = df['fg3_pct_home_avg5'] * df['ft_pct_home_avg5']
    df['off_away'] = df['fg3_pct_away_avg5'] * df['ft_pct_away_avg5']

    df = df.drop(columns=['team_abbreviation_home',
                                'team_abbreviation_away',
                                'season_type',
                                'home_wl_pre5',
                                'away_wl_pre5',
                                'min_avg5',
                              ])
    
    print(df.std()/df.mean())
    
    df = df.drop(columns=['fg3_pct_home_avg5',
                                'fg3_pct_away_avg5',
                                'ft_pct_home_avg5',
                                'ft_pct_away_avg5',
                                'pts_home_avg5',
                                'pts_away_avg5',
                                'ast_home_avg5',
                                'ast_away_avg5',
                                'tov_home_avg5',
                                'tov_away_avg5',
                                'pf_home_avg5',
                                'pf_away_avg5',
                                ])
    return df


def calc_mean_and_var(df) :
    means = df.mean()
    variances = df.var()

    stats = pd.DataFrame({'mean': means, 'variance': variances})
    return stats

def seperate_by_label(df):
    label = df['label']
    df = df.drop(columns=['label'])

    win = df[label == 1]
    loss = df[label == 0]

    return win, loss

def gaussian_likelihood( x, mean, var):
    return (1/math.sqrt(2*math.pi*(var))*math.exp(-(math.pow(x-mean, 2))/(2*(var))))
    
def predict(stats_w, stats_l, prior_w, test_df, testing):
    count_w = 0
    count_l = 0
    for row in test_df.itertuples(index=False):
        log_likelihood_w = math.log(prior_w)
        log_likelihood_l = math.log(1-prior_w)
        for col in test_df.columns:
            if col == 'label': continue

            log_likelihood_w += math.log(gaussian_likelihood(getattr(row, col), stats_w.at[col, 'mean'], stats_w.at[col, 'variance']))
            log_likelihood_l += math.log(gaussian_likelihood(getattr(row, col), stats_l.at[col, 'mean'], stats_l.at[col, 'variance']))
        
        if testing==False: print(str( 1 if log_likelihood_w > log_likelihood_l else 0))
        else:
            label = getattr(row, 'label')
            if label == 1 and log_likelihood_w > log_likelihood_l:
                count_w += 1
            elif label == 0 and log_likelihood_l > log_likelihood_w:
                count_l += 1
    
    if testing:
        print("accuracy:" + str((count_w + count_l) / test_df.shape[0]))


def NaiveBayesClassifier(train_path, test_path):
    train_df = preprocess(train_path)
    train_w, train_l = seperate_by_label(train_df)
    stats_w = calc_mean_and_var(train_w)
    stats_l = calc_mean_and_var(train_l)

    test_df = preprocess(test_path)
    prior_w = train_w.shape[0] / (train_w.shape[0] + train_l.shape[0])

    #predict(stats_w, stats_l, prior_w, test_df, False)
    predict(stats_w, stats_l, prior_w, test_df, True)  

if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    start = time.time()

    NaiveBayesClassifier(train_path, test_path)
    end = time.time()
    print(end-start)

    





