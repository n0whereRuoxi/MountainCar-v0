import re
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--log", help="directory of the log file",
                           type=str, default=None, required=False)
    args = argparser.parse_args()
    log_file_name = args.log
    idx = []
    scores = []
    scores_avg = []
    with open(log_file_name, "r") as log_file:
        line = log_file.readline()
        while line:
            line_ele = line.split('\t')
            idx.append(int(line_ele[0].split(' ')[1]))
            scores.append(float(line_ele[2].split(' ')[1]))
            scores_avg.append(float(line_ele[7].split(' ')[2]))
            line = log_file.readline()
    print(idx)
    print(scores)
    print(scores_avg)
    plt.plot(idx, scores, label='Reward obtained from each episode')
    plt.plot(idx, scores_avg, label='Average reward in the last 100 episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()