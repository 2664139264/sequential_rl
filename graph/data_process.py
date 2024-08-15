import re
import os

pattern = re.compile(r"^Eval num_timesteps=\d+, episode_reward=(-?\d+\.\d*) \+/\- \d+\.\d+$")

def extract_episode_rewards(lines):
    results = list()
    for line in lines:
        result = re.match(pattern, line)
        if result is not None:
            results.append(float(result.group(1)))
            
    return max(results)
        

def record(record_dict, logfilename, avg_reward):
    
    if "CartPole" in logfilename:
        subkey = "CartPole"
    elif "Pendulum" in logfilename:
        subkey = "Pendulum"
    
    record_dict = record_dict[subkey]
    
    if "Dif" in logfilename:
        subkey = "Dif"
    elif "InvExp" in logfilename:
        subkey = "InvExp"
    elif "Exp" in logfilename:
        subkey = "Exp"
    else:
        subkey = "Agg"
    
    record_dict = record_dict[subkey]
    
    if "PPOLSTM" in logfilename:
        subkey = "PPO_LSTM"
    else:
        subkey = "PPO"
    
    record_dict = record_dict[subkey]
    
    logfilename = int(re.sub(r"^[a-zA-Z]+", "", logfilename).split(".")[0])
    record_dict[logfilename] = avg_reward

if __name__ == "__main__":
    
    record_dict = dict(
        CartPole= dict(
            Agg = dict(
                PPO = dict(), PPO_LSTM = dict()
            ),
            Dif = dict(
                PPO = dict(), PPO_LSTM = dict()
            ),
            Exp = dict(
                PPO = dict(), PPO_LSTM = dict()
            ),
            InvExp = dict(
                PPO = dict(), PPO_LSTM = dict()
            )           
        ),
        Pendulum= dict(
            Agg = dict(
                PPO = dict(), PPO_LSTM = dict()
            ),
            Dif = dict(
                PPO = dict(), PPO_LSTM = dict()
            ),
            Exp = dict(
                PPO = dict(), PPO_LSTM = dict()
            ),
            InvExp = dict(
                PPO = dict(), PPO_LSTM = dict()
            )   
        )
    )

    basedir = "experiment_log2"

    for logfilename in os.listdir(basedir):
        with open(f"{basedir}/{logfilename}", "r") as logfile:
            avg_reward = extract_episode_rewards(logfile.readlines())
        record(record_dict, logfilename, avg_reward)

    for env in record_dict:
        for wrp in record_dict[env]:
            for algo in record_dict[env][wrp]:
                if wrp in {"Exp", "InvExp"}:
                    record_dict[env][wrp][algo] = [
                        record_dict[env][wrp][algo][k] for k in range(2, 11, 2)
                    ]
                else:
                    record_dict[env][wrp][algo] = [
                        record_dict[env][wrp][algo][k] for k in range(6)
                    ]
    
    for env in record_dict:
        for wrp in record_dict[env]:
            for algo in record_dict[env][wrp]:
                if wrp in {"Exp", "InvExp"}:
                    record_dict[env][wrp][algo] = [
                        record_dict[env]["Agg"][algo][0]    
                    ] + record_dict[env][wrp][algo]    


    from pprint import pprint
    pprint(record_dict)