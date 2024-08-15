import seaborn as sns
import matplotlib.pyplot as plt

pku = "#8c0000"
miku = "#39c5bb"
rin = "#ffa500" 
ren = "#ffe211"
ruka = "#ffc0cb"
kaito = "#0000ff"
meiko = "#d80000"

n = [0, 1, 2, 3, 4, 5]
lmd = [0, 1/5, 2/5, 3/5, 4/5, 5/5]

with open("gatherdata0.txt", "r", encoding="utf-16") as datafile:
    data = eval("".join(datafile.readlines()))

with open("gatherdata1.txt", "r", encoding="utf-16") as datafile:
    data1 = eval("".join(datafile.readlines()))

with open("gatherdata2.txt", "r", encoding="utf-16") as datafile:
    data2 = eval("".join(datafile.readlines()))

for env in data:
    for wrp in data[env]:
        for algo in data[env][wrp]:
            for i in range(len(data[env][wrp][algo])):
                data[env][wrp][algo][i] += data1[env][wrp][algo][i] + data2[env][wrp][algo][i]
                data[env][wrp][algo][i] /= 3

font = {'family' : 'Times New Roman',
        'size'   : 20}
plt.rc('font', **font)

sns.set_style("whitegrid")

# sns.lineplot(x=n, y=data["CartPole"]["Agg"]["PPO"], color=kaito, linewidth=2.0, marker="x", markersize=8, markeredgecolor=kaito, markeredgewidth=1.5, label='S^n, PPO')
# sns.lineplot(x=n, y=data["CartPole"]["Agg"]["PPO_LSTM"], color=ruka, linewidth=2.0, marker="s", markersize=8, markeredgecolor=ruka, markeredgewidth=1.5, label='S^n, PPO_LSTM')
# sns.lineplot(x=n, y=data["CartPole"]["Dif"]["PPO"], color=meiko, linewidth=2.0, marker="o", markersize=8, markeredgecolor=meiko, markeredgewidth=1.5, label='D^n, PPO')
# sns.lineplot(x=n, y=data["CartPole"]["Dif"]["PPO_LSTM"], color=miku, linewidth=2.0, marker="^", markersize=8, markeredgecolor=miku, markeredgewidth=1.5, label='D^n, PPO_LSTM')

# sns.lineplot(x=lmd, y=data["CartPole"]["Exp"]["PPO"], color=kaito, linewidth=2.0, marker="x", markersize=8, markeredgecolor=kaito, markeredgewidth=1.5, label='S_λ, PPO')
# sns.lineplot(x=lmd, y=data["CartPole"]["Exp"]["PPO_LSTM"], color=ruka, linewidth=2.0, marker="s", markersize=8, markeredgecolor=ruka, markeredgewidth=1.5, label='S_λ, PPO_LSTM')
# sns.lineplot(x=lmd, y=data["CartPole"]["InvExp"]["PPO"], color=meiko, linewidth=2.0, marker="o", markersize=8, markeredgecolor=meiko, markeredgewidth=1.5, label='D_λ, PPO')
# sns.lineplot(x=lmd, y=data["CartPole"]["InvExp"]["PPO_LSTM"], color=miku, linewidth=2.0, marker="^", markersize=8, markeredgecolor=miku, markeredgewidth=1.5, label='D_λ, PPO_LSTM')

# sns.lineplot(x=n, y=data["Pendulum"]["Agg"]["PPO"], color=kaito, linewidth=2.0, marker="x", markersize=8, markeredgecolor=kaito, markeredgewidth=1.5, label='S^n, PPO')
# sns.lineplot(x=n, y=data["Pendulum"]["Agg"]["PPO_LSTM"], color=ruka, linewidth=2.0, marker="s", markersize=8, markeredgecolor=ruka, markeredgewidth=1.5, label='S^n, PPO_LSTM')
# sns.lineplot(x=n, y=data["Pendulum"]["Dif"]["PPO"], color=meiko, linewidth=2.0, marker="o", markersize=8, markeredgecolor=meiko, markeredgewidth=1.5, label='D^n, PPO')
# sns.lineplot(x=n, y=data["Pendulum"]["Dif"]["PPO_LSTM"], color=miku, linewidth=2.0, marker="^", markersize=8, markeredgecolor=miku, markeredgewidth=1.5, label='D^n, PPO_LSTM')

sns.lineplot(x=lmd, y=data["Pendulum"]["Exp"]["PPO"], color=kaito, linewidth=2.0, marker="x", markersize=8, markeredgecolor=kaito, markeredgewidth=1.5, label='S_λ, PPO')
sns.lineplot(x=lmd, y=data["Pendulum"]["Exp"]["PPO_LSTM"], color=ruka, linewidth=2.0, marker="s", markersize=8, markeredgecolor=ruka, markeredgewidth=1.5, label='S_λ, PPO_LSTM')
sns.lineplot(x=lmd, y=data["Pendulum"]["InvExp"]["PPO"], color=meiko, linewidth=2.0, marker="o", markersize=8, markeredgecolor=meiko, markeredgewidth=1.5, label='D_λ, PPO')
sns.lineplot(x=lmd, y=data["Pendulum"]["InvExp"]["PPO_LSTM"], color=miku, linewidth=2.0, marker="^", markersize=8, markeredgecolor=miku, markeredgewidth=1.5, label='D_λ, PPO_LSTM')

# plt.title("CartPole-v1", fontweight='bold', fontsize=23)
# plt.title("CartPole-v1", fontweight='bold', fontsize=23)
# plt.title("Pendulum-v1", fontweight='bold', fontsize=23)
plt.title("Pendulum-v1", fontweight='bold', fontsize=23)

# plt.xlabel("n", fontsize=20)
# plt.xlabel("λ", fontsize=20)
# plt.xlabel("n", fontsize=20)
plt.xlabel("λ", fontsize=20)

plt.ylabel("Average Episode Reward", fontsize=20)

# plt.legend(loc='upper right', frameon=True, fontsize=17)
# plt.legend(loc='lower left', frameon=True, fontsize=17)
# plt.legend(loc='upper right', frameon=True, fontsize=17)
plt.legend(loc='lower left', frameon=True, fontsize=17)

# plt.xticks(fontsize=17, ticks=[0,1,2,3,4,5])
# plt.xticks(fontsize=17, ticks=[0,.2,.4,.6,.8,1])
# plt.xticks(fontsize=17, ticks=[0,1,2,3,4,5])
plt.xticks(fontsize=17, ticks=[0,.2,.4,.6,.8,1])

plt.yticks(fontsize=17)

# plt.xlim(0, 5.2)
# plt.ylim(0, 520)

# plt.xlim(0, 1.05)
# plt.ylim(0, 520)

# plt.xlim(0, 5.2)
# plt.ylim(-1400, 0)

plt.xlim(0, 1.05)
plt.ylim(-1400, 0)

for spine in plt.gca().spines.values():
    spine.set_edgecolor("#CCCCCC")
    spine.set_linewidth(1.5)

id = input("input id\n")
plt.savefig(f'lineplot{id}.png', dpi=300, bbox_inches='tight')
plt.show()