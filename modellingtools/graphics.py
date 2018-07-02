import matplotlib.pyplot as plt
import pandas as pd

def plot_emblem(data, var_targ, ytarg, pred):
    df = data.copy()
    df['pred'] = pred
    df['target'] = ytarg
    if len(set(df[var_targ])) > 20:
        df[var_targ] = pd.cut(df[var_targ], 20)

    gb = df.groupby(var_targ)
    counts = gb.size().to_frame(name='counts')
    gb = (counts
          .join(gb.agg({'target': 'mean'}).rename(columns={'target': 'actual'}))
          .join(gb.agg({'pred': 'mean'}).rename(columns={'pred': 'predict'}))
          .reset_index()
          )
    var_x = np.arange(len(gb[var_targ]))
    fig, ax = plt.subplots()
    plt.bar(var_x, gb.counts)
    plt.xticks(var_x, gb[var_targ])
    ax.grid(False)
    ax2 = ax.twinx()
    plt.title(var_targ)
    plt.plot(var_x, gb.actual, color="red", marker='o')
    plt.plot(var_x, gb.predict, color="green", marker='o')
    ax2.grid(False)
    plt.tight_layout()
    fig.autofmt_xdate()
    plt.xlabel(var_targ)


def lift_curve(ytarg, pred):
    data = (pd.DataFrame({"actual": ytarg, "model": pred})
            .groupby(pd.cut(pred, np.percentile(a=pred, q=np.arange(0, 100, 5))))
            .agg(['mean', 'count']))

    fig, ax = plt.subplots()
    ax.grid(False)
    plt.title("WIP")
    plt.plot(data.index.categories.astype(str), data['actual']['mean'], color="red", marker='o')
    plt.plot(data.index.categories.astype(str), data['model']['mean'], color="green", marker='o')
    plt.tight_layout()
    fig.autofmt_xdate()
    plt.legend(('Actual', 'Model'))
    # plt.xlabel(var_targ)
