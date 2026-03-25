import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# load data
df = pd.read_csv('Theory_Simulation_Sparsification_Updated.csv')

# reorder the dataset (clear naming scheme! split over multi-index to work easier with it)
df = df.drop(columns=['Unnamed: 0', 'rate_lower001', 'rate_lower01'])
df = df.set_index('Num_satellites')
df.index.name = 'sats'

df = df.rename(mapper = {
        'size_data_tight_theory_all_sat_q1': (1, 'bound'),
        'size_data_tight_theory_all_sat01': (.1, 'bound'),
        'size_data_tight_theory_all_sat001': (.01, 'bound'),
        'TopK_wo_IA_rate_q1': (1, 'no IA'),
        'TopK_wo_IA_rate01': (.1, 'no IA'),
        'TopK_wo_IA_rate001': (.01, 'no IA'),
        'sim_q1': (1, 'simulated'),
        'rate_sim_main01': (.1, 'simulated'),
        'rate_sim_main001': (.01, 'simulated')

}, axis = 1)

df.columns = pd.MultiIndex.from_tuples(df, names = ['q', 'method'])
df = df.T.sort_index().T
print(df.columns)

# let's undo some normilzation
n = 7850 # number model parameters
q = np.array([0.1, 0.01]) # quantization rate
esize = 32 # float size

msize = np.ceil(np.log2(n)) # meta data size (extra storage for address fields)
effective_q = np.floor(q*n)/n

print(df)

for qq, eq in zip(q, effective_q):
    df[qq] /= eq * (esize+msize) / esize

#print(df)
# massage for easy plotting
df2 = df.T.unstack(level=[0,1]).T.reset_index(name = 'values')

# plot raw data
#sns.lineplot(data = df2, x = 'sats', y = 'values', hue = 'q', style = 'method', palette = 'muted').set_title('transmissions normalized to non IA')


#rp = sns.relplot(data = df2, x = 'sats', y = 'values', col = 'q', kind = 'line', hue = 'method', palette = 'muted')
#rp.fig.subplots_adjust(top=0.9) # adjust the Figure in rp
#rp.fig.suptitle('transmissions normalized to non IA')

# store data
tmp = df.copy()
tmp.columns = ['{}_{:g}'.format(m, q).replace(" ", "_") for q, m in df.columns.to_flat_index()]
col = [s for s in tmp.columns if s.startswith('no_IA')]
tmp = tmp.drop(col[1:], axis = 1).rename(columns={col[0]: "noIA"})
assert np.all(tmp['bound_1'] == tmp['simulated_1'])
tmp = tmp.drop('bound_1', axis = 1).rename(columns={'simulated_1': 'IA'})
tmp.to_csv('IA_vs_q.dat')

print(df)

# IA gain
tmp = (1-df.xs('simulated', level=1, axis=1) / df.xs('no IA', level = 1, axis = 1))*100
#tmp.plot(title = 'IA gain (relative)')



# quality of bound
tmp = abs(1-df.xs('bound', level=1, axis=1) / df.xs('simulated', level = 1, axis = 1))*100
tmp.plot(title = 'bound mismatch (relative)')


# total trasnmitted data data
plt.figure()
r = df.drop('bound', level = 1, axis = 1)
for qq, eq in zip(q, effective_q):
    r[qq] *= eq
r = r.apply(lambda x: x*esize*n if x.name[0] == 1 else x*(esize+msize)*n)

tmp = r.copy()
tmp.columns = ['{}_{:g}'.format(m, q).replace(" ", "_") for q, m in r.columns.to_flat_index()]
tmp.to_csv('totaldata_IA.dat')

r = r.T.unstack(level=[0,1]).T.reset_index(name = 'values')
sns.lineplot(data = r, x = 'sats', y = 'values', hue = 'q', style = 'method', palette = 'muted').set_title('total transmitted data per epoch [bits]')

plt.figure()
sns.lineplot(data = r.query('q != 1'), x = 'sats', y = 'values', hue = 'q', style = 'method', palette = 'muted').set_title('total transmitted data per epoch [bits]')

plt.figure()
sns.lineplot(data = r.query('q == .01'), x = 'sats', y = 'values', hue = 'q', style = 'method', palette = 'muted').set_title('total transmitted data per epoch [bits]')


plt.show()
#plt.show(block = False)
