import pandas as pd

train = pd.read_csv('morpheme_train.csv')
val = pd.read_csv('morpheme_validation_first_meaning.csv')

# confirm = val['filename'] == 'NIA_SL_SEN0012_REAL09_L.mp4'
# new = val[confirm]
# print(new)

# hi = train[train['meaning']=='말해주다']
# hi_ = hi[hi['filename'].str.contains('SEN')]
# print(hi_)

flag = 'SEN0025'

# WRD = val[val['filename'].str.contains(flag)]
WRD_ = train[train['filename'].str.contains(flag)]

print(WRD_)
# print('---------------------------------')
# print(WRD)

# REAL: 400,000 + 40,000 = 440,000 (60,000)
# CROWD: 17,000 + 3,000 = 20,000 (1,000)