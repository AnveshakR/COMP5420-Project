import pandas as pd

file1 = pd.read_csv('/Users/sameedkhan/Desktop/NLP/Project/COMP5420-project/Reddit API/superbowl_53_PRE.csv')
file2 = pd.read_csv('/Users/sameedkhan/Desktop/NLP/Project/COMP5420-project/Reddit API/superbowl_53_FIRSTHALF.csv')
file3 = pd.read_csv('/Users/sameedkhan/Desktop/NLP/Project/COMP5420-project/Reddit API/superbowl_53_SECONDHALF.csv')
file4 = pd.read_csv('/Users/sameedkhan/Desktop/NLP/Project/COMP5420-project/Reddit API/superbowl_53_POST.csv')


#print('file1',file1.head())
#print('file2',file2.head())
#print('file3',file3.head())
#print('file4',file4.head())

file1['phase'] = '1'
file2['phase'] = '2'
file3['phase'] = '3'
file4['phase'] = '4'

dfmerged = pd.concat([file1,file2,file3,file4],ignore_index=True)


fav = 'Patriots'
underdog = 'Rams'
# key for fav: 0 means irrelevant fans, 1 means fan of underdog, 2 means fan of fav
dfmerged['fav'] = 0

# key for condition:
# 1 is the fav team winning a close match
# 2 is the fav team winning a one-sided match
# 3 is the underdog team winning a close match
# 4 is the underdog team winning a one-sided match

dfmerged['condition'] = 2

for ind in dfmerged.index:
    flair = str(dfmerged['flair'][ind])
    print(flair,type(flair),'flair')
    if flair:
        print('test')
        if 'Patriots' in flair:
            print('fav')
            dfmerged['fav'][ind] = 2
        elif 'Rams' in flair:
            print('underdog')
            dfmerged['fav'][ind] = 1

dfmerged.to_csv('LVI_merged_Condition2.csv')