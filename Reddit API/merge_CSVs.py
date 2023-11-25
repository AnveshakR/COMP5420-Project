import pandas as pd

file1 = pd.read_csv('/Users/sameedkhan/Desktop/NLP/Project/COMP5420-project/Reddit API/superbowl_LVI_PRE.csv')
file2 = pd.read_csv('/Users/sameedkhan/Desktop/NLP/Project/COMP5420-project/Reddit API/superbowl_LVI_FIRSTHALF.csv')
file3 = pd.read_csv('/Users/sameedkhan/Desktop/NLP/Project/COMP5420-project/Reddit API/superbowl_LV_SECONDHALF.csv')
file4 = pd.read_csv('/Users/sameedkhan/Desktop/NLP/Project/COMP5420-project/Reddit API/superbowl_LVI_POST.csv')


print('file1',file1.head())
print('file2',file2.head())
print('file3',file3.head())
print('file4',file4.head())

file1['phase'] = '1'
file2['phase'] = '2'
file3['phase'] = '3'
file4['phase'] = '4'

dfmerged = pd.concat([file1,file2,file3,file4],ignore_index=True)


fav = 'Rams'
underdog = 'Bengals'

# key for fav: 0 means irrelevant fans, 1 means fan of underdog, 2 means fan of
dfmerged['fav'] = 0
for ind in dfmerged.index:
    flair = str(dfmerged['flair'][ind])
    print(flair,type(flair),'flair')
    if flair:
        print('test')
        if 'Rams' in flair:
            print('fav')
            dfmerged['fav'][ind] = 2
        elif 'Bengals' in flair:
            print('underdog')
            dfmerged['fav'][ind] = 1

dfmerged.to_csv('LVI_merged_v2.csv')