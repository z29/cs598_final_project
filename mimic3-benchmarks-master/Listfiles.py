import glob, os
os.chdir("C:\Users\Bhushan\Desktop\All folders\university of illinois\CS598-DeepLearningForHealthCare\Project\ConCare\data\demographic")
f = open("train_list.csv", "w")
for file in glob.glob("*.csv"):
    print(file)