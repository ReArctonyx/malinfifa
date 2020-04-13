# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "F:\Kuliah\Semester 6\Pembelajaran Mesin\malinfifa\fifa20.csv"
names = ['id', 'url', 'short_name', 'long_name', 'age', 'dob', 'height','weight', 'nationality', 'club', 'overall','potential','value','wage', 'position',
        'foot_pref', 'reputation', 'weak_foot','skill_moves','work_rate','body_type','real_face','release_clause','tags','team_position','jersey_numb','pace',
        'shooting','passing','dribbling','defending','physic','gk_diving', 'gk_handling','gk_kicking','gk_reflex','gk_speed','gk_positioning','traits','crossing',
        'finisihing', 'heading', 'short_pass','volleys','skill_dribbling','skill_curve','accuracy','long_pass','ball_control','acceleration','sprint','agility','reactions',
        'movement_balance','shot_power','jumping','sta','str','long_shots','aggression','interception','positioning','vision','penalty','marking','sliding','gk_handling','gk_position',
        'gk_reflexes','ls','st','rs','lw','lf','cf','rf','rw','lam','cam','ram','lm','lcm','cm','rcm','nm','lwb','ldm','cdm','rdm','rwb','lb','lcb','cb','rcb','rb']
dataset = read_csv(url, names=names)

# head (mengecek 20 data terdepan)
print(dataset.head(20))
# descriptions (count,mean,min,max)
print(dataset.describe())
# class distribution (mengecek row masing2 kelas)
print(dataset.groupby('class').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()