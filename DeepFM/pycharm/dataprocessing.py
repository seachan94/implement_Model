from lib import *
from pycharm import config
class Dataprocessing:
    def __init__(self, data, typeofnormalization = True,field_index = True):

        self.data = data
        self.processed_data = pd.DataFrame()
        self.field_index = field_index
        if typeofnormalization:
            self.scalar = MinMaxScaler()
        else:
            self.scalar = StandardScaler()
    def continues(self,cols):

        if cols is None:
            return pd.DataFrame()

        return self.scalar.fit_transform(self.data[cols])

    def discrete(self,cols):

        if cols is None:
            return pd.DataFrame()

        return pd.get_dummies(self.data[cols],prefix_sep='/')
    def checkcolumn(self):

        for col in self.data.columns:
            if col not in config.FIELDS:
                print("{} is not in field Column")
                raise ValueError

    def make_Modified_Field_Index(self,x_cate):
        config.Modified_Field_Index = [config.FIELDS.index(val) for val in config.CONTINUE_FIELD]

        idx = 0
        for col in x_cate:

            for realcol in config.Categorical_FIELD:

                if realcol == col.split('/')[0]:

                    config.Modified_Field_Index.append(config.FIELDS.index(realcol))
                    break

        config.Modified_Field_Index.sort()

    def run(self):

        #self.checkcolumn()
        x_continue = self.continues(config.CONTINUE_FIELD)
        x_category = self.discrete(config.Categorical_FIELD)



        if self.field_index:
            self.make_Modified_Field_Index(x_category.columns)


        return pd.concat((pd.DataFrame(x_continue,columns=config.CONTINUE_FIELD),x_category),axis = 1)


data = pd.read_csv(r'C:\Users\KTDS\Desktop\im\DeepFM\data\adult_csv.csv')

'''
config.FIELDS = list(data.columns)

config.CONTINUE_FIELD = ['age', 'fnlwgt', 'education-num',
               'capitalgain', 'capitalloss', 'hoursperweek']
config.Categorical_FIELD = list(set(config.FIELDS).difference(config.CONTINUE_FIELD))

cls = Dataprocessing(data,False)
a = cls.run()

CONTINUE_FIELD = ["a","b","c"]
Categorical_FIELD = ["d","e","f"]
FIELDS = CONTINUE_FIELD+Categorical_FIELD

col = ["a","b","c","d","e","f"]

da = [[1,2,3,"q","w","e"],[4,5,6,"r","t","y"],[7,8,9,"u","i","o"]]

f = pd.DataFrame(da,columns=col)


cls = Dataprocessing(f,False)

a = cls.run()
print(a)


'''
