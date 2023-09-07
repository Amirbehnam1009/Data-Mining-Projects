import pandas as pd
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from imblearn.combine import SMOTETomek, SMOTEENN

df = pd.read_csv(r'C:\Users\dr. Behnam\Documents\Penguins\penguins.csv' , names=['island', 'bill _length_mm', 'bill_depth_mm', 'lipper_length_mm', 'body_mass_g' , 'sex' , 'species'])
print(df.isna().sum())

#Q2 -Some Of This Block Of Code Is Commented Because In Other Steps We Wanted To Replace NaNs With The Most Frequent And The Mean
# If You Want To Properly Run This Take Line 17 Out Of Comment


number_of_rows_before_deletion = df.shape[0]
#df = df.dropna(how ='any')
number_of_rows_after_deletion = df.shape[0]
print(df.isna().sum())
print('Number of rows before deletion:', number_of_rows_before_deletion)
print('Number of rows after deletion:', number_of_rows_after_deletion)


#Question 3 - Replacing NaNs With The Mean For Integer Columns And The Most Frequent For String Columns

string_columns = df.select_dtypes(include=['object']).columns.tolist()
number_columns = df.select_dtypes(include=[pd.np.number]).columns.tolist()
imputer = SimpleImputer(strategy='mean')
imputer_str = SimpleImputer(strategy='most_frequent')
df[string_columns] = imputer_str.fit_transform(df[string_columns])
df[number_columns] = imputer.fit_transform(df[number_columns])
print(df.isnull().sum())
print(df)


#Q4 - Label Encoding


string_columns_for_label_encoding= ['sex', 'island', 'species']
le_sex = LabelEncoder()
le_island = LabelEncoder()
le_species = LabelEncoder()
df['sex'] = le_sex.fit_transform(df['sex'])
df['island'] = le_island.fit_transform(df['island'])
df['species'] = le_species.fit_transform(df['species'])
print(df[string_columns_for_label_encoding])


#Q12 - Box Plots
# If You Want To Create The BoxPlot For Any Of The 7 Columns Make Sure To Comment Everything Else Apart From The Column Of Your Choice

plt.boxplot(df['island'])
plt.boxplot(df['bill _length_mm'])
plt.boxplot(df['bill_depth_mm'])
plt.boxplot(df['lipper_length_mm'])
plt.boxplot(df['body_mass_g'])
plt.boxplot(df['sex'])
plt.boxplot(df['species'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('island')
plt.title('bill_lenth_mm')
plt.title('bill_depth_mm')
plt.title('lipper_length_mm')
plt.title('body_mass_g')
plt.title('sex')
plt.title('species')
plt.show()


#Q8 - Data Augmentation

print(f"Number of Rows Before The Process Of Removing Data: {df.shape[0]}")
# I Selected Equal To 1 Instead Of Chinstrap Because In The Previous Step We Label Encoded Each Class And Chinstrap Was Equal To 1
Rows_With_Chinstrap = df[df.species == 1].index.tolist()
Remove_Rows_With_Chinstrap = Rows_With_Chinstrap[:int(0.9 * len(Rows_With_Chinstrap))]
df.drop(Remove_Rows_With_Chinstrap , inplace=True)
print(f"Number of Rows After Removing 90% Of Rows Which Contains Chinstrap: {df.shape[0]}")
X = df.drop('species', axis=1)
y = df['species']
smote_tomek = SMOTETomek(sampling_strategy='auto')
X_resampled_1, y_resampled_1 = smote_tomek.fit_resample(X, y)
print(f"How The Csv File And The Data Looks After Handling And Balancing The Data With SMOTETomek: {X_resampled_1.shape}")
smote_enn = SMOTEENN(sampling_strategy='auto')
X_resampled_2, y_resampled_2 = smote_enn.fit_resample(X, y)
print(f"How The Csv File And The Data Looks After Handling And Balancing The Data With SMOTEENN:: {X_resampled_2.shape}")


#Q9  Normalization

scaler = StandardScaler()
scaler.fit(df.get(['island', 'bill _length_mm', 'bill_depth_mm', 'lipper_length_mm', 'body_mass_g' , 'sex']))
print("Variance Before " + str(scaler.scale_))
print("Mean Before :" + str(scaler.mean_))
scaled_data = scaler.transform(df.get(['island', 'bill _length_mm', 'bill_depth_mm', 'lipper_length_mm', 'body_mass_g' , 'sex']))
scaler.fit(scaled_data) 
print("Variance After  :" + str(scaler.scale_))
print("Mean After  :" + str(scaler.mean_))
df['island'] = scaled_data[:, 0]
df['bill _length_mm'] = scaled_data[:, 1]
df['bill_depth_mm'] = scaled_data[:, 2]
df['lipper_length_mm'] = scaled_data[:, 3]
df['body_mass_g'] = scaled_data[:, 4]
df['sex'] = scaled_data[:, 5]


#Question 10 - PCA

pca = PCA(n_components=3)
pca.fit(df.get(['island', 'bill _length_mm', 'bill_depth_mm', 'lipper_length_mm', 'body_mass_g' , 'sex']))
features = pca.transform(df.get(['island', 'bill _length_mm', 'bill_depth_mm', 'lipper_length_mm', 'body_mass_g' , 'sex']))
print(features)


# Question 11 - Visualization

fig = plt.figure()
colors = ['red', 'yellow', 'blue']
colormap = matplotlib.colors.ListedColormap(colors)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=df.species ,cmap=colormap )
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
plt.show()
